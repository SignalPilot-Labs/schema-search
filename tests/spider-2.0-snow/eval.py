"""Spider 2.0-Snow evaluation: Claude vanilla vs Claude + schema-search MCP."""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import anthropic
from dotenv import load_dotenv
from sqlalchemy import Engine
from tqdm import tqdm

from agent import run_agent
from constants import (
    CREDENTIAL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROLE,
    DEFAULT_WAREHOUSE,
    DOCUMENTS_DIR,
    ENV_PATH,
    JSONL_PATH,
    PROMPT_PATH,
    SNOWFLAKE_ACCOUNT,
    TOOLS_MCP,
    TOOLS_VANILLA,
)
from models import Instance, ModeStats
from schema_search.schema_search import SchemaSearch
from schema_search.utils.utils import create_engine_from_url
from stats import print_summary

logger = logging.getLogger(__name__)

MODE_TOOLS = {
    "vanilla": TOOLS_VANILLA,
    "mcp": TOOLS_MCP,
}


def load_instances(jsonl_path: Path) -> List[Instance]:
    """Load instances from spider2-snow.jsonl."""
    instances = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line.strip())
            instances.append(
                Instance(
                    instance_id=item["instance_id"],
                    instruction=item["instruction"],
                    db_id=item["db_id"],
                    external_knowledge=item.get("external_knowledge"),
                )
            )
    return instances


def group_by_db_id(instances: List[Instance]) -> Dict[str, List[Instance]]:
    """Group instances by db_id to minimize re-indexing."""
    groups = defaultdict(list)
    for inst in instances:
        groups[inst.db_id].append(inst)
    return dict(groups)


def load_snowflake_credential(credential_path: Path) -> dict:
    """Load Snowflake credentials from JSON file."""
    if not credential_path.exists():
        print(f"ERROR: {credential_path} not found.")
        print("Copy from Spider2 guideline and fill in your username/password:")
        print(
            f'  {{"username": "...", "password": "...", "account": "{SNOWFLAKE_ACCOUNT}",'
            f' "role": "{DEFAULT_ROLE}", "warehouse": "{DEFAULT_WAREHOUSE}"}}'
        )
        sys.exit(1)
    with open(credential_path) as f:
        return json.load(f)


def build_snowflake_url(credential: dict, target_db: str) -> str:
    """Build a Snowflake SQLAlchemy URL from credentials and target database."""
    user = credential["username"]
    password = quote_plus(credential["password"])
    account = credential["account"]
    warehouse = credential.get("warehouse", DEFAULT_WAREHOUSE)
    role = credential.get("role", DEFAULT_ROLE)
    return (
        f"snowflake://{user}:{password}@{account}/{target_db}"
        f"?warehouse={warehouse}&role={role}"
    )


def load_external_knowledge(filename: Optional[str]) -> Optional[str]:
    """Load external knowledge document content."""
    if not filename:
        return None
    doc_path = DOCUMENTS_DIR / filename
    if not doc_path.exists():
        logger.warning("External knowledge file not found: %s", doc_path)
        return None
    return doc_path.read_text()


def load_system_prompt() -> str:
    """Load the system prompt from prompt.md."""
    return PROMPT_PATH.read_text()


def save_sql(sql: str, instance_id: str, output_dir: Path) -> None:
    """Save generated SQL to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{instance_id}.sql").write_text(sql)


def _instance_already_done(instance_id: str, mode: str, output_dir: Path) -> bool:
    """Check if an instance result already exists (for resumability)."""
    sql_path = output_dir / mode / f"{instance_id}.sql"
    if not sql_path.exists():
        return False
    content = sql_path.read_text().strip()
    return len(content) > 0


def _load_api_key() -> str:
    """Load and validate the Anthropic API key from .env."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("ERROR: LLM_API_KEY not set in tests/.env")
        sys.exit(1)
    if "=" in api_key:
        api_key = api_key.split("=", 1)[1]
    return api_key


def _filter_instances(
    instances: List[Instance], args: argparse.Namespace
) -> List[Instance]:
    """Apply db_filter and limit to instance list."""
    if args.db_filter:
        filter_set = set(args.db_filter)
        instances = [i for i in instances if i.db_id in filter_set]
        print(f"Filtered to {len(instances)} instances for db_ids: {args.db_filter}")
    if args.limit:
        instances = instances[: args.limit]
        print(f"Limited to {args.limit} instances")
    return instances


def _index_database(
    credential: dict, db_id: str
) -> tuple[Engine, SchemaSearch]:
    """Connect to Snowflake and index a database's schema.

    Returns:
        Tuple of (engine, search_engine).

    Raises:
        Exception: If connection or indexing fails.
    """
    snowflake_url = build_snowflake_url(credential, db_id)
    engine = create_engine_from_url(snowflake_url)
    search_engine = SchemaSearch(engine)
    t0 = time.time()
    search_engine.index(force=False)
    tqdm.write(f"  Indexed {db_id} in {time.time() - t0:.1f}s")
    return engine, search_engine


def _record_failure(
    instance_id: str, modes: List[str], mode_stats: Dict[str, ModeStats],
    output_dir: Path,
) -> None:
    """Record a failed instance across all modes."""
    for mode in modes:
        save_sql("", instance_id, output_dir / mode)
        mode_stats[mode].record_failure(instance_id)


def _run_instance_mode(
    inst: Instance, mode: str, search_engine: SchemaSearch,
    system_prompt: str, client: anthropic.Anthropic, mode_stats: Dict[str, ModeStats],
    output_dir: Path,
) -> None:
    """Run a single instance in a single mode and record results."""
    if _instance_already_done(inst.instance_id, mode, output_dir):
        mode_stats[mode].record_success(inst.instance_id, 0, 0.0)
        tqdm.write(f"  [{mode}] {inst.instance_id}: SKIP (already exists)")
        return

    ext_knowledge = load_external_knowledge(inst.external_knowledge)
    try:
        t0 = time.time()
        result = run_agent(
            instance_id=inst.instance_id,
            instruction=inst.instruction,
            external_knowledge=ext_knowledge,
            search_engine=search_engine,
            system_prompt=system_prompt,
            client=client,
            db_id=inst.db_id,
            tools=MODE_TOOLS[mode],
        )
        latency = time.time() - t0
        save_sql(result.sql, inst.instance_id, output_dir / mode)

        if result.error:
            mode_stats[mode].record_failure(inst.instance_id)
            status = f"FAIL ({result.error})"
        else:
            mode_stats[mode].record_success(
                inst.instance_id, result.tool_calls_count, latency
            )
            status = f"OK (tools={result.tool_calls_count}, {latency:.1f}s)"

        tqdm.write(f"  [{mode}] {inst.instance_id}: {status}")

    except Exception as e:
        save_sql("", inst.instance_id, output_dir / mode)
        mode_stats[mode].record_failure(inst.instance_id)
        tqdm.write(f"  [{mode}] {inst.instance_id}: ERROR {e}")


def run_evaluation(args: argparse.Namespace) -> None:
    """Run the full evaluation pipeline."""
    api_key = _load_api_key()
    credential = load_snowflake_credential(CREDENTIAL_PATH)
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = load_system_prompt()

    instances = load_instances(JSONL_PATH)
    print(f"Loaded {len(instances)} instances")
    instances = _filter_instances(instances, args)

    grouped = group_by_db_id(instances)
    print(f"Spanning {len(grouped)} unique databases")

    modes = args.modes
    output_dir = Path(args.output_dir)
    for mode in modes:
        (output_dir / mode).mkdir(parents=True, exist_ok=True)

    mode_stats: Dict[str, ModeStats] = {m: ModeStats() for m in modes}
    skipped_dbs: List[str] = []
    total = len(instances)

    pbar = tqdm(total=total, desc="Instances", unit="inst")

    for db_id, db_instances in grouped.items():
        pbar.set_postfix(db=db_id)

        try:
            engine, search_engine = _index_database(credential, db_id)
        except Exception as e:
            tqdm.write(f"SKIP: Failed to index {db_id}: {e}")
            skipped_dbs.append(db_id)
            for inst in db_instances:
                _record_failure(inst.instance_id, modes, mode_stats, output_dir)
            pbar.update(len(db_instances))
            continue

        for inst in db_instances:
            for mode in modes:
                _run_instance_mode(
                    inst, mode, search_engine, system_prompt, client,
                    mode_stats, output_dir,
                )
            pbar.update(1)

        engine.dispose()

    pbar.close()
    print_summary(total, skipped_dbs, mode_stats, modes, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spider 2.0-Snow: Claude vanilla vs MCP evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max instances to evaluate (for testing)",
    )
    parser.add_argument(
        "--db-filter",
        nargs="+",
        help="Only evaluate specific db_id(s)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["vanilla", "mcp"],
        default=["vanilla", "mcp"],
        help="Evaluation modes to run (default: both)",
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
