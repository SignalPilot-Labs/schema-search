"""Spider 2.0-Snow evaluation: Claude vanilla vs Claude + schema-search MCP."""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import anthropic
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from tqdm import tqdm

from agent import CreditExhaustedError, run_agent
from constants import (
    CREDENTIAL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROLE,
    DEFAULT_WAREHOUSE,
    DEFAULT_WORKERS,
    DOCUMENTS_DIR,
    ENV_PATH,
    JSONL_PATH,
    MODE_TOOLS,
    PROMPT_PATH,
    SNOWFLAKE_ACCOUNT,
)
from models import Instance, ModeStats
from schema_search.schema_search import SchemaSearch
from stats import print_summary

logger = logging.getLogger(__name__)


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


def build_snowflake_engine(credential: dict, target_db: str) -> Engine:
    """Build a Snowflake SQLAlchemy engine from credentials and target database."""
    user = credential["username"]
    account = credential["account"]
    warehouse = credential.get("warehouse", DEFAULT_WAREHOUSE)
    role = credential.get("role", DEFAULT_ROLE)
    token = credential.get("token", "")

    url = (
        f"snowflake://{user}@{account}/{target_db}"
        f"?warehouse={warehouse}&role={role}"
    )
    connect_args: dict = {}
    if token:
        connect_args["authenticator"] = "programmatic_access_token"
        connect_args["token"] = token
    else:
        password = quote_plus(credential["password"])
        url = (
            f"snowflake://{user}:{password}@{account}/{target_db}"
            f"?warehouse={warehouse}&role={role}"
        )
    return create_engine(url, connect_args=connect_args)


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
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("ERROR: LLM_API_KEY not set in tests/.env")
        sys.exit(1)
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
    engine = build_snowflake_engine(credential, db_id)
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


async def _run_instance_mode(
    inst: Instance, mode: str, search_engine: SchemaSearch,
    engine: Engine, system_prompt: str, client: anthropic.AsyncAnthropic,
    output_dir: Path,
) -> tuple[str, str, Optional[int], Optional[float]]:
    """Run a single instance in a single mode.

    Returns:
        Tuple of (instance_id, status, tool_calls_count, latency).
        tool_calls_count and latency are None on failure/skip.
    """
    if _instance_already_done(inst.instance_id, mode, output_dir):
        return (inst.instance_id, "skip", 0, 0.0)

    ext_knowledge = load_external_knowledge(inst.external_knowledge)
    try:
        t0 = time.time()
        result = await run_agent(
            instance_id=inst.instance_id,
            instruction=inst.instruction,
            external_knowledge=ext_knowledge,
            search_engine=search_engine,
            system_prompt=system_prompt,
            client=client,
            db_id=inst.db_id,
            tools=MODE_TOOLS[mode],
            engine=engine,
        )
        latency = time.time() - t0
        save_sql(result.sql, inst.instance_id, output_dir / mode)

        if result.error:
            return (inst.instance_id, f"fail:{result.error}", None, None)
        return (inst.instance_id, "ok", result.tool_calls_count, latency)

    except CreditExhaustedError:
        raise
    except Exception as e:
        save_sql("", inst.instance_id, output_dir / mode)
        return (inst.instance_id, f"error:{e}", None, None)


async def run_evaluation(args: argparse.Namespace) -> None:
    """Run the full evaluation pipeline with asyncio.

    Indexes databases concurrently and starts instance tasks as soon as
    each database is ready. All API calls are non-blocking (async client).
    """
    api_key = _load_api_key()
    credential = load_snowflake_credential(CREDENTIAL_PATH)
    system_prompt = load_system_prompt()

    instances = load_instances(JSONL_PATH)
    print(f"Loaded {len(instances)} instances")
    instances = _filter_instances(instances, args)

    grouped = group_by_db_id(instances)
    print(f"Spanning {len(grouped)} unique databases")

    modes = args.modes
    output_dir = Path(args.output_dir)
    workers = args.workers

    if args.reset:
        for mode in modes:
            mode_dir = output_dir / mode
            if mode_dir.exists():
                shutil.rmtree(mode_dir)
                print(f"Reset: removed {mode_dir}")

    for mode in modes:
        (output_dir / mode).mkdir(parents=True, exist_ok=True)

    total_tasks = sum(len(insts) * len(modes) for insts in grouped.values())
    total_instances = len(instances)
    print(f"Running {total_tasks} tasks with {workers} concurrent worker(s)")

    mode_stats: Dict[str, ModeStats] = {m: ModeStats() for m in modes}
    db_resources: Dict[str, tuple[Engine, SchemaSearch]] = {}
    skipped_dbs: List[str] = []
    pbar = tqdm(total=total_tasks, desc="Tasks", unit="task")
    semaphore = asyncio.Semaphore(workers)
    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _run_task(inst: Instance, mode: str) -> None:
        """Run a single (instance, mode) task with semaphore-based concurrency."""
        async with semaphore:
            iid, status, tc, lat = await _run_instance_mode(
                inst, mode, db_resources[inst.db_id][1],
                db_resources[inst.db_id][0], system_prompt, client,
                output_dir,
            )

        if status == "skip":
            mode_stats[mode].record_success(iid, tc or 0, lat or 0.0)
            tqdm.write(f"  [{mode}] {iid}: SKIP (already exists)")
        elif status == "ok":
            mode_stats[mode].record_success(iid, tc or 0, lat or 0.0)
            tqdm.write(f"  [{mode}] {iid}: OK (tools={tc}, {lat:.1f}s)")
        else:
            mode_stats[mode].record_failure(iid)
            tqdm.write(f"  [{mode}] {iid}: {status.upper()}")
        pbar.update(1)

    async def _index_and_run(db_id: str) -> List[asyncio.Task]:
        """Index one database (in thread) and launch its instance tasks."""
        try:
            engine, search_engine = await asyncio.to_thread(
                _index_database, credential, db_id
            )
            db_resources[db_id] = (engine, search_engine)
        except Exception as e:
            tqdm.write(f"SKIP: Failed to index {db_id}: {e}")
            skipped_dbs.append(db_id)
            for inst in grouped[db_id]:
                for mode in modes:
                    save_sql("", inst.instance_id, output_dir / mode)
                    mode_stats[mode].record_failure(inst.instance_id)
                    pbar.update(1)
            return []

        # Launch instance tasks immediately
        tasks = []
        for inst in grouped[db_id]:
            for mode in modes:
                tasks.append(asyncio.create_task(_run_task(inst, mode)))
        return tasks

    # Index all databases concurrently, each spawns instance tasks on completion
    index_tasks = [_index_and_run(db_id) for db_id in grouped]
    instance_task_lists = await asyncio.gather(*index_tasks, return_exceptions=True)

    # Collect all instance tasks and wait for completion
    all_instance_tasks = []
    for result in instance_task_lists:
        if isinstance(result, Exception):
            tqdm.write(f"FATAL: indexing crashed: {result}")
        else:
            all_instance_tasks.extend(result)

    # Wait for all instance tasks
    if all_instance_tasks:
        results = await asyncio.gather(*all_instance_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, CreditExhaustedError):
                tqdm.write("STOP: API credits exhausted — cancelling remaining tasks")
                for task in all_instance_tasks:
                    task.cancel()
                break
            elif isinstance(r, Exception):
                tqdm.write(f"FATAL: worker crashed: {r}")

    pbar.close()
    await client.close()

    # Dispose engines
    for engine, _ in db_resources.values():
        engine.dispose()

    print_summary(total_instances, skipped_dbs, mode_stats, modes, output_dir)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing results before running (default: resume)",
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
