"""Spider 2.0-Snow evaluation: Claude vanilla vs Claude + schema-search MCP."""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import anthropic
import numpy as np
from dotenv import load_dotenv
from scipy import stats as scipy_stats
from tqdm import tqdm

# Ensure project root is on path before local modules
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_this_dir = str(Path(__file__).parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from agent import run_agent
from constants import MODEL_NAME, TOOLS_MCP, TOOLS_VANILLA
from schema_search.schema_search import SchemaSearch
from schema_search.utils.utils import create_engine_from_url

logger = logging.getLogger(__name__)

SPIDER2_SNOW_DIR = Path(__file__).parent / "Spider2" / "spider2-snow"
JSONL_PATH = SPIDER2_SNOW_DIR / "spider2-snow.jsonl"
DOCUMENTS_DIR = SPIDER2_SNOW_DIR / "resource" / "documents"
CREDENTIAL_PATH = Path(__file__).parent / "snowflake_credential.json"

MODE_TOOLS = {
    "vanilla": TOOLS_VANILLA,
    "mcp": TOOLS_MCP,
}


@dataclass
class Instance:
    """A single Spider 2.0-Snow evaluation instance."""

    instance_id: str
    instruction: str
    db_id: str
    external_knowledge: Optional[str]


@dataclass
class ModeStats:
    """Tracks per-instance results for a mode."""

    succeeded: int = 0
    failed: int = 0
    tool_calls: List[int] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    # Paired binary outcomes per instance_id (1=sql generated, 0=error)
    outcomes: Dict[str, int] = field(default_factory=dict)


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
            '  {"username": "...", "password": "...", "account": "RSRSBDK-YDB67606",'
            ' "role": "PARTICIPANT", "warehouse": "COMPUTE_WH_PARTICIPANT"}'
        )
        sys.exit(1)
    with open(credential_path) as f:
        return json.load(f)


def build_snowflake_url(credential: dict, target_db: str) -> str:
    """Build a Snowflake SQLAlchemy URL from credentials and target database."""
    user = credential["username"]
    password = quote_plus(credential["password"])
    account = credential["account"]
    warehouse = credential.get("warehouse", "COMPUTE_WH_PARTICIPANT")
    role = credential.get("role", "PARTICIPANT")
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
    prompt_path = Path(__file__).parent / "prompt.md"
    return prompt_path.read_text()


def save_sql(sql: str, instance_id: str, output_dir: Path) -> None:
    """Save generated SQL to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{instance_id}.sql").write_text(sql)


def compute_paired_stats(stats_a: ModeStats, stats_b: ModeStats) -> dict:
    """Compute paired statistical comparison between two modes.

    Uses McNemar's test for paired binary outcomes and reports
    mean, std, confidence interval, and effect size.
    """
    common_ids = sorted(set(stats_a.outcomes) & set(stats_b.outcomes))
    n = len(common_ids)
    if n == 0:
        return {"error": "no_common_instances"}

    a_vals = np.array([stats_a.outcomes[i] for i in common_ids])
    b_vals = np.array([stats_b.outcomes[i] for i in common_ids])

    mean_a = float(np.mean(a_vals))
    mean_b = float(np.mean(b_vals))
    std_a = float(np.std(a_vals, ddof=1)) if n > 1 else 0.0
    std_b = float(np.std(b_vals, ddof=1)) if n > 1 else 0.0

    # McNemar's test: count discordant pairs
    b_wins = int(np.sum((a_vals == 0) & (b_vals == 1)))
    a_wins = int(np.sum((a_vals == 1) & (b_vals == 0)))
    total_discordant = a_wins + b_wins

    if total_discordant == 0:
        p_value = 1.0
    else:
        # McNemar's test with continuity correction
        chi2 = (abs(b_wins - a_wins) - 1) ** 2 / total_discordant
        p_value = float(1 - scipy_stats.chi2.cdf(chi2, df=1))

    # Effect size (Cohen's h for proportions)
    effect_size = 2 * np.arcsin(np.sqrt(mean_b)) - 2 * np.arcsin(np.sqrt(mean_a))

    # 95% CI for difference in proportions (Wald)
    diff = mean_b - mean_a
    se_diff = np.sqrt((mean_a * (1 - mean_a) + mean_b * (1 - mean_b)) / n) if n > 0 else 0
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    return {
        "n_paired": n,
        "vanilla": {"mean": mean_a, "std": std_a, "succeeded": int(np.sum(a_vals))},
        "mcp": {"mean": mean_b, "std": std_b, "succeeded": int(np.sum(b_vals))},
        "difference": diff,
        "ci_95": [float(ci_low), float(ci_high)],
        "p_value": p_value,
        "effect_size_cohens_h": float(effect_size),
        "mcnemar_discordant": {"vanilla_wins": a_wins, "mcp_wins": b_wins},
    }


def print_statistical_report(mode_stats: Dict[str, ModeStats], output_dir: Path) -> None:
    """Print and save statistical comparison report."""
    print(f"\n{'='*60}")
    print("STATISTICAL REPORT")
    print(f"{'='*60}")

    for mode, stats in mode_stats.items():
        total = stats.succeeded + stats.failed
        rate = stats.succeeded / total if total > 0 else 0
        print(f"\n[{mode}]")
        print(f"  SQL generation rate: {stats.succeeded}/{total} ({rate:.1%})")
        if stats.tool_calls:
            tc = np.array(stats.tool_calls)
            print(f"  Tool calls: mean={np.mean(tc):.1f}, std={np.std(tc):.1f}")
        if stats.latencies:
            lat = np.array(stats.latencies)
            print(f"  Latency: mean={np.mean(lat):.1f}s, std={np.std(lat):.1f}s")

    if "vanilla" in mode_stats and "mcp" in mode_stats:
        paired = compute_paired_stats(mode_stats["vanilla"], mode_stats["mcp"])
        if "error" not in paired:
            print(f"\n--- Paired Comparison (n={paired['n_paired']}) ---")
            print(f"  Vanilla success rate: {paired['vanilla']['mean']:.3f}")
            print(f"  MCP success rate:     {paired['mcp']['mean']:.3f}")
            print(f"  Difference (MCP - vanilla): {paired['difference']:+.3f}")
            print(f"  95% CI: [{paired['ci_95'][0]:+.3f}, {paired['ci_95'][1]:+.3f}]")
            print(f"  McNemar p-value: {paired['p_value']:.4f}")
            print(f"  Effect size (Cohen's h): {paired['effect_size_cohens_h']:.3f}")
            print(f"  Discordant pairs: vanilla_wins={paired['mcnemar_discordant']['vanilla_wins']}, mcp_wins={paired['mcnemar_discordant']['mcp_wins']}")

            sig = "YES" if paired["p_value"] < 0.05 else "NO"
            print(f"  Statistically significant (p<0.05): {sig}")

            report_path = output_dir / "statistical_report.json"
            with open(report_path, "w") as f:
                json.dump(paired, f, indent=2)
            print(f"\n  Report saved to: {report_path}")


def _instance_already_done(instance_id: str, mode: str, output_dir: Path) -> bool:
    """Check if an instance result already exists (for resumability)."""
    sql_path = output_dir / mode / f"{instance_id}.sql"
    if not sql_path.exists():
        return False
    content = sql_path.read_text().strip()
    return len(content) > 0


def _load_api_key() -> str:
    """Load and validate the Anthropic API key from .env."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
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
) -> tuple["Engine", SchemaSearch]:
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
        mode_stats[mode].failed += 1
        mode_stats[mode].outcomes[instance_id] = 0


def _run_instance_mode(
    inst: Instance, mode: str, search_engine: SchemaSearch,
    system_prompt: str, client: anthropic.Anthropic, mode_stats: Dict[str, ModeStats],
    output_dir: Path,
) -> None:
    """Run a single instance in a single mode and record results."""
    if _instance_already_done(inst.instance_id, mode, output_dir):
        mode_stats[mode].succeeded += 1
        mode_stats[mode].outcomes[inst.instance_id] = 1
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
        mode_stats[mode].latencies.append(latency)
        mode_stats[mode].tool_calls.append(result.tool_calls_count)

        if result.error:
            mode_stats[mode].failed += 1
            mode_stats[mode].outcomes[inst.instance_id] = 0
            status = f"FAIL ({result.error})"
        else:
            mode_stats[mode].succeeded += 1
            mode_stats[mode].outcomes[inst.instance_id] = 1
            status = f"OK (tools={result.tool_calls_count}, {latency:.1f}s)"

        tqdm.write(f"  [{mode}] {inst.instance_id}: {status}")

    except Exception as e:
        save_sql("", inst.instance_id, output_dir / mode)
        mode_stats[mode].failed += 1
        mode_stats[mode].outcomes[inst.instance_id] = 0
        tqdm.write(f"  [{mode}] {inst.instance_id}: ERROR {e}")


def _print_summary(
    total: int, skipped_dbs: List[str], mode_stats: Dict[str, ModeStats],
    modes: List[str], output_dir: Path,
) -> None:
    """Print final summary and evaluation instructions."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total instances: {total}")
    print(f"Skipped databases: {len(skipped_dbs)}")
    if skipped_dbs:
        print(f"  {skipped_dbs}")

    print_statistical_report(mode_stats, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print(f"\nTo evaluate with Spider 2.0 suite:")
    eval_suite = SPIDER2_SNOW_DIR / "evaluation_suite"
    for mode in modes:
        result_dir = output_dir / mode
        print(
            f"  cd {eval_suite} && python evaluate.py"
            f" --mode sql --result_dir {result_dir.resolve()}"
        )


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
    _print_summary(total, skipped_dbs, mode_stats, modes, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spider 2.0-Snow: Claude vanilla vs MCP evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results"),
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
