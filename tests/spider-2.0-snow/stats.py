"""Statistical analysis and reporting for Spider 2.0-Snow evaluation."""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import stats as scipy_stats

from constants import MODEL_NAME, SPIDER2_SNOW_DIR
from models import ModeStats


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
        chi2 = (abs(b_wins - a_wins) - 1) ** 2 / total_discordant
        p_value = float(1 - scipy_stats.chi2.cdf(chi2, df=1))

    # Effect size (Cohen's h for proportions)
    effect_size = 2 * np.arcsin(np.sqrt(mean_b)) - 2 * np.arcsin(np.sqrt(mean_a))

    # 95% CI for difference in proportions (Wald)
    diff = mean_b - mean_a
    se_diff = np.sqrt((mean_a * (1 - mean_a) + mean_b * (1 - mean_b)) / n)
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


def _print_mode_stats(mode: str, stats: ModeStats) -> None:
    """Print stats for a single mode."""
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


def _print_paired_comparison(paired: dict, output_dir: Path) -> None:
    """Print paired comparison results and save report."""
    print(f"\n--- Paired Comparison (n={paired['n_paired']}) ---")
    print(f"  Vanilla success rate: {paired['vanilla']['mean']:.3f}")
    print(f"  MCP success rate:     {paired['mcp']['mean']:.3f}")
    print(f"  Difference (MCP - vanilla): {paired['difference']:+.3f}")
    print(f"  95% CI: [{paired['ci_95'][0]:+.3f}, {paired['ci_95'][1]:+.3f}]")
    print(f"  McNemar p-value: {paired['p_value']:.4f}")
    print(f"  Effect size (Cohen's h): {paired['effect_size_cohens_h']:.3f}")
    disc = paired["mcnemar_discordant"]
    print(f"  Discordant pairs: vanilla_wins={disc['vanilla_wins']}, mcp_wins={disc['mcp_wins']}")

    sig = "YES" if paired["p_value"] < 0.05 else "NO"
    print(f"  Statistically significant (p<0.05): {sig}")

    report_path = output_dir / "statistical_report.json"
    with open(report_path, "w") as f:
        json.dump(paired, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


def print_statistical_report(
    mode_stats: Dict[str, ModeStats], output_dir: Path
) -> None:
    """Print and save statistical comparison report."""
    print(f"\n{'='*60}")
    print("STATISTICAL REPORT")
    print(f"{'='*60}")

    for mode, stats in mode_stats.items():
        _print_mode_stats(mode, stats)

    if "vanilla" in mode_stats and "mcp" in mode_stats:
        paired = compute_paired_stats(mode_stats["vanilla"], mode_stats["mcp"])
        if "error" not in paired:
            _print_paired_comparison(paired, output_dir)


def print_summary(
    total: int, skipped_dbs: list, mode_stats: Dict[str, ModeStats],
    modes: list, output_dir: Path,
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
