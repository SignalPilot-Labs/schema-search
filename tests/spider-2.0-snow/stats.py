"""Statistical analysis and reporting for Spider 2.0-Snow evaluation."""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import stats as scipy_stats

from constants import CREDENTIAL_PATH, MODEL_NAME, SPIDER2_SNOW_DIR
from models import ModeStats
from scorer import score_mode


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


def _compute_tool_call_comparison(
    stats_a: ModeStats, stats_b: ModeStats,
) -> dict:
    """Compare tool call counts between two modes."""
    tc_a = np.array(stats_a.tool_calls) if stats_a.tool_calls else np.array([])
    tc_b = np.array(stats_b.tool_calls) if stats_b.tool_calls else np.array([])

    result = {
        "vanilla": {
            "n": len(tc_a),
            "mean": float(np.mean(tc_a)) if len(tc_a) > 0 else 0.0,
            "std": float(np.std(tc_a, ddof=1)) if len(tc_a) > 1 else 0.0,
            "median": float(np.median(tc_a)) if len(tc_a) > 0 else 0.0,
        },
        "mcp": {
            "n": len(tc_b),
            "mean": float(np.mean(tc_b)) if len(tc_b) > 0 else 0.0,
            "std": float(np.std(tc_b, ddof=1)) if len(tc_b) > 1 else 0.0,
            "median": float(np.median(tc_b)) if len(tc_b) > 0 else 0.0,
        },
    }

    # Mann-Whitney U test (non-parametric, doesn't assume normal distribution)
    if len(tc_a) >= 2 and len(tc_b) >= 2:
        u_stat, p_value = scipy_stats.mannwhitneyu(tc_a, tc_b, alternative="two-sided")
        result["mann_whitney_u"] = float(u_stat)
        result["p_value"] = float(p_value)
        result["difference_in_means"] = result["mcp"]["mean"] - result["vanilla"]["mean"]

    return result


def _print_paired_comparison(
    paired: dict, tool_comparison: dict, output_dir: Path,
) -> None:
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

    print(f"\n--- Tool Calls Comparison ---")
    v = tool_comparison["vanilla"]
    m = tool_comparison["mcp"]
    print(f"  Vanilla: mean={v['mean']:.1f}, median={v['median']:.0f}, std={v['std']:.1f} (n={v['n']})")
    print(f"  MCP:     mean={m['mean']:.1f}, median={m['median']:.0f}, std={m['std']:.1f} (n={m['n']})")
    if "difference_in_means" in tool_comparison:
        print(f"  Difference (MCP - vanilla): {tool_comparison['difference_in_means']:+.1f}")
        print(f"  Mann-Whitney U p-value: {tool_comparison['p_value']:.4f}")

    report = {"accuracy": paired, "tool_calls": tool_comparison}
    report_path = output_dir / "statistical_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
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
        tool_comparison = _compute_tool_call_comparison(
            mode_stats["vanilla"], mode_stats["mcp"]
        )
        if "error" not in paired:
            _print_paired_comparison(paired, tool_comparison, output_dir)


def _load_credential() -> dict:
    """Load Snowflake credential for scoring."""
    with open(CREDENTIAL_PATH) as f:
        return json.load(f)


def _run_execution_scoring(
    modes: list, output_dir: Path,
) -> Dict[str, Dict[str, dict]]:
    """Run execution-based scoring for all modes.

    Returns:
        Dict of mode -> {instance_id: {"score": 0|1, "error": ...}}.
    """
    credential = _load_credential()
    scores: Dict[str, Dict[str, dict]] = {}
    for mode in modes:
        print(f"\nScoring {mode} SQL against Snowflake...")
        scores[mode] = score_mode(mode, output_dir, credential)
    return scores


def _print_execution_scores(
    scores: Dict[str, Dict[str, dict]], output_dir: Path,
) -> dict:
    """Print execution accuracy and return paired data for report."""
    print(f"\n--- Execution Accuracy (Spider2 Score) ---")
    exec_report = {}

    for mode, results in scores.items():
        total = len(results)
        correct = sum(1 for r in results.values() if r["score"] == 1)
        rate = correct / total if total > 0 else 0
        print(f"  [{mode}] {correct}/{total} ({rate:.1%})")
        exec_report[mode] = {
            "correct": correct, "total": total, "accuracy": rate,
        }

        # Show per-instance details
        for iid in sorted(results):
            r = results[iid]
            status = "PASS" if r["score"] == 1 else f"FAIL ({r.get('error', '')})"
            print(f"    {iid}: {status}")

    # Paired comparison on execution accuracy
    modes = list(scores.keys())
    if len(modes) == 2 and "vanilla" in scores and "mcp" in scores:
        common_ids = sorted(set(scores["vanilla"]) & set(scores["mcp"]))
        if common_ids:
            v_scores = {iid: scores["vanilla"][iid]["score"] for iid in common_ids}
            m_scores = {iid: scores["mcp"][iid]["score"] for iid in common_ids}

            v_correct = sum(v_scores.values())
            m_correct = sum(m_scores.values())
            n = len(common_ids)
            print(f"\n  Paired (n={n}): vanilla={v_correct}/{n}, mcp={m_correct}/{n}")

            # Show discordant pairs
            mcp_wins = [iid for iid in common_ids if v_scores[iid] == 0 and m_scores[iid] == 1]
            van_wins = [iid for iid in common_ids if v_scores[iid] == 1 and m_scores[iid] == 0]
            if mcp_wins:
                print(f"  MCP wins: {mcp_wins}")
            if van_wins:
                print(f"  Vanilla wins: {van_wins}")

            exec_report["paired"] = {
                "n": n,
                "vanilla_correct": v_correct,
                "mcp_correct": m_correct,
                "mcp_wins": mcp_wins,
                "vanilla_wins": van_wins,
            }

    return exec_report


def print_summary(
    total: int, skipped_dbs: list, mode_stats: Dict[str, ModeStats],
    modes: list, output_dir: Path,
) -> None:
    """Print final summary with generation stats, execution scoring, and tool call comparison."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total instances: {total}")
    print(f"Skipped databases: {len(skipped_dbs)}")
    if skipped_dbs:
        print(f"  {skipped_dbs}")

    print_statistical_report(mode_stats, output_dir)

    # Run execution scoring
    exec_scores = _run_execution_scoring(modes, output_dir)
    exec_report = _print_execution_scores(exec_scores, output_dir)

    # Save combined report
    report_path = output_dir / "statistical_report.json"
    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    report["execution_accuracy"] = exec_report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Report: {report_path}")
