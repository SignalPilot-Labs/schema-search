import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

strategies = ["Semantic", "BM25", "Fuzzy", "Hybrid"]

data_without_reranker = {
    "scores": [44, 41, 23, 42],
    "latencies": [216, 16, 16, 104],
    "memory": [466.27, 403.53, 403.56, 480.00],
}

data_with_reranker = {
    "scores": [49, 44, 45, 44],
    "latencies": [801, 760, 579, 652],
    "memory": [549.11, 560.14, 560.77, 591.86],
}

color_without = "#3b82f6"
color_with = "#f97316"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

x = np.arange(len(strategies))
width = 0.18

# Plot 1: Score Comparison
bars1_1 = ax1.bar(
    x - width / 2,
    data_without_reranker["scores"],
    width,
    label="Without Reranker",
    color=color_without,
    edgecolor="white",
    linewidth=1,
)

bars1_2 = ax1.bar(
    x + width / 2,
    data_with_reranker["scores"],
    width,
    label="With Reranker",
    color=color_with,
    edgecolor="white",
    linewidth=1,
)

ax1.set_xlabel("Strategy", fontsize=13, fontweight="600")
ax1.set_ylabel("Score (out of 50)", fontsize=13, fontweight="600")
ax1.set_title("Score Comparison", fontsize=14, fontweight="700", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=12)
ax1.set_xlim(-0.5, len(strategies) - 0.5)
ax1.set_ylim(0, 55)
ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
ax1.tick_params(axis="both", labelsize=11)
ax1.legend(fontsize=11, loc="upper right", framealpha=0.9)

for bar, score in zip(bars1_1, data_without_reranker["scores"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 1,
        f"{score}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

for bar, score in zip(bars1_2, data_with_reranker["scores"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 1,
        f"{score}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

# Plot 2: Latency Comparison
bars2_1 = ax2.bar(
    x - width / 2,
    data_without_reranker["latencies"],
    width,
    label="Without Reranker",
    color=color_without,
    edgecolor="white",
    linewidth=1,
)

bars2_2 = ax2.bar(
    x + width / 2,
    data_with_reranker["latencies"],
    width,
    label="With Reranker",
    color=color_with,
    edgecolor="white",
    linewidth=1,
)

ax2.set_xlabel("Strategy", fontsize=13, fontweight="600")
ax2.set_ylabel("Latency (ms)", fontsize=13, fontweight="600")
ax2.set_title("Latency Comparison", fontsize=14, fontweight="700", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(strategies, fontsize=12)
ax2.set_xlim(-0.5, len(strategies) - 0.5)
ax2.set_ylim(0, max(data_with_reranker["latencies"]) * 1.2)
ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
ax2.tick_params(axis="both", labelsize=11)
ax2.legend(fontsize=11, loc="upper right", framealpha=0.9)

for bar, latency in zip(bars2_1, data_without_reranker["latencies"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["latencies"]) * 0.02,
        f"{latency}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

for bar, latency in zip(bars2_2, data_with_reranker["latencies"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["latencies"]) * 0.02,
        f"{latency}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

# Plot 3: Memory Comparison
bars3_1 = ax3.bar(
    x - width / 2,
    data_without_reranker["memory"],
    width,
    label="Without Reranker",
    color=color_without,
    edgecolor="white",
    linewidth=1,
)

bars3_2 = ax3.bar(
    x + width / 2,
    data_with_reranker["memory"],
    width,
    label="With Reranker",
    color=color_with,
    edgecolor="white",
    linewidth=1,
)

ax3.set_xlabel("Strategy", fontsize=13, fontweight="600")
ax3.set_ylabel("Peak Memory (MB)", fontsize=13, fontweight="600")
ax3.set_title("Memory Comparison", fontsize=14, fontweight="700", pad=12)
ax3.set_xticks(x)
ax3.set_xticklabels(strategies, fontsize=12)
ax3.set_xlim(-0.5, len(strategies) - 0.5)
ax3.set_ylim(0, max(data_with_reranker["memory"]) * 1.2)
ax3.grid(axis="y", alpha=0.3, linewidth=0.5)
ax3.tick_params(axis="both", labelsize=11)
ax3.legend(fontsize=11, loc="upper right", framealpha=0.9)

for bar, mem in zip(bars3_1, data_without_reranker["memory"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["memory"]) * 0.02,
        f"{mem:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

for bar, mem in zip(bars3_2, data_with_reranker["memory"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["memory"]) * 0.02,
        f"{mem:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

plt.tight_layout()
plt.savefig(
    "img/strategy_comparison.png", dpi=300, bbox_inches="tight", facecolor="white"
)
print("✅ Plot saved to img/strategy_comparison.png")
