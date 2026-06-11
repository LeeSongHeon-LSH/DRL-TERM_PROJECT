import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Overall (AIME I + II combined) values only
greedy_overall = {
    2023: 0.06667,
    2024: 0.03333,
    2025: 0.00000,   # AIME I only
}

pass_at_k_overall = {
    2023: {"pass@1": 0.05534, "pass@8": 0.08513, "pass@64": 0.14948, "pass@256": 0.30000},
    2024: {"pass@1": 0.02734, "pass@8": 0.11922, "pass@64": 0.28883, "pass@256": 0.43333},
    2025: {"pass@1": 0.01016, "pass@8": 0.06985, "pass@64": 0.24688, "pass@256": 0.40000},  # AIME I only
}

YEARS = [2023, 2024, 2025]
K_LABELS = ["pass@1", "pass@8", "pass@64", "pass@256"]
YEAR_COLORS = {2023: "#4C72B0", 2024: "#DD8452", 2025: "#55A868"}
BAR_COLOR = "#4C72B0"

fig = plt.figure(figsize=(17, 9))
fig.patch.set_facecolor("#F8F9FA")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.32)

# ── Row 0: Greedy pass@1 bar charts ──────────────────────────────────────────
for col, year in enumerate(YEARS):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("#FFFFFF")

    val = greedy_overall[year] * 100
    bar = ax.bar([0], [val], color=BAR_COLOR, width=0.4,
                 edgecolor="white", linewidth=1.2, zorder=3)
    ax.text(0, val + 0.3, f"{val:.2f}%",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color="#1A1A2E")

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 12)
    ax.set_xticks([0])
    ax.set_xticklabels(["Overall"], fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=9, color="#555555")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    note = " (AIME I only)" if year == 2025 else ""
    ax.set_title(f"Greedy pass@1  ·  {year}{note}",
                 fontsize=11, fontweight="bold", color="#1F4E79", pad=8)

# ── Row 1: pass@k line charts ─────────────────────────────────────────────────
for col, year in enumerate(YEARS):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor("#FFFFFF")

    k_data = pass_at_k_overall[year]
    x = np.arange(len(K_LABELS))
    y = [k_data[k] * 100 for k in K_LABELS]

    ax.plot(x, y, color=YEAR_COLORS[year], linewidth=2.5,
            marker="o", markersize=8, markerfacecolor="white",
            markeredgewidth=2.5, markeredgecolor=YEAR_COLORS[year], zorder=4)
    ax.fill_between(x, y, alpha=0.10, color=YEAR_COLORS[year], zorder=2)

    for xi, yi in zip(x, y):
        offset = 1.8 if xi < len(K_LABELS) - 1 else -3.5
        ax.text(xi, yi + offset, f"{yi:.1f}%",
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=YEAR_COLORS[year])

    ax.set_xticks(x)
    ax.set_xticklabels(K_LABELS, fontsize=10)
    ax.set_ylim(0, 55)
    ax.set_ylabel("Score (%)", fontsize=9, color="#555555")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.grid(axis="x", alpha=0.15, linestyle="--", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    note = " (AIME I only)" if year == 2025 else ""
    ax.set_title(f"pass@k  ·  {year}{note}",
                 fontsize=11, fontweight="bold", color="#7B2D8B", pad=8)

fig.suptitle("Qwen2.5-Math-1.5B  —  AIME Overall Performance",
             fontsize=15, fontweight="bold", color="#1A1A2E", y=0.99)

out = "aime_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
