"""
Generate a combined pass@k figure for Qwen2.5-1.5B-Instruct
on AIME 2023 + 2024 + 2025 (combined evaluation).

Layout:
  Left  — pass@k curve (sampled) + greedy pass@1 marker, combined across all years
  Right — per-year bar chart comparing pass@256 and greedy pass@1

Usage:
    python plot_results.py
    python plot_results.py --output my_figure.png
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
YEARS    = [2023, 2024, 2025]

# ── Update these values after re-running with the combined evaluation ─────────
# combined: pass@k values averaged across all 2023+2024+2025 problems
# by_year : pass@256 and greedy_pass@1 per year (from wandb combined run)
HARDCODED = {
    "combined": {
        # sampled pass@k (fill all K_VALUES after re-run; currently only 4 measured)
        "pass@1":   0.030295,
        "pass@8":   0.089473,
        "pass@64":  0.234631,
        "pass@256": 0.366667,
        # greedy pass@1 — update after re-run
        "greedy_pass@1": None,
    },
    "by_year": {
        2023: {"pass@256": 0.26667, "greedy_pass@1": None},
        2024: {"pass@256": 0.46667, "greedy_pass@1": None},
        2025: {"pass@256": 0.36667, "greedy_pass@1": None},
    },
}


def smooth_curve(ks, vals, n_interp=300):
    log_ks   = np.log2(ks)
    log_x    = np.linspace(log_ks[0], log_ks[-1], n_interp)
    interp_y = np.interp(log_x, log_ks, vals)
    return 2 ** log_x, interp_y


def make_figure(data: dict, output: str) -> None:
    combined = data.get("combined", {})
    by_year  = data.get("by_year", {})

    # Collect measured pass@k points
    ks   = [k for k in K_VALUES if f"pass@{k}" in combined and combined[f"pass@{k}"] is not None]
    vals = [combined[f"pass@{k}"] for k in ks]
    greedy = combined.get("greedy_pass@1")

    if not ks:
        print("[error] No pass@k data in combined.", file=sys.stderr)
        sys.exit(1)

    COLOR_SAMPLE = "#2166ac"
    COLOR_GREEDY = "#d6604d"

    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig, wspace=0.35)
    ax_main = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # ── Left panel: pass@k curve ──────────────────────────────────────────────
    if len(ks) >= 3:
        xs, ys = smooth_curve(ks, vals)
        ax_main.plot(xs, ys, color=COLOR_SAMPLE, linewidth=1.8, alpha=0.9, zorder=3)

    ax_main.plot(ks, vals,
                 color=COLOR_SAMPLE, linewidth=0,
                 marker="o", markersize=5.5,
                 markerfacecolor="white", markeredgewidth=1.6,
                 markeredgecolor=COLOR_SAMPLE, zorder=5,
                 label="sampled pass@$k$")

    for k, v in zip(ks, vals):
        ax_main.annotate(f"{v:.3f}", xy=(k, v),
                         xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=6.5, color=COLOR_SAMPLE)

    # Greedy pass@1 marker (diamond, different colour)
    if greedy is not None:
        ax_main.plot(1, greedy,
                     color=COLOR_GREEDY, linewidth=0,
                     marker="D", markersize=6,
                     markerfacecolor=COLOR_GREEDY, markeredgewidth=0,
                     zorder=6, label="greedy pass@1")
        ax_main.annotate(f"{greedy:.3f}", xy=(1, greedy),
                         xytext=(0, -13), textcoords="offset points",
                         ha="center", fontsize=6.5, color=COLOR_GREEDY)

    y_max = max(vals + ([greedy] if greedy is not None else [])) * 1.32
    ax_main.set_xscale("log", base=2)
    ax_main.set_xlim(0.7, 340)
    ax_main.set_ylim(0.0, y_max)
    ax_main.set_xticks(K_VALUES)
    ax_main.get_xaxis().set_major_formatter(
        ticker.FixedFormatter([str(k) for k in K_VALUES])
    )
    ax_main.tick_params(axis="x", labelsize=7.5, rotation=45)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_main.tick_params(axis="y", labelsize=8.5)
    ax_main.set_xlabel("Number of Samples $k$", fontsize=9)
    ax_main.set_ylabel("pass@$k$", fontsize=9.5)
    ax_main.set_title("Combined AIME 2023 + 2024 + 2025", fontsize=11, fontweight="bold", pad=6)
    ax_main.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.35, alpha=0.45)
    ax_main.spines[["top", "right"]].set_visible(False)

    # ── Right panel: per-year bar chart (pass@256 + greedy_pass@1) ────────────
    years_present = [y for y in YEARS if y in by_year]
    n_years = len(years_present)

    p256_vals   = [by_year[y].get("pass@256")      for y in years_present]
    greedy_vals = [by_year[y].get("greedy_pass@1") for y in years_present]

    x      = np.arange(n_years)
    width  = 0.35
    bar_p256   = ax_bar.bar(x - width / 2, p256_vals,   width, color=COLOR_SAMPLE, alpha=0.85, label="pass@256")
    bar_greedy = ax_bar.bar(x + width / 2,
                            [v if v is not None else 0.0 for v in greedy_vals],
                            width, color=COLOR_GREEDY, alpha=0.85, label="greedy pass@1",
                            hatch="//" if any(v is None for v in greedy_vals) else "")

    # Annotate bars
    for bar, v in zip(bar_p256, p256_vals):
        if v is not None:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=COLOR_SAMPLE)

    for bar, v in zip(bar_greedy, greedy_vals):
        label = f"{v:.3f}" if v is not None else "N/A"
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    label, ha="center", va="bottom", fontsize=6.5, color=COLOR_GREEDY)

    if any(v is None for v in greedy_vals):
        ax_bar.text(0.97, 0.97, "// = not yet measured",
                    transform=ax_bar.transAxes, ha="right", va="top",
                    fontsize=6, color="#888888", style="italic")

    bar_max = max(v for v in p256_vals + [v or 0.0 for v in greedy_vals] if v is not None)
    ax_bar.set_ylim(0.0, bar_max * 1.35)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(y) for y in years_present], fontsize=9)
    ax_bar.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.set_xlabel("AIME Year", fontsize=9)
    ax_bar.set_title("pass@256 vs greedy pass@1\nby Year", fontsize=9.5, fontweight="bold", pad=6)
    ax_bar.legend(fontsize=7.5, framealpha=0.9)
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.45)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Figure: Qwen2.5-1.5B-Instruct — AIME 2023/2024/2025 Combined Evaluation",
        fontsize=8.5, y=1.02, style="italic",
    )

    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"[done] saved → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_figure.png")
    args = parser.parse_args()
    make_figure(HARDCODED, args.output)


if __name__ == "__main__":
    main()
