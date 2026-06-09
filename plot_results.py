"""
Generate a pass@k figure (styled after SimpleRLZoo Fig.10)
for Qwen2.5-1.5B-Instruct on AIME 2023 / 2024 / 2025.

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
import matplotlib.ticker as ticker

YEARS = [2023, 2024, 2025]
K_VALUES = [1, 8, 64, 256]

# ── hardcoded data from wandb ──────────────────────────────────────────────────
HARDCODED = {
    2023: {
        "overall": {
            "pass@1": 0.053515624999999976,
            "pass@8": 0.08279817120721059,
            "pass@64": 0.16721986158379815,
            "pass@256": 0.26666666666666666,
        },
        "AIME_I": {
            "pass@1": 0.04973958333333329,
            "pass@8": 0.08496207470670052,
            "pass@64": 0.18750353485516696,
            "pass@256": 0.3333333333333333,
        },
        "AIME_II": {
            "pass@1": 0.057291666666666664,
            "pass@8": 0.08063426770772066,
            "pass@64": 0.1469361883124293,
            "pass@256": 0.2,
        },
    },
    2024: {
        "overall": {
            "pass@1": 0.027864583333333293,
            "pass@8": 0.12057864411555592,
            "pass@64": 0.3060115578475281,
            "pass@256": 0.4666666666666667,
        },
        "AIME_I": {
            "pass@1": 0.0434895833333333,
            "pass@8": 0.1636409827499138,
            "pass@64": 0.3518196813628896,
            "pass@256": 0.4666666666666667,
        },
        "AIME_II": {
            "pass@1": 0.012239583333333288,
            "pass@8": 0.07751630548119805,
            "pass@64": 0.2602034343321666,
            "pass@256": 0.4666666666666667,
        },
    },
    2025: {
        "overall": {
            "pass@1": 0.009505208333333274,
            "pass@8": 0.0650410010260333,
            "pass@64": 0.23066172116212208,
            "pass@256": 0.36666666666666664,
        },
        "AIME_I": {
            "pass@1": 0.009505208333333274,
            "pass@8": 0.0650410010260333,
            "pass@64": 0.23066172116212208,
            "pass@256": 0.36666666666666664,
        },
    },
}


def smooth_curve(ks, vals, n_interp=300):
    log_ks   = np.log2(ks)
    log_x    = np.linspace(log_ks[0], log_ks[-1], n_interp)
    interp_y = np.interp(log_x, log_ks, vals)
    return 2 ** log_x, interp_y


def get_avg_vals(year_data: dict) -> dict:
    """Average AIME_I and AIME_II; fall back to AIME_I if AIME_II absent."""
    if "AIME_II" in year_data:
        avg = {}
        for k in K_VALUES:
            key = f"pass@{k}"
            v_i  = year_data["AIME_I"].get(key)
            v_ii = year_data["AIME_II"].get(key)
            if v_i is not None and v_ii is not None:
                avg[key] = (v_i + v_ii) / 2
        return avg
    return dict(year_data.get("AIME_I", year_data.get("overall", {})))


def make_figure(data: dict, output: str) -> None:
    years_with_data = sorted(y for y in YEARS if y in data)
    n_cols = len(years_with_data)

    if n_cols == 0:
        print("[error] No data available.", file=sys.stderr)
        sys.exit(1)

    COLOR = "#2166ac"
    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 3.2), sharey=True)
    if n_cols == 1:
        axes = [axes]

    global_max = 0.0
    for year in years_with_data:
        vals = list(get_avg_vals(data[year]).values())
        if vals:
            global_max = max(global_max, max(vals))

    for ax, year in zip(axes, years_with_data):
        avg = get_avg_vals(data[year])
        ks   = [k for k in K_VALUES if f"pass@{k}" in avg]
        vals = [avg[f"pass@{k}"] for k in ks]

        if len(ks) >= 3:
            xs, ys = smooth_curve(ks, vals)
            ax.plot(xs, ys, color=COLOR, linewidth=1.8, alpha=0.9, zorder=3)

        ax.plot(ks, vals, color=COLOR, linewidth=0,
                marker="o", markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                markeredgecolor=COLOR, zorder=4)

        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.3f}", xy=(k, v),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color=COLOR)

        # note for 2025 that AIME_II is absent
        if "AIME_II" not in data[year]:
            ax.text(0.97, 0.04, "※ AIME I only",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color="#888888", style="italic")

        ax.set_xscale("log", base=2)
        ax.set_xlim(0.7, 340)
        ax.set_ylim(0.0, global_max * 1.28)

        ax.set_xticks([1, 8, 64, 256])
        ax.get_xaxis().set_major_formatter(ticker.FixedFormatter(["1", "8", "64", "256"]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=8.5)

        ax.set_title(f"AIME {year}", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Number of Samples $k$", fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("pass@$k$", fontsize=9.5)

        ax.grid(True, which="both", linestyle="--", linewidth=0.35, alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=COLOR, linewidth=1.8,
                   marker="o", markersize=5, markerfacecolor="white",
                   markeredgewidth=1.5, markeredgecolor=COLOR,
                   label="Qwen2.5-1.5B-Instruct"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=1,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle(
        "Figure: pass@k Results of Qwen2.5-1.5B-Instruct on AIME 2023 / 2024 / 2025",
        fontsize=8, y=-0.04, style="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"[done] saved → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_figure.png")
    args = parser.parse_args()
    make_figure(HARDCODED, args.output)


if __name__ == "__main__":
    main()
