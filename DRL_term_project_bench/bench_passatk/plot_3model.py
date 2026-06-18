#!/usr/bin/env python3
"""Pass@K 차트에서 PAV-distribution(zero-shot)을 제외한 3-모델 버전 생성.

원본 4-모델 그림(runs/AIME_4model_passk.png)은 그대로 두고
runs/AIME_3model_passk.png 로 저장한다.

Usage:
    python -m bench_passatk.plot_3model
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_passatk.eval.metrics import aggregate_metrics

YEARS = ["AIME2023", "AIME2024", "AIME2025"]
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# PAV-distribution(zero-shot) 제외 — 3개 모델
MODELS = [
    ("PAV-dist+few-shot", "runs/pav-fewshot-ckpt500-aime3", "few-shot (Step k:)", "#d62728", "D"),
    ("PAV-scalar-c2", "runs/pav-scalar-c2-checkpoint-500", "zero-shot", "#ff7f0e", "s"),
    ("Baseline", "runs/qwen-math-1.5b-baseline", "zero-shot", "#2ca02c", "^"),
]


def load(d):
    out = []
    for y in YEARS:
        for line in open(Path(d) / (y + ".jsonl"), encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    aggs = {m[0]: aggregate_metrics(load(m[1])) for m in MODELS}

    fig, ax = plt.subplots(figsize=(9, 6))
    for n, d, setup, color, marker in MODELS:
        vals = [aggs[n]["pass@k"][k]["mean"] for k in K_VALUES]
        ax.plot(K_VALUES, vals, color=color, marker=marker, markersize=7, lw=2,
                label=f"{n} ({setup})", markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=color)
        for k, v in zip(K_VALUES, vals):
            if k in (1, 8, 64, 256):
                ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=color, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 64, 256]); ax.set_xticklabels(["1", "8", "64", "256"], fontsize=12)
    ax.set_ylim(-0.02, 0.6); ax.grid(alpha=.3, ls="--")
    ax.set_xlabel("Number of Samples k", fontsize=12); ax.set_ylabel("pass@k", fontsize=12)
    ax.set_title("AIME 2023–2025 Combined (89 problems) — 3 models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    out_png = Path("runs/AIME_3model_passk.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    for n in [m[0] for m in MODELS]:
        print(f"{n}: pass@256={aggs[n]['pass@k'][256]['mean']:.4f}")
    print(f"\nPlot: {out_png}")


if __name__ == "__main__":
    main()
