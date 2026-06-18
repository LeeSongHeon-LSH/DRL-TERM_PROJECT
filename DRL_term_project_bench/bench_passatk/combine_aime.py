#!/usr/bin/env python3
"""
Combine AIME 2023/2024/2025 per-problem logs into a single 89-problem set and
generate a unified Pass@K markdown report.

Each model's three yearly JSONL files (one problem per line, with per-problem
pass@k already computed) are concatenated and re-aggregated with the existing
``aggregate_metrics`` so confidence intervals are recomputed over all 89 problems.

The red "Qwen2.5-1.5B-Instruct (from image)" line has no per-problem data, so its
yearly hardcoded numbers are combined via a problem-count weighted average.

Usage:
    python -m bench_passatk.combine_aime
    python -m bench_passatk.combine_aime --out_path runs/AIME_summary_combined.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from bench_passatk.eval.metrics import aggregate_metrics

# Years to merge and their problem counts (AIME2023 = 29: I+II combined).
YEARS = ["AIME2023", "AIME2024", "AIME2025"]

# Real models with per-problem JSONL logs. Order/labels/colors mirror plot_passk.py.
MODELS = [
    {"name": "PAV", "dir": "runs/pav-checkpoint-500",
     "model_path": "PAV-distribution-test/checkpoint-500", "desc": "PAV distribution 학습 모델",
     "color": "#1f77b4", "marker": "o"},
    {"name": "PAV-scalar-c2", "dir": "runs/pav-scalar-c2-checkpoint-500",
     "model_path": "PAV-scalar-c2-test/checkpoint-500", "desc": "PAV scalar-c2 학습 모델",
     "color": "#ff7f0e", "marker": "s"},
    {"name": "Baseline", "dir": "runs/qwen-math-1.5b-baseline",
     "model_path": "Qwen/Qwen2.5-Math-1.5B-Instruct", "desc": "베이스라인 (Qwen2.5-Math-1.5B-Instruct)",
     "color": "#2ca02c", "marker": "^"},
]

# "from image" Qwen line styling (matches plot_passk.py red diamond).
IMAGE_LABEL = "Qwen2.5-1.5B-Instruct (from image)"
IMAGE_COLOR = "#d62728"
IMAGE_MARKER = "D"

# "from image" Qwen line: yearly hardcoded pass@k (source: plot_passk.py:47-51).
IMAGE_BY_YEAR = {
    "AIME2023": {1: 0.054, 8: 0.083, 64: 0.167, 256: 0.267},
    "AIME2024": {1: 0.028, 8: 0.121, 64: 0.306, 256: 0.467},
    "AIME2025": {1: 0.010, 8: 0.065, 64: 0.231, 256: 0.367},
}

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of per-problem dicts."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_combined_results(model_dir: Path) -> List[Dict]:
    """Concatenate the three yearly JSONL files for one model (89 problems)."""
    combined = []
    for year in YEARS:
        path = model_dir / f"{year}.jsonl"
        combined.extend(load_jsonl(path))
    return combined


def weighted_image_passk() -> Dict[int, float]:
    """Problem-count weighted average of the image-line yearly pass@k."""
    counts = {"AIME2023": 29, "AIME2024": 30, "AIME2025": 30}
    total = sum(counts.values())
    ks = sorted(IMAGE_BY_YEAR["AIME2023"].keys())
    return {
        k: sum(IMAGE_BY_YEAR[y][k] * counts[y] for y in YEARS) / total
        for k in ks
    }


def fmt(v: float) -> str:
    return f"{v:.4f}"


def build_report(aggs: Dict[str, Dict], n_total: int, image_passk: Dict[int, float]) -> str:
    """Render the unified markdown report."""
    L = []
    L.append("# AIME 2023–2025 Combined 통합 리포트 (89문제)\n")
    L.append("> 생성일: 2026-06-11")
    L.append("> 출처: `runs/pav-checkpoint-500`, `runs/pav-scalar-c2-checkpoint-500`, "
             "`runs/qwen-math-1.5b-baseline`")
    L.append("> 생성 방식: 세 연도(AIME2023/2024/2025) 문제별 로그를 합쳐 재집계 "
             "(`python -m bench_passatk.combine_aime`)\n")

    # 1. 개요
    L.append("## 1. 개요\n")
    L.append("AIME 2023 / 2024 / 2025를 **하나의 89문제 집합으로 합쳐** Pass@K 방식으로 재집계한 "
             "통합 리포트입니다. 연도별(29~30문제)로는 한 문제 차이(≈3.3%p)가 지표에 과도하게 "
             "반영되고 95% 신뢰구간이 매우 넓었으나, 89문제로 합치면 표본이 약 3배가 되어 지표가 "
             "안정되고 CI가 좁아집니다.\n")
    L.append("| 약칭 | 모델 경로 | 설명 |")
    L.append("|------|-----------|------|")
    for m in MODELS:
        bold = "**" if m["name"] != "PAV" else ""
        L.append(f"| {bold}{m['name']}{bold} | `{m['model_path']}` | {m['desc']} |")
    L.append("| Qwen (image) | `Qwen2.5-1.5B-Instruct (from image)` | "
             "이미지 출처 값 (연도별 가중평균, 비교용) |\n")

    L.append("### 공통 평가 설정\n")
    L.append("| 파라미터 | 값 |")
    L.append("|----------|-----|")
    for k, v in [("Backend", "vLLM"), ("K (samples)", "256"), ("Temperature", "0.7"),
                 ("Top-p", "0.95"), ("Max Tokens", "2048"), ("Seed", "42"),
                 ("GPU", "NVIDIA GeForce RTX 3090 Ti (25.3 GB)")]:
        L.append(f"| {k} | {v} |")
    L.append("")

    # 2. 통합 데이터셋 정의
    L.append("---\n")
    L.append("## 2. 통합 데이터셋 정의\n")
    L.append("| 데이터셋 | 문제 수 |")
    L.append("|----------|---------|")
    L.append("| AIME2023 (I+II) | 29 |")
    L.append("| AIME2024 | 30 |")
    L.append("| AIME2025 | 30 |")
    L.append(f"| **AIME 2023–2025 Combined** | **{n_total}** |\n")
    L.append("> 참고: 통상 \"AIME 90문제\"로 부르지만, 본 로그상 AIME2023은 I/II 합산 29문제로 "
             "기록되어 실제 통합 문제 수는 **89**입니다.\n")

    # 3. 핵심 지표 종합표
    L.append("---\n")
    L.append("## 3. 핵심 지표 종합표 (95% CI 병기)\n")
    L.append("각 셀: 값 [95% CI]. Pass@k는 불편추정량(Codex estimator)의 문제별 평균.\n")
    L.append("| 모델 | Pass@1 | Pass@256 | Best-of-256 | Majority@256 | Oracle@256 |")
    L.append("|------|--------|----------|-------------|--------------|------------|")
    for m in MODELS:
        agg = aggs[m["name"]]
        p1 = agg["pass@1_explicit"]
        p256 = agg["pass@k"][256]
        bon = agg["bon@n"][256]
        maj = agg["maj@k"][256]
        orc = agg["oracle"]
        L.append(
            f"| {m['name']} "
            f"| {fmt(p1['accuracy'])} [{fmt(p1['ci_low'])}, {fmt(p1['ci_high'])}] "
            f"| {fmt(p256['mean'])} [{fmt(p256['ci_low'])}, {fmt(p256['ci_high'])}] "
            f"| {fmt(bon['accuracy'])} [{fmt(bon['ci_low'])}, {fmt(bon['ci_high'])}] "
            f"| {fmt(maj['accuracy'])} [{fmt(maj['ci_low'])}, {fmt(maj['ci_high'])}] "
            f"| {fmt(orc['accuracy'])} [{fmt(orc['ci_low'])}, {fmt(orc['ci_high'])}] |"
        )
    L.append("")

    # 4. Pass@K 곡선 상세
    L.append("---\n")
    L.append("## 4. Pass@K 곡선 상세 (통합, 불편추정량)\n")
    L.append("![pass@k combined](passk_combined.png)\n")
    L.append("> Qwen (image)는 연도별 하드코딩 값(k∈{1,8,64,256})의 문제수 가중평균이며, "
             "나머지 k는 데이터가 없어 `—`로 표기합니다.\n")
    L.append("| k | PAV | PAV-scalar-c2 | Baseline | Qwen (image) |")
    L.append("|---|-----|---------------|----------|--------------|")
    for k in K_VALUES:
        cells = [str(k)]
        for m in MODELS:
            cells.append(fmt(aggs[m["name"]]["pass@k"][k]["mean"]))
        cells.append(fmt(image_passk[k]) if k in image_passk else "—")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # 5. 메모
    L.append("---\n")
    L.append("## 5. 메모\n")
    L.append(f"- 통합 표본이 89문제로 늘어나 95% 신뢰구간이 연도별(29~30문제) 대비 좁아졌습니다.")
    L.append("- Pass@k 통합값은 89개 문제별 pass@k의 단순 평균으로, 연도별 값의 문제수 "
             "(29/30/30) 가중평균과 일치합니다.")
    L.append("- Qwen (image) 행은 문제별 데이터가 없어 연도별 수치의 가중평균으로만 비교 "
             "참고용으로 포함했습니다.")
    L.append("")

    return "\n".join(L)


def plot_combined(aggs: Dict[str, Dict], n_total: int, image_passk: Dict[int, float],
                  out_path: Path):
    """Single-panel Pass@K curve for the combined 89-problem set."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in MODELS:
        passk = aggs[m["name"]]["pass@k"]
        ks = K_VALUES
        values = [passk[k]["mean"] for k in ks]
        ax.plot(ks, values, color=m["color"], marker=m["marker"], markersize=8,
                linewidth=2, label=m["name"], markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=m["color"])
        for k, v in zip(ks, values):
            if k in (1, 8, 64, 256):
                ax.annotate(f"{v:.3f}", xy=(k, v), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=9,
                            color=m["color"], fontweight="bold")

    # Image line: only k in {1,8,64,256}.
    img_ks = sorted(image_passk.keys())
    img_vals = [image_passk[k] for k in img_ks]
    ax.plot(img_ks, img_vals, color=IMAGE_COLOR, marker=IMAGE_MARKER, markersize=8,
            linewidth=2, label=IMAGE_LABEL, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=IMAGE_COLOR)
    for k, v in zip(img_ks, img_vals):
        ax.annotate(f"{v:.3f}", xy=(k, v), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=IMAGE_COLOR, fontweight="bold")

    ax.set_title(f"AIME 2023–2025 Combined ({n_total} problems)",
                 fontsize=15, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 64, 256])
    ax.set_xticklabels(["1", "8", "64", "256"], fontsize=12)
    ax.set_ylim(-0.02, 0.6)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylabel("pass@k", fontsize=12)
    ax.set_xlabel("Number of Samples k", fontsize=12)
    ax.legend(fontsize=10, frameon=True, loc="upper left")

    fig.text(0.5, -0.02,
             "Figure: pass@k on AIME 2023–2025 combined (89 problems)",
             ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Combine AIME years into one 89-problem report.")
    parser.add_argument("--out_path", type=str, default="runs/AIME_summary_combined.md",
                        help="Output markdown path.")
    parser.add_argument("--plot_path", type=str, default="runs/passk_combined.png",
                        help="Output plot image path.")
    args = parser.parse_args()

    aggs = {}
    n_total = None
    for m in MODELS:
        results = load_combined_results(Path(m["dir"]))
        agg = aggregate_metrics(results, k_values=K_VALUES)
        aggs[m["name"]] = agg
        if n_total is None:
            n_total = agg["n_problems"]
        print(f"{m['name']}: n_problems={agg['n_problems']}, "
              f"pass@256={agg['pass@k'][256]['mean']:.4f}")

    image_passk = weighted_image_passk()
    print(f"Qwen (image, weighted): pass@256={image_passk[256]:.4f}")

    report = build_report(aggs, n_total, image_passk)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_combined(aggs, n_total, image_passk, plot_path)


if __name__ == "__main__":
    main()
