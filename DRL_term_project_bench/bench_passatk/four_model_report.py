#!/usr/bin/env python3
"""
Four-model comparison report on the combined AIME 2023-2025 set (89 problems).

Models:
  1. PAV-distribution+few-shot  (runs/pav-fewshot-ckpt500-aime3, --few_shot eval)
  2. PAV-distribution           (runs/pav-checkpoint-500, zero-shot)
  3. PAV-scalar-c2              (runs/pav-scalar-c2-checkpoint-500, zero-shot)
  4. Baseline (Qwen2.5-Math)    (runs/qwen-math-1.5b-baseline, zero-shot)

Produces runs/AIME_4model_comparison.md + runs/AIME_4model_passk.png.
Reuses aggregate_metrics; paired bootstrap on the same 89 problems.

Usage:
    python -m bench_passatk.four_model_report
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_passatk.eval.metrics import aggregate_metrics

YEARS = ["AIME2023", "AIME2024", "AIME2025"]
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# (display name, dir, setup note, color, marker)
MODELS = [
    ("PAV-dist+few-shot", "runs/pav-fewshot-ckpt500-aime3", "few-shot (Step k:)", "#d62728", "D"),
    ("PAV-distribution", "runs/pav-checkpoint-500", "zero-shot", "#1f77b4", "o"),
    ("PAV-scalar-c2", "runs/pav-scalar-c2-checkpoint-500", "zero-shot", "#ff7f0e", "s"),
    ("Baseline", "runs/qwen-math-1.5b-baseline", "zero-shot", "#2ca02c", "^"),
]
NB = 10000
SEED = 42


def load(d):
    out = []
    for y in YEARS:
        for line in open(Path(d) / (y + ".jsonl"), encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def divstats(r):
    dis, ent, mm = [], [], []
    for p in r:
        preds = [str(s.get("pred")) for s in p["samples"]]
        c = Counter(preds)
        f = np.array(list(c.values()), float)
        pr = f / f.sum()
        dis.append(len(c))
        ent.append(float(-(pr * np.log(pr)).sum()) / math.log(len(preds)))
        mm.append(f.max() / len(preds))
    return np.mean(dis), np.mean(ent), np.mean(mm)


def paired_boot(a, b, rng, nb=NB):
    n = len(a)
    d = float(a.mean() - b.mean())
    bs = np.empty(nb)
    for i in range(nb):
        idx = rng.integers(0, n, n)
        bs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min(float((bs <= 0).mean()), float((bs >= 0).mean())))
    return d, float(lo), float(hi), p


def main():
    names = [m[0] for m in MODELS]
    data = {m[0]: load(m[1]) for m in MODELS}
    aggs = {m[0]: aggregate_metrics(load(m[1])) for m in MODELS}
    divs = {m[0]: divstats(data[m[0]]) for m in MODELS}

    # alignment for paired tests
    ids = [x["problem_id"] for x in data[names[0]]]
    B = {n: {x["problem_id"]: x for x in data[n]} for n in names}
    aligned = all(set(B[n]) == set(B[names[0]]) for n in names)

    L = []
    L.append("# AIME 2023–2025 — 4개 모델 비교 리포트 (89문제)\n")
    L.append("> 생성일: 2026-06-12")
    L.append("> 표본: 89문제 (AIME2023 29 + 2024 30 + 2025 30), K=256, temp=0.7, top-p=0.95, seed=42, vLLM\n")
    L.append("| 모델 | 디렉터리 | 셋업 |")
    L.append("|------|----------|------|")
    for n, d, setup, _, _ in MODELS:
        L.append(f"| {n} | `{d}` | {setup} |")
    L.append("\n> ⚠️ **교란 주의:** `PAV-dist+few-shot`만 fewshot-학습 체크포인트 + `--few_shot`(Step k: 형식) "
             "추론입니다. 나머지 셋은 일반 체크포인트 + zero-shot. 따라서 이 모델의 차이는 **보상 형태**가 아니라 "
             "**few-shot 셋업** 때문일 수 있습니다(분리 불가).\n")

    # 1. Pass@k table
    L.append("---\n")
    L.append("## 1. Pass@k (불편추정량)\n")
    L.append("| k | " + " | ".join(names) + " |")
    L.append("|---|" + "|".join(["---"] * len(names)) + "|")
    for k in K_VALUES:
        L.append(f"| {k} | " + " | ".join(f"{aggs[n]['pass@k'][k]['mean']:.4f}" for n in names) + " |")
    L.append("")

    # 2. Summary metrics
    L.append("---\n")
    L.append("## 2. 요약 지표\n")
    L.append("| 지표 | " + " | ".join(names) + " |")
    L.append("|------|" + "|".join(["---"] * len(names)) + "|")
    def row(lab, fn):
        L.append(f"| {lab} | " + " | ".join(f"{fn(aggs[n]):.4f}" for n in names) + " |")
    row("Pass@1 (first)", lambda a: a["pass@1_explicit"]["accuracy"])
    row("Pass@256", lambda a: a["pass@k"][256]["mean"])
    row("Majority@256", lambda a: a["maj@k"][256]["accuracy"])
    row("Oracle@256", lambda a: a["oracle"]["accuracy"])
    row("Sampling gain (256−1)", lambda a: a["pass@k"][256]["mean"] - a["pass@k"][1]["mean"])
    L.append("")

    # 3. Diversity
    L.append("---\n")
    L.append("## 3. 출력 다양성 (256샘플 답 분포)\n")
    L.append("| 지표 | " + " | ".join(names) + " |")
    L.append("|------|" + "|".join(["---"] * len(names)) + "|")
    for i, lab in enumerate(["평균 distinct 답 수", "정규화 엔트로피", "mode_mass(집중도)"]):
        L.append(f"| {lab} | " + " | ".join(f"{divs[n][i]:.4f}" for n in names) + " |")
    L.append("")

    # 4. Paired bootstrap
    L.append("---\n")
    L.append(f"## 4. Paired Bootstrap (동일 89문제, {NB:,}회)\n")
    if not aligned:
        L.append("> ⚠️ 문제 정렬 불일치 — paired 검정 생략.\n")
    else:
        L.append("Δ = A − B. 95% CI가 0을 제외하면 유의(✅).\n")
        rng = np.random.default_rng(SEED)
        def parr(n, key):
            return np.array([B[n][i]["per_problem"][key] for i in ids], float)
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        L.append("| 비교 (A vs B) | Δ Pass@256 | 95% CI | p | 유의? |")
        L.append("|---------------|-----------|--------|---|------|")
        for a, b in pairs:
            d, lo, hi, p = paired_boot(parr(a, "pass@256"), parr(b, "pass@256"), rng)
            sig = "✅" if (lo > 0 or hi < 0) else "❌"
            L.append(f"| {a} vs {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {sig} |")
        L.append("")
        # few-shot across k
        L.append("**few-shot 모델의 k별 우위 (vs PAV-distribution / vs Baseline):**\n")
        L.append("| k | Δ vs PAV-dist [CI] p | Δ vs Baseline [CI] p |")
        L.append("|---|----------------------|----------------------|")
        rng = np.random.default_rng(SEED)
        for k in [8, 64, 256]:
            fa = parr("PAV-dist+few-shot", f"pass@{k}")
            d1, lo1, hi1, p1 = paired_boot(fa, parr("PAV-distribution", f"pass@{k}"), rng)
            d2, lo2, hi2, p2 = paired_boot(fa, parr("Baseline", f"pass@{k}"), rng)
            L.append(f"| {k} | {d1:+.4f} [{lo1:+.4f},{hi1:+.4f}] p={p1:.3f} | "
                     f"{d2:+.4f} [{lo2:+.4f},{hi2:+.4f}] p={p2:.3f} |")
        L.append("")

    # 5. Interpretation
    L.append("---\n")
    L.append("## 5. 해석\n")
    L.append("- **pass@1은 네 모델 모두 ~0.08로 동일** — 단일 샘플 정확도 차이 없음.")
    L.append("- **few-shot 모델만 곡선이 가파름**: pass@256 0.40로 나머지(~0.34) 대비 +5~6문제, "
             "sampling gain도 최대. 출력 다양성(distinct 34.9, 엔트로피 0.47)이 가장 높아 "
             "**다양성→커버리지** 패턴이 일관됨.")
    L.append("- **그러나 paired bootstrap에서 유의하지 않음**(vs Baseline p≈0.07로 경계선). 89문제로는 "
             "+5~6문제도 노이즈와 구분 불가.")
    L.append("- **일반 3개 모델(PAV-distribution / scalar-c2 / Baseline)은 모든 지표에서 사실상 동일** — "
             "앞선 분석과 일치(차이 ≈ 실행 간 노이즈).")
    L.append("- **교란 미분리**: few-shot 모델의 우위는 보상 형태가 아니라 few-shot 프롬프트/체크포인트 때문일 수 "
             "있음. 확정하려면 scalar-c2·Baseline도 `--few_shot`으로 평가해야 함.\n")

    out_md = Path("runs/AIME_4model_comparison.md")
    out_md.write_text("\n".join(L), encoding="utf-8")

    # plot
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
    ax.set_title("AIME 2023–2025 Combined (89 problems) — 4 models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    out_png = Path("runs/AIME_4model_passk.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

    for n in names:
        print(f"{n}: pass@256={aggs[n]['pass@k'][256]['mean']:.4f} "
              f"distinct={divs[n][0]:.1f} entropy={divs[n][1]:.3f}")
    print(f"\nReport: {out_md}\nPlot:   {out_png}")


if __name__ == "__main__":
    main()
