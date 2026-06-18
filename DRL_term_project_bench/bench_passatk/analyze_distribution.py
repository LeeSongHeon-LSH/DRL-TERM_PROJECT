#!/usr/bin/env python3
"""
Distributional-signature analysis of PAV-distribution vs the other models.

PAV-distribution is trained with a *distributional* reward, so its fingerprint
should show up in the SHAPE of the 256-sample answer distribution, not in the
headline pass@k accuracy. This script computes, per model on the combined
89-problem AIME set:

  - answer diversity     : mean #distinct answers / problem
  - normalized entropy   : spread of the answer distribution (0=peaked, 1=uniform)
  - mode mass            : fraction of samples on the single most common answer
  - correct concentration: among *solvable* problems, fraction of correct samples
                           (how sharply the model piles probability on the right answer)
  - sampling gain        : pass@256 - pass@1 (how much extra samples help)
  - majority efficiency  : maj@256, and P(mode == correct | solvable)

All metrics are also computed paired (same problems) so PAV-distribution can be
contrasted against each model on identical inputs.

Usage:
    python -m bench_passatk.analyze_distribution
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

YEARS = ["AIME2023", "AIME2024", "AIME2025"]
N_SAMPLES = 256

MODELS = [
    ("PAV-distribution", "runs/pav-checkpoint-500"),     # <- our method
    ("PAV-scalar-c2", "runs/pav-scalar-c2-checkpoint-500"),
    ("Baseline", "runs/qwen-math-1.5b-baseline"),
]
FOCUS = "PAV-distribution"


def load_combined(model_dir: str) -> List[Dict]:
    out = []
    for year in YEARS:
        with open(Path(model_dir) / f"{year}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def problem_stats(prob: Dict) -> Dict:
    samples = prob["samples"]
    preds = [str(s.get("pred")) for s in samples]  # None -> "None" bucket
    corr = [bool(s.get("is_correct")) for s in samples]
    n = len(samples)
    counts = Counter(preds)
    freqs = np.array(list(counts.values()), dtype=float)
    p = freqs / freqs.sum()
    entropy = float(-(p * np.log(p)).sum())          # nats
    norm_entropy = entropy / math.log(n)             # 0..1 (max when all distinct)
    mode_mass = float(freqs.max() / n)
    c = sum(corr)
    solved = c > 0
    # is the most common answer the correct one? (self-consistency would pick it)
    gold = str(prob.get("gold"))
    mode_answer = counts.most_common(1)[0][0]
    mode_correct = (mode_answer == gold)
    return {
        "distinct": len(counts),
        "norm_entropy": norm_entropy,
        "mode_mass": mode_mass,
        "correct_frac": c / n,
        "solved": solved,
        "c": c,
        "mode_correct": mode_correct,
        "pass1": prob["per_problem"]["pass@1"],
        "pass256": prob["per_problem"]["pass@256"],
    }


def aggregate(stats: List[Dict]) -> Dict:
    arr = lambda k: np.array([s[k] for s in stats], dtype=float)
    solved = [s for s in stats if s["solved"]]
    return {
        "n": len(stats),
        "distinct": float(arr("distinct").mean()),
        "norm_entropy": float(arr("norm_entropy").mean()),
        "mode_mass": float(arr("mode_mass").mean()),
        "pass1": float(arr("pass1").mean()),
        "pass256": float(arr("pass256").mean()),
        "gain": float(arr("pass256").mean() - arr("pass1").mean()),
        "n_solved": len(solved),
        "correct_conc_solved": float(np.mean([s["correct_frac"] for s in solved])) if solved else float("nan"),
        "mode_correct_solved": float(np.mean([s["mode_correct"] for s in solved])) if solved else float("nan"),
    }


def main():
    data = {name: load_combined(d) for name, d in MODELS}
    # align by problem_id
    ids = [r["problem_id"] for r in data[MODELS[0][0]]]
    by = {name: {r["problem_id"]: r for r in rs} for name, rs in data.items()}

    stats = {name: [problem_stats(by[name][pid]) for pid in ids] for name, _ in MODELS}
    agg = {name: aggregate(stats[name]) for name, _ in MODELS}

    L = []
    L.append("# PAV-distribution 분포적 특징 분석 (AIME 89문제)\n")
    L.append("> 생성일: 2026-06-11")
    L.append("> 목적: 분포형 보상 모델인 **PAV-distribution**이 정확도가 아닌 *출력 분포의 모양*에서 "
             "다른 모델과 어떻게 다른지 본다. 문제당 256샘플의 답(pred) 분포를 분석.\n")

    # 1. Distributional signature table
    L.append("---\n")
    L.append("## 1. 분포 시그니처 (모델별, 89문제 평균)\n")
    L.append("| 지표 | PAV-distribution | PAV-scalar-c2 | Baseline | 의미 |")
    L.append("|------|------------------|---------------|----------|------|")
    rows = [
        ("평균 distinct 답 수", "distinct", "{:.1f}", "샘플 다양성 (256개 중 서로 다른 답 수)"),
        ("정규화 엔트로피", "norm_entropy", "{:.3f}", "분포 퍼짐 (0=한 답에 집중, 1=완전 균일)"),
        ("최빈답 질량(mode mass)", "mode_mass", "{:.3f}", "가장 많이 나온 답의 비율 (집중도)"),
        ("Pass@1 (불편)", "pass1", "{:.4f}", "단일 샘플 기대 정확도"),
        ("Pass@256", "pass256", "{:.4f}", "256샘플 정답 도달률"),
        ("샘플링 이득(256−1)", "gain", "{:.4f}", "샘플을 늘려 얻는 정확도 증가폭"),
    ]
    for label, key, fmt, meaning in rows:
        cells = [fmt.format(agg[name][key]) for name, _ in MODELS]
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {meaning} |")
    L.append("")

    # 2. Conditional on solvable
    L.append("---\n")
    L.append("## 2. 풀 수 있는 문제에서의 행동 (solved: 256개 중 1개라도 정답)\n")
    L.append("| 지표 | PAV-distribution | PAV-scalar-c2 | Baseline | 의미 |")
    L.append("|------|------------------|---------------|----------|------|")
    rows2 = [
        ("풀린 문제 수", "n_solved", "{:.0f}", "oracle 정답 문제 수"),
        ("정답 집중도", "correct_conc_solved", "{:.3f}", "풀린 문제에서 정답 샘플 비율 (정답에 질량 쏠림)"),
        ("최빈답=정답 비율", "mode_correct_solved", "{:.3f}", "다수결이 정답을 고르는 비율 (self-consistency)"),
    ]
    for label, key, fmt, meaning in rows2:
        cells = [fmt.format(agg[name][key]) for name, _ in MODELS]
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {meaning} |")
    L.append("")

    # 3. Paired contrast: PAV-distribution vs each other on same problems
    L.append("---\n")
    L.append(f"## 3. {FOCUS} 의 상대적 특징 (동일 문제 paired)\n")
    L.append("Δ = PAV-distribution − 상대모델. 양수면 PAV-distribution이 더 큼.\n")
    L.append("| 비교 지표 | vs PAV-scalar-c2 | vs Baseline |")
    L.append("|-----------|------------------|-------------|")
    for label, key in [("Δ distinct 답 수", "distinct"),
                       ("Δ 정규화 엔트로피", "norm_entropy"),
                       ("Δ mode mass", "mode_mass"),
                       ("Δ 정답집중도(solved)", "correct_conc_solved")]:
        f = agg[FOCUS][key]
        d_scalar = f - agg["PAV-scalar-c2"][key]
        d_base = f - agg["Baseline"][key]
        L.append(f"| {label} | {d_scalar:+.4f} | {d_base:+.4f} |")
    L.append("")

    # console + write
    out = Path("runs/AIME_distribution_analysis.md")
    out.write_text("\n".join(L), encoding="utf-8")
    for name, _ in MODELS:
        a = agg[name]
        print(f"{name}: distinct={a['distinct']:.1f} entropy={a['norm_entropy']:.3f} "
              f"mode_mass={a['mode_mass']:.3f} corr_conc(solved)={a['correct_conc_solved']:.3f} "
              f"mode=correct={a['mode_correct_solved']:.3f}")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
