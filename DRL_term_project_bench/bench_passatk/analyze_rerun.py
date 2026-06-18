#!/usr/bin/env python3
"""
Analyze the AIME 2023-2025 re-run (seed=42, K=256) vs the original logs.

For each model it:
  - aggregates the combined 89-problem metrics for both original and re-run,
  - quantifies reproducibility at the problem level (correct-count agreement,
    oracle flips), since the re-run used the SAME seed.

Outputs runs/AIME_rerun_analysis.md.

Usage:
    python -m bench_passatk.analyze_rerun
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from bench_passatk.eval.metrics import aggregate_metrics

YEARS = ["AIME2023", "AIME2024", "AIME2025"]

# (display name, original dir, rerun dir)
MODELS = [
    ("PAV", "runs/pav-checkpoint-500", "runs/pav-checkpoint-500-rerun"),
    ("PAV-scalar-c2", "runs/pav-scalar-c2-checkpoint-500", "runs/pav-scalar-c2-checkpoint-500-rerun"),
    ("Baseline", "runs/qwen-math-1.5b-baseline", "runs/qwen-math-1.5b-baseline-rerun"),
]

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SHOW_KS = [1, 8, 64, 256]


def load_combined(model_dir: str) -> List[Dict]:
    out = []
    for year in YEARS:
        with open(Path(model_dir) / f"{year}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def by_id(results: List[Dict]) -> Dict[str, Dict]:
    return {r["problem_id"]: r for r in results}


def fmt(v):
    return f"{v:.4f}"


def main():
    L = []
    L.append("# AIME 2023–2025 Re-run 분석 (seed=42, K=256, 89문제)\n")
    L.append("> 생성일: 2026-06-11")
    L.append("> 비교: 원본 로그 vs 재실행(`*-rerun`). 동일 seed=42 → 재현성 확인이 목적.")
    L.append("> vLLM은 배치 스케줄링 때문에 비트 단위로 결정적이지 않으므로, 문제별 정답 수 "
             "c가 소폭 달라질 수 있다.\n")

    rows_metric = []   # for summary
    rerun_aggs = {}

    for name, orig_dir, rerun_dir in MODELS:
        orig = load_combined(orig_dir)
        rerun = load_combined(rerun_dir)
        o_by, r_by = by_id(orig), by_id(rerun)
        ids = [pid for pid in o_by if pid in r_by]
        n = len(ids)

        agg_o = aggregate_metrics(orig, k_values=K_VALUES)
        agg_r = aggregate_metrics(rerun, k_values=K_VALUES)
        rerun_aggs[name] = agg_r

        # Problem-level reproducibility on correct-count c (out of 256).
        c_o = np.array([o_by[pid]["per_problem"]["c"] for pid in ids], dtype=float)
        c_r = np.array([r_by[pid]["per_problem"]["c"] for pid in ids], dtype=float)
        dc = c_r - c_o
        oracle_o = np.array([1 if o_by[pid]["per_problem"].get("oracle") else 0 for pid in ids])
        oracle_r = np.array([1 if r_by[pid]["per_problem"].get("oracle") else 0 for pid in ids])
        oracle_flips = int(np.sum(oracle_o != oracle_r))
        identical_c = int(np.sum(dc == 0))
        corr = float(np.corrcoef(c_o, c_r)[0, 1]) if n > 1 else float("nan")

        rows_metric.append({
            "name": name, "n": n,
            "agg_o": agg_o, "agg_r": agg_r,
            "mean_c_o": float(c_o.mean()), "mean_c_r": float(c_r.mean()),
            "mean_abs_dc": float(np.abs(dc).mean()), "max_abs_dc": float(np.abs(dc).max()),
            "identical_c": identical_c, "oracle_flips": oracle_flips, "corr": corr,
        })

    # 1. Reproducibility summary
    L.append("---\n")
    L.append("## 1. 재현성 요약 (문제별 정답 수 c, 256개 중)\n")
    L.append("| 모델 | 평균 c (원본→rerun) | 평균 \\|Δc\\| | 최대 \\|Δc\\| | c 동일 문제 | Oracle 뒤집힘 | c 상관계수 |")
    L.append("|------|---------------------|-----------|-----------|------------|--------------|-----------|")
    for r in rows_metric:
        L.append(f"| {r['name']} | {r['mean_c_o']:.2f} → {r['mean_c_r']:.2f} "
                 f"| {r['mean_abs_dc']:.2f} | {r['max_abs_dc']:.0f} "
                 f"| {r['identical_c']}/{r['n']} | {r['oracle_flips']} | {r['corr']:.4f} |")
    L.append("")

    # 2. Aggregate metric comparison (original vs rerun)
    L.append("---\n")
    L.append("## 2. 통합 지표 비교 (원본 vs rerun)\n")
    for r in rows_metric:
        name = r["name"]; ao = r["agg_o"]; ar = r["agg_r"]
        L.append(f"### {name}\n")
        L.append("| 지표 | 원본 | rerun | Δ |")
        L.append("|------|------|-------|-----|")
        # pass@k
        for k in SHOW_KS:
            vo = ao["pass@k"][k]["mean"]; vr = ar["pass@k"][k]["mean"]
            L.append(f"| Pass@{k} | {fmt(vo)} | {fmt(vr)} | {vr-vo:+.4f} |")
        # pass@1 explicit, maj@256, oracle
        po, pr = ao["pass@1_explicit"]["accuracy"], ar["pass@1_explicit"]["accuracy"]
        L.append(f"| Pass@1 (first) | {fmt(po)} | {fmt(pr)} | {pr-po:+.4f} |")
        mo, mr = ao["maj@k"][256]["accuracy"], ar["maj@k"][256]["accuracy"]
        L.append(f"| Majority@256 | {fmt(mo)} | {fmt(mr)} | {mr-mo:+.4f} |")
        oo, orr = ao["oracle"]["accuracy"], ar["oracle"]["accuracy"]
        L.append(f"| Oracle@256 | {fmt(oo)} | {fmt(orr)} | {orr-oo:+.4f} |")
        L.append("")

    # 3. Rerun cross-model comparison (the actual "synthesis" of rerun)
    L.append("---\n")
    L.append("## 3. Rerun 기준 모델 간 통합 비교 (89문제)\n")
    L.append("| k | PAV | PAV-scalar-c2 | Baseline |")
    L.append("|---|-----|---------------|----------|")
    for k in K_VALUES:
        cells = [str(k)]
        for name, *_ in MODELS:
            cells.append(fmt(rerun_aggs[name]["pass@k"][k]["mean"]))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("| 지표 | PAV | PAV-scalar-c2 | Baseline |")
    L.append("|------|-----|---------------|----------|")
    for label, getter in [
        ("Pass@1 (first)", lambda a: a["pass@1_explicit"]["accuracy"]),
        ("Majority@256", lambda a: a["maj@k"][256]["accuracy"]),
        ("Oracle@256", lambda a: a["oracle"]["accuracy"]),
    ]:
        cells = [label] + [fmt(getter(rerun_aggs[name])) for name, *_ in MODELS]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    out = Path("runs/AIME_rerun_analysis.md")
    out.write_text("\n".join(L), encoding="utf-8")
    # console summary
    for r in rows_metric:
        print(f"{r['name']}: pass@256 orig={r['agg_o']['pass@k'][256]['mean']:.4f} "
              f"rerun={r['agg_r']['pass@k'][256]['mean']:.4f} | "
              f"mean|Δc|={r['mean_abs_dc']:.2f}, identical c={r['identical_c']}/{r['n']}, "
              f"oracle flips={r['oracle_flips']}")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
