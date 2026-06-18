#!/usr/bin/env python3
"""
Paired model comparison on the combined AIME 2023-2025 set (89 problems).

Because all three models are evaluated on the *same* problems, a paired test
removes the large between-problem variance (some problems are just hard for
everyone) and gives far more statistical power than comparing two wide,
overlapping confidence intervals.

Two complementary tests are run for each model pair:

  - McNemar's exact test on binary per-problem outcomes
    (pass@1_explicit, maj@256, bon@256, oracle@256). Uses only the discordant
    problems (one model right, the other wrong) — the matched pairs.

  - Paired bootstrap on continuous per-problem pass@k (k = 1/8/64/256):
    resample the 89 problems with replacement, recompute mean_A - mean_B, and
    build a 95% CI / two-sided p for the difference. A Wilcoxon signed-rank
    test is reported alongside as a non-parametric cross-check.

The "from image" Qwen line has no per-problem data and is therefore excluded.

Usage:
    python -m bench_passatk.paired_compare
    python -m bench_passatk.paired_compare --n_bootstrap 10000 --seed 42
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

YEARS = ["AIME2023", "AIME2024", "AIME2025"]

MODELS = [
    {"name": "PAV", "dir": "runs/pav-checkpoint-500"},
    {"name": "PAV-scalar-c2", "dir": "runs/pav-scalar-c2-checkpoint-500"},
    {"name": "Baseline", "dir": "runs/qwen-math-1.5b-baseline"},
]

# Pairs to test (A vs B); positive diff => A better.
PAIRS = [("PAV", "Baseline"), ("PAV-scalar-c2", "Baseline"), ("PAV", "PAV-scalar-c2")]

CONT_KS = [1, 8, 64, 256]          # continuous pass@k for paired bootstrap
BINARY_METRICS = [                  # (per_problem key, display label)
    ("pass@1_explicit", "Pass@1 (first sample)"),
    ("maj@256", "Majority@256"),
    ("bon@256", "Best-of-256"),
    ("oracle", "Oracle@256"),
]


def load_combined(model_dir: Path) -> List[Dict]:
    """Concatenate the three yearly JSONL files (problems stay aligned by order)."""
    out = []
    for year in YEARS:
        with open(model_dir / f"{year}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Exact two-sided: 2 * P(X <= k) under Binom(n, 0.5), capped at 1.
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> Dict:
    """Paired bootstrap over problems for the difference of means (A - B)."""
    n = len(a)
    diff_obs = float(np.mean(a) - np.mean(b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(a[idx]) - np.mean(b[idx])
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    # Two-sided p: twice the smaller tail mass around 0.
    p = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    p = min(1.0, p)
    return {"diff": diff_obs, "ci_low": ci_low, "ci_high": ci_high,
            "p": p, "significant": (ci_low > 0 or ci_high < 0)}


def build_report(data: Dict[str, List[Dict]], n_total: int, n_boot: int,
                 seed: int) -> str:
    L = []
    L.append("# AIME 2023–2025 Combined — Paired 모델 비교 (89문제)\n")
    L.append("> 생성일: 2026-06-11")
    L.append(f"> 방식: 동일 문제 짝짓기(paired). McNemar 정확검정 + Paired bootstrap "
             f"(반복 {n_boot:,}회, seed={seed})")
    L.append(f"> 표본: 89문제 (AIME2023 29 + 2024 30 + 2025 30), 모든 모델 동일 문제·동일 순서 확인됨\n")

    L.append("## 왜 paired인가\n")
    L.append("세 모델이 **같은 89문제**를 풀었으므로, 문제 자체의 난이도 차이(누구에게나 어려운 "
             "문제)를 상쇄하고 모델 간 차이만 본다. 넓게 겹치는 두 신뢰구간을 비교하는 것보다 "
             "검정력이 훨씬 높다.\n")

    # Helper to fetch arrays
    def arr(model, key):
        return np.array([float(r["per_problem"][key]) for r in data[model]])

    def barr(model, key):
        return np.array([1 if r["per_problem"].get(key, False) else 0 for r in data[model]])

    # 1. Paired bootstrap on continuous pass@k
    L.append("---\n")
    L.append("## 1. Pass@k Paired Bootstrap (연속값)\n")
    L.append("Δ = A − B (양수면 A 우위). 95% CI가 0을 포함하지 않으면 유의(✅).\n")
    for a_name, b_name in PAIRS:
        L.append(f"### {a_name} vs {b_name}\n")
        L.append("| k | A mean | B mean | Δ | 95% CI (Δ) | p (bootstrap) | Wilcoxon p | 유의? |")
        L.append("|---|--------|--------|-----|------------|---------------|------------|------|")
        rng = np.random.default_rng(seed)
        for k in CONT_KS:
            a = arr(a_name, f"pass@{k}")
            b = arr(b_name, f"pass@{k}")
            res = paired_bootstrap(a, b, n_boot, rng)
            # Wilcoxon signed-rank (skip if all diffs zero)
            d = a - b
            if np.any(d != 0):
                try:
                    w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
                except ValueError:
                    w_p = float("nan")
            else:
                w_p = 1.0
            sig = "✅" if res["significant"] else "❌"
            wp = "n/a" if math.isnan(w_p) else f"{w_p:.3f}"
            L.append(f"| {k} | {np.mean(a):.4f} | {np.mean(b):.4f} | {res['diff']:+.4f} "
                     f"| [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] | {res['p']:.3f} "
                     f"| {wp} | {sig} |")
        L.append("")

    # 2. McNemar on binary outcomes
    L.append("---\n")
    L.append("## 2. McNemar 정확검정 (이진 지표)\n")
    L.append("b = A만 정답, c = B만 정답 (불일치 쌍). 일치 쌍은 검정에 기여하지 않는다.\n")
    for a_name, b_name in PAIRS:
        L.append(f"### {a_name} vs {b_name}\n")
        L.append("| 지표 | A 정답수 | B 정답수 | b (A만) | c (B만) | McNemar p | 유의? |")
        L.append("|------|----------|----------|---------|---------|-----------|------|")
        for key, label in BINARY_METRICS:
            a = barr(a_name, key)
            b = barr(b_name, key)
            b_only = int(np.sum((a == 1) & (b == 0)))
            c_only = int(np.sum((a == 0) & (b == 1)))
            p = mcnemar_exact(b_only, c_only)
            sig = "✅" if p < 0.05 else "❌"
            L.append(f"| {label} | {int(a.sum())} | {int(b.sum())} | {b_only} | {c_only} "
                     f"| {p:.3f} | {sig} |")
        L.append("")

    # 3. Summary
    L.append("---\n")
    L.append("## 3. 요약\n")
    L.append("- 유의(✅)로 표시된 항목만 95% 수준에서 모델 간 차이가 통계적으로 뒷받침된다.")
    L.append("- 모두 ❌이면, paired 비교로 검정력을 높여도 세 모델의 AIME 성능 차이는 "
             "통계적으로 구분되지 않는다는 의미다 (표본/효과크기 한계).")
    L.append("- **다중비교 주의:** 총 24개 검정(쌍 3 × 연속 4 + 쌍 3 × 이진 4)을 수행했다. "
             "α=0.05에서 우연히 1개 내외의 거짓 양성이 기대되므로, 단일 항목이 p≈0.05로 "
             "겨우 유의하게 나오고 다른 검정(예: Wilcoxon)이 이를 뒷받침하지 못한다면 "
             "실제 차이로 보기 어렵다. Bonferroni 보정 시 임계값은 α≈0.002 수준이다.")
    L.append("- 'from image' Qwen 라인은 문제별 데이터가 없어 이 검정에서 제외된다.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Paired comparison on combined AIME (89 problems).")
    ap.add_argument("--out_path", type=str, default="runs/AIME_paired_comparison.md")
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = {m["name"]: load_combined(Path(m["dir"])) for m in MODELS}
    n_total = len(next(iter(data.values())))
    # Sanity: identical problem ordering across models.
    ref = [r["problem_id"] for r in data[MODELS[0]["name"]]]
    for m in MODELS[1:]:
        assert [r["problem_id"] for r in data[m["name"]]] == ref, \
            f"problem_id order mismatch for {m['name']}"

    report = build_report(data, n_total, args.n_bootstrap, args.seed)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"n_problems={n_total}")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
