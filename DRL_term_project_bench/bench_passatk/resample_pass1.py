#!/usr/bin/env python3
"""
Pass@1 Bootstrap Resampling Analysis.

Re-estimates Pass@1 by repeatedly sampling one answer per problem from
the existing K=256 samples, producing a distribution of Pass@1 estimates
with confidence intervals. This avoids the high variance of a single
Pass@1 measurement on small datasets like AIME (29-30 problems).

Usage:
    python -m bench_passatk.resample_pass1 \
        --dirs runs/pav-checkpoint-500 runs/pav-scalar-c2-checkpoint-500 runs/qwen-math-1.5b-baseline \
        --names PAV-distribution PAV-scalar-c2 Qwen-Baseline \
        --datasets AIME2023 AIME2024 AIME2025 MATH \
        --n_bootstrap 1000 \
        --seed 42 \
        --out_dir runs/resample_analysis
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    """Load results from a JSONL file."""
    if not path.exists():
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def bootstrap_pass1(
    problems: List[Dict],
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Bootstrap resampling for Pass@1.

    For each bootstrap iteration:
      1. For each problem, randomly pick one sample (simulating a single inference).
      2. Check if that sample is correct.
      3. Compute Pass@1 = fraction of problems with correct answer.

    This produces a distribution of Pass@1 estimates.

    Args:
        problems: List of problem dicts, each with 'samples' containing
                  'is_correct' fields.
        n_bootstrap: Number of bootstrap iterations.
        rng: NumPy random generator for reproducibility.

    Returns:
        Dictionary with mean, std, CI, and full distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Pre-extract is_correct arrays for each problem
    problem_correct_arrays = []
    for prob in problems:
        samples = prob.get("samples", [])
        if not samples:
            continue
        correct = [int(s.get("is_correct", False)) for s in samples]
        problem_correct_arrays.append(np.array(correct))

    if not problem_correct_arrays:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "distribution": [],
            "n_problems": 0,
        }

    n_problems = len(problem_correct_arrays)
    pass1_values = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        correct_count = 0
        for j, correct_arr in enumerate(problem_correct_arrays):
            # Randomly pick one sample index
            idx = rng.integers(0, len(correct_arr))
            correct_count += correct_arr[idx]
        pass1_values[i] = correct_count / n_problems

    mean = float(np.mean(pass1_values))
    std = float(np.std(pass1_values))
    ci_low = float(np.percentile(pass1_values, 2.5))
    ci_high = float(np.percentile(pass1_values, 97.5))

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "distribution": pass1_values.tolist(),
        "n_problems": n_problems,
    }


def paired_bootstrap_test(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
) -> Dict:
    """
    Perform a paired bootstrap test comparing two distributions.

    Returns the probability that A > B, and the mean difference.
    """
    diff = dist_a - dist_b
    mean_diff = float(np.mean(diff))
    p_a_greater = float(np.mean(dist_a > dist_b))
    p_b_greater = float(np.mean(dist_b > dist_a))

    # 95% CI of the difference
    ci_low = float(np.percentile(diff, 2.5))
    ci_high = float(np.percentile(diff, 97.5))

    return {
        "mean_diff": mean_diff,
        "p_a_greater": p_a_greater,
        "p_b_greater": p_b_greater,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant_95": ci_low > 0 or ci_high < 0,
    }


def generate_report(
    all_results: Dict[str, Dict[str, Dict]],
    model_names: List[str],
    datasets: List[str],
    n_bootstrap: int,
    out_dir: Path,
) -> str:
    """Generate a markdown comparison report."""
    lines = []

    lines.append("# Pass@1 Bootstrap Resampling 비교 분석\n")
    lines.append(f"> 생성일: {datetime.now().isoformat()}\n")
    lines.append(f"> Bootstrap 반복 횟수: {n_bootstrap}\n")
    lines.append("")
    lines.append("기존 K=256 샘플 데이터에서 각 문제별로 무작위 1개의 샘플을 추출하여")
    lines.append("Pass@1을 계산하는 과정을 여러 번 반복하여, 단일 측정의 불확실성을 줄이고")
    lines.append("신뢰 구간을 추정합니다.\n")

    # 1. Pass@1 Summary Table
    lines.append("## 1. Pass@1 요약 (Bootstrap 평균 ± 95% CI)\n")
    lines.append("| 데이터셋 | " + " | ".join(model_names) + " |")
    lines.append("|-----------|" + "|".join(["---"] * len(model_names)) + "|")

    for dataset in datasets:
        row = f"| **{dataset}** |"
        for model in model_names:
            res = all_results.get(model, {}).get(dataset, {})
            mean = res.get("mean", float("nan"))
            ci_low = res.get("ci_low", float("nan"))
            ci_high = res.get("ci_high", float("nan"))
            if np.isnan(mean):
                row += " N/A |"
            else:
                row += f" {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}] |"
        lines.append(row)

    # 2. Detailed per-dataset results
    lines.append("\n## 2. 데이터셋별 상세 결과\n")

    for dataset in datasets:
        lines.append(f"### {dataset}\n")

        # Collect distributions for paired tests
        dists = {}
        for model in model_names:
            res = all_results.get(model, {}).get(dataset, {})
            dist = res.get("distribution", [])
            if dist:
                dists[model] = np.array(dist)

        # Table with statistics
        lines.append("| 모델 | 평균 | 표준편차 | 95% CI | 문제 수 |")
        lines.append("|------|------|----------|--------|---------|")

        for model in model_names:
            res = all_results.get(model, {}).get(dataset, {})
            mean = res.get("mean", float("nan"))
            std = res.get("std", float("nan"))
            ci_low = res.get("ci_low", float("nan"))
            ci_high = res.get("ci_high", float("nan"))
            n = res.get("n_problems", 0)
            if np.isnan(mean):
                lines.append(f"| {model} | N/A | N/A | N/A | {n} |")
            else:
                lines.append(
                    f"| {model} | {mean:.4f} | {std:.4f} | "
                    f"[{ci_low:.4f}, {ci_high:.4f}] | {n} |"
                )

        # Paired bootstrap tests
        if len(dists) >= 2:
            lines.append(f"\n**쌍대 비교 (Paired Bootstrap Test):**\n")
            lines.append("| 비교 | 평균 차이 | P(A>B) | 95% CI (차이) | 유의미? |")
            lines.append("|------|-----------|--------|---------------|---------|")

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    a_name = model_names[i]
                    b_name = model_names[j]
                    if a_name in dists and b_name in dists:
                        test = paired_bootstrap_test(dists[a_name], dists[b_name])
                        sig = "✅" if test["significant_95"] else "❌"
                        lines.append(
                            f"| {a_name} vs {b_name} | "
                            f"{test['mean_diff']:+.4f} | "
                            f"{test['p_a_greater']:.3f} | "
                            f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}] | "
                            f"{sig} |"
                        )

        lines.append("")

    # 3. Key findings
    lines.append("## 3. 핵심 발견\n")

    # Find best model per dataset
    lines.append("### 데이터셋별 최고 모델 (Bootstrap 평균 기준)\n")
    for dataset in datasets:
        best_model = None
        best_mean = -1
        for model in model_names:
            res = all_results.get(model, {}).get(dataset, {})
            mean = res.get("mean", float("nan"))
            if not np.isnan(mean) and mean > best_mean:
                best_mean = mean
                best_model = model
        if best_model:
            lines.append(f"- **{dataset}**: {best_model} ({best_mean:.4f})")

    # Check if differences are significant
    lines.append("\n### 유의미한 차이 (95% 수준)\n")
    sig_count = 0
    for dataset in datasets:
        dists = {}
        for model in model_names:
            res = all_results.get(model, {}).get(dataset, {})
            dist = res.get("distribution", [])
            if dist:
                dists[model] = np.array(dist)

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a_name = model_names[i]
                b_name = model_names[j]
                if a_name in dists and b_name in dists:
                    test = paired_bootstrap_test(dists[a_name], dists[b_name])
                    if test["significant_95"]:
                        direction = a_name if test["mean_diff"] > 0 else b_name
                        lines.append(
                            f"- **{dataset}**: {a_name} vs {b_name} → "
                            f"{direction} 우위 (차이: {abs(test['mean_diff']):.4f})"
                        )
                        sig_count += 1

    if sig_count == 0:
        lines.append("- 95% 수준에서 유의미한 차이가 발견되지 않았습니다.")
    else:
        lines.append(f"\n총 {sig_count}개의 유의미한 차이가 발견되었습니다.")

    # 4. Comparison with original Pass@1
    lines.append("\n## 4. 기존 Pass@1 (단일 측정) vs Bootstrap Pass@1 비교\n")
    lines.append("기존 레포트의 Pass@1은 첫 번째 샘플의 정확도이며,")
    lines.append("Bootstrap Pass@1은 1000회 재샘플링의 평균입니다.\n")
    lines.append("| 데이터셋 | 모델 | 기존 Pass@1 | Bootstrap 평균 | 차이 |")
    lines.append("|-----------|------|-------------|----------------|------|")

    # We'll note that original Pass@1 values come from the reports
    # and bootstrap means come from our analysis
    original_pass1 = {
        "PAV-distribution": {
            "AIME2023": 0.1034,
            "AIME2024": 0.1000,
            "AIME2025": 0.1000,
            "MATH": 0.4615,
        },
        "PAV-scalar-c2": {
            "AIME2023": 0.1034,
            "AIME2024": 0.0333,
            "AIME2025": 0.0667,
            "MATH": 0.4585,
        },
        "Qwen-Baseline": {
            "AIME2023": 0.1034,
            "AIME2024": 0.0789,
            "AIME2025": 0.0667,
            "MATH": 0.4668,
        },
    }

    for dataset in datasets:
        for model in model_names:
            res = all_results.get(model, {}).get(dataset, {})
            boot_mean = res.get("mean", float("nan"))
            orig = original_pass1.get(model, {}).get(dataset, float("nan"))
            if not np.isnan(boot_mean) and not np.isnan(orig):
                diff = boot_mean - orig
                lines.append(
                    f"| {dataset} | {model} | {orig:.4f} | {boot_mean:.4f} | {diff:+.4f} |"
                )

    # Write report
    report_path = out_dir / "resample_pass1_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to: {report_path}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Pass@1 Bootstrap Resampling Analysis"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Directories containing benchmark results (JSONL files).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Model names corresponding to each directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AIME2023", "AIME2024", "AIME2025", "MATH"],
        help="Datasets to analyze.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/resample_analysis",
        help="Output directory for results.",
    )

    args = parser.parse_args()

    if len(args.dirs) != len(args.names):
        print("Error: Number of directories must match number of names.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyze data
    all_results = {}

    for model_dir, model_name in zip(args.dirs, args.names):
        model_path = Path(model_dir)
        all_results[model_name] = {}

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Directory: {model_dir}")
        print(f"{'='*60}")

        for dataset in args.datasets:
            jsonl_path = model_path / f"{dataset}.jsonl"

            if not jsonl_path.exists():
                print(f"  {dataset}: File not found - {jsonl_path}")
                continue

            problems = load_jsonl(jsonl_path)
            if not problems:
                print(f"  {dataset}: No data loaded")
                continue

            print(f"  {dataset}: {len(problems)} problems loaded")

            # Bootstrap Pass@1
            result = bootstrap_pass1(problems, n_bootstrap=args.n_bootstrap, rng=rng)
            all_results[model_name][dataset] = result

            print(
                f"    Pass@1 (bootstrap): {result['mean']:.4f} "
                f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
                f"(std={result['std']:.4f})"
            )

    # Save raw results as JSON
    raw_results_path = out_dir / "resample_pass1_raw.json"
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, datasets in all_results.items():
        json_results[model_name] = {}
        for dataset_name, result in datasets.items():
            json_results[model_name][dataset_name] = {
                k: v for k, v in result.items() if k != "distribution"
            }
            json_results[model_name][dataset_name]["distribution_stats"] = {
                "p10": float(np.percentile(result["distribution"], 10)),
                "p25": float(np.percentile(result["distribution"], 25)),
                "p50": float(np.percentile(result["distribution"], 50)),
                "p75": float(np.percentile(result["distribution"], 75)),
                "p90": float(np.percentile(result["distribution"], 90)),
            }

    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to: {raw_results_path}")

    # Generate report
    report = generate_report(
        all_results, args.names, args.datasets, args.n_bootstrap, out_dir
    )

    # Also save distributions as numpy arrays for further analysis
    for model_name in args.names:
        for dataset in args.datasets:
            dist = all_results.get(model_name, {}).get(dataset, {}).get("distribution", [])
            if dist:
                dist_path = out_dir / f"dist_{model_name}_{dataset}.npy"
                np.save(dist_path, np.array(dist))

    print("\nDone!")


if __name__ == "__main__":
    main()