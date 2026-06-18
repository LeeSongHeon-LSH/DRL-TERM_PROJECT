#!/usr/bin/env python3
"""
Compare benchmark results between base model and trained model.

Usage:
    python -m bench_passatk.compare \
        --base_dir runs/base_model/ \
        --trained_dir runs/trained_model/ \
        --output comparison_report.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def load_results(result_dir: Path) -> Dict[str, List]:
    """Load all results from a directory."""
    results = {}
    for jsonl_file in result_dir.glob("*.jsonl"):
        dataset_name = jsonl_file.stem
        with open(jsonl_file, "r") as f:
            results[dataset_name] = [json.loads(line) for line in f if line.strip()]
    return results


def compute_comparison_stats(
    base_values: List[float],
    trained_values: List[float],
    confidence: float = 0.95,
) -> Dict:
    """Compute comparison statistics between two sets of values."""
    base_mean = np.mean(base_values)
    trained_mean = np.mean(trained_values)
    
    # Difference
    diff = trained_mean - base_mean
    
    # Paired t-test for significance
    if len(base_values) == len(trained_values) and len(base_values) > 1:
        t_stat, p_value = stats.ttest_rel(trained_values, base_values)
    else:
        # Unpaired t-test
        t_stat, p_value = stats.ttest_ind(trained_values, base_values)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(base_values) + np.var(trained_values)) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    # Confidence interval for difference
    n = len(base_values)
    std_err = np.std([t - b for t, b in zip(trained_values, base_values)]) / np.sqrt(n)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_low = diff - z * std_err
    ci_high = diff + z * std_err
    
    return {
        "base_mean": base_mean,
        "trained_mean": trained_mean,
        "diff": diff,
        "diff_pct": (diff / base_mean * 100) if base_mean != 0 else 0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
    }


def compare_results(
    base_results: Dict[str, List],
    trained_results: Dict[str, List],
    metrics: List[str] = None,
) -> Dict:
    """Compare results between base and trained model."""
    if metrics is None:
        metrics = ["pass@1", "pass@4", "pass@16", "pass@64", "pass@256",
                   "maj@1", "maj@4", "maj@16", "maj@64", "maj@256",
                   "bon@1", "bon@4", "bon@16", "bon@64", "bon@256"]
    
    comparison = {}
    
    for dataset_name in base_results:
        if dataset_name not in trained_results:
            continue
        
        base_data = base_results[dataset_name]
        trained_data = trained_results[dataset_name]
        
        dataset_comparison = {}
        
        for metric in metrics:
            # Extract metric values
            base_values = []
            trained_values = []
            
            for b, t in zip(base_data, trained_data):
                b_metric = b.get("per_problem", {}).get(metric)
                t_metric = t.get("per_problem", {}).get(metric)
                
                if b_metric is not None and t_metric is not None:
                    base_values.append(float(b_metric))
                    trained_values.append(float(t_metric))
            
            if base_values and trained_values:
                dataset_comparison[metric] = compute_comparison_stats(
                    base_values, trained_values
                )
        
        # Oracle comparison
        base_oracle = [1 if b.get("per_problem", {}).get("oracle", False) else 0 
                       for b in base_data]
        trained_oracle = [1 if t.get("per_problem", {}).get("oracle", False) else 0 
                         for t in trained_data]
        
        dataset_comparison["oracle"] = compute_comparison_stats(
            base_oracle, trained_oracle
        )
        
        comparison[dataset_name] = dataset_comparison
    
    return comparison


def generate_comparison_report(
    comparison: Dict,
    base_name: str,
    trained_name: str,
    output_path: Path,
):
    """Generate a markdown comparison report."""
    lines = []
    
    # Header
    lines.append("# Model Comparison Report\n")
    lines.append(f"\n**Base Model:** `{base_name}`")
    lines.append(f"\n**Trained Model:** `{trained_name}`\n")
    
    # Summary table
    lines.append("\n## Summary\n")
    lines.append("\n| Dataset | Pass@1 Δ | Pass@256 Δ | Maj@256 Δ | Oracle Δ |")
    lines.append("\n|---------|----------|------------|-----------|----------|")
    
    for dataset_name, metrics in comparison.items():
        p1 = metrics.get("pass@1", {})
        p256 = metrics.get("pass@256", {})
        m256 = metrics.get("maj@256", {})
        oracle = metrics.get("oracle", {})
        
        def fmt_delta(d):
            if not d:
                return "N/A"
            diff = d.get("diff", 0)
            sig = "*" if d.get("significant", False) else ""
            return f"{diff:+.4f}{sig}"
        
        lines.append(f"\n| {dataset_name} | {fmt_delta(p1)} | {fmt_delta(p256)} | {fmt_delta(m256)} | {fmt_delta(oracle)} |")
    
    # Detailed results per dataset
    for dataset_name, metrics in comparison.items():
        lines.append(f"\n\n## {dataset_name}\n")
        
        # Pass@K comparison
        lines.append("\n### Pass@K Comparison\n")
        lines.append("\n| k | Base | Trained | Δ | Δ% | p-value | Significant |")
        lines.append("\n|---|------|---------|---|----|---------|-------------|")
        
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            metric = f"pass@{k}"
            if metric in metrics:
                m = metrics[metric]
                sig = "✓" if m["significant"] else ""
                lines.append(f"\n| {k} | {m['base_mean']:.4f} | {m['trained_mean']:.4f} | {m['diff']:+.4f} | {m['diff_pct']:+.1f}% | {m['p_value']:.4f} | {sig} |")
        
        # Majority@K comparison
        lines.append("\n\n### Majority@K Comparison\n")
        lines.append("\n| k | Base | Trained | Δ | Δ% | p-value | Significant |")
        lines.append("\n|---|------|---------|---|----|---------|-------------|")
        
        for k in [1, 4, 16, 64, 256]:
            metric = f"maj@{k}"
            if metric in metrics:
                m = metrics[metric]
                sig = "✓" if m["significant"] else ""
                lines.append(f"\n| {k} | {m['base_mean']:.4f} | {m['trained_mean']:.4f} | {m['diff']:+.4f} | {m['diff_pct']:+.1f}% | {m['p_value']:.4f} | {sig} |")
        
        # Best-of-N comparison
        lines.append("\n\n### Best-of-N Comparison\n")
        lines.append("\n| N | Base | Trained | Δ | Δ% | p-value | Significant |")
        lines.append("\n|---|------|---------|---|----|---------|-------------|")
        
        for n in [1, 4, 16, 64, 256]:
            metric = f"bon@{n}"
            if metric in metrics:
                m = metrics[metric]
                sig = "✓" if m["significant"] else ""
                lines.append(f"\n| {n} | {m['base_mean']:.4f} | {m['trained_mean']:.4f} | {m['diff']:+.4f} | {m['diff_pct']:+.1f}% | {m['p_value']:.4f} | {sig} |")
        
        # Oracle
        if "oracle" in metrics:
            m = metrics["oracle"]
            lines.append("\n\n### Oracle\n")
            lines.append(f"\n| Metric | Base | Trained | Δ | p-value |")
            lines.append(f"\n|--------|------|---------|---|---------|")
            sig = "✓" if m["significant"] else ""
            lines.append(f"\n| Oracle | {m['base_mean']:.4f} | {m['trained_mean']:.4f} | {m['diff']:+.4f} | {m['p_value']:.4f} |")
    
    # Legend
    lines.append("\n\n---\n")
    lines.append("\n*Δ = Trained - Base, Δ% = Percentage improvement, * = p < 0.05*\n")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Directory containing base model results")
    parser.add_argument("--trained_dir", type=str, required=True,
                        help="Directory containing trained model results")
    parser.add_argument("--output", type=str, default="comparison_report.md",
                        help="Output path for comparison report")
    
    args = parser.parse_args()
    
    # Load results
    base_results = load_results(Path(args.base_dir))
    trained_results = load_results(Path(args.trained_dir))
    
    # Compare
    comparison = compare_results(base_results, trained_results)
    
    # Generate report
    generate_comparison_report(
        comparison,
        base_name=args.base_dir,
        trained_name=args.trained_dir,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()