#!/usr/bin/env python3
"""
Generate Pass@K comparison plots for the 3 models on AIME 2023/2024/2025.

Reads pass@k values from each model's report.md and creates a side-by-side
figure with one subplot per AIME year, comparing the three models.

Usage:
    python -m bench_passatk.plot_passk \
        --out_path runs/passk_comparison.png
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# Data extracted from the model report.md files
# (model_name, model_dir, k -> pass@k values per dataset)
MODELS = [
    {
        "name": "PAV-distribution",
        "dir": "runs/pav-checkpoint-500",
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "name": "PAV-scalar-c2",
        "dir": "runs/pav-scalar-c2-checkpoint-500",
        "color": "#ff7f0e",
        "marker": "s",
    },
    {
        "name": "Qwen2.5-1.5B-Instruct",
        "dir": "runs/qwen-math-1.5b-baseline",
        "color": "#2ca02c",
        "marker": "^",
    },
    {
        "name": "Qwen2.5-1.5B-Instruct (from image)",
        "dir": None,  # Hardcoded from the provided image
        "color": "#d62728",
        "marker": "D",
        "hardcoded": {
            "AIME2023": {1: 0.054, 8: 0.083, 64: 0.167, 256: 0.267},
            "AIME2024": {1: 0.028, 8: 0.121, 64: 0.306, 256: 0.467},
            "AIME2025": {1: 0.010, 8: 0.065, 64: 0.231, 256: 0.367},
        },
    },
]

DATASETS = ["AIME2023", "AIME2024", "AIME2025"]


def parse_passk_from_report(report_path: Path) -> Dict[int, float]:
    """
    Parse Pass@K values from a report.md file.

    Looks for the AIME2023/2024/2025 section's Pass@K table.
    """
    if not report_path.exists():
        return {}

    content = report_path.read_text(encoding="utf-8")
    passk = {}

    # Find the Pass@K section
    pattern = re.compile(
        r"### Pass@K \(Unbiased Estimator\)\s*\n\s*\n\s*\| k \| Pass@k \| 95% CI \|\s*\n"
        r"\|\-+\|\-+\|\-+\|\s*\n"
        r"((?:\| \d+ \| [\d.]+ \| \[.*?\] \|\s*\n)+)",
        re.MULTILINE,
    )

    match = pattern.search(content)
    if not match:
        return passk

    table = match.group(1)
    row_pattern = re.compile(r"\| (\d+) \| ([\d.]+) \|")
    for row_match in row_pattern.finditer(table):
        k = int(row_match.group(1))
        value = float(row_match.group(2))
        passk[k] = value

    return passk


def parse_passk_per_dataset(model_dir: Path) -> Dict[str, Dict[int, float]]:
    """
    Parse Pass@K values for each AIME dataset from individual report files.

    For each dataset, looks for the AIME YYYY section and extracts Pass@K.
    Falls back to the combined report.md if the per-dataset file lacks the data.
    """
    result = {}

    for dataset in DATASETS:
        # Try the per-dataset report first
        per_dataset_report = model_dir / f"{dataset}_report.md"
        combined_report = model_dir / "report.md"

        passk = {}

        # Per-dataset report has only one AIME section, so this gives us that dataset
        if per_dataset_report.exists():
            content = per_dataset_report.read_text(encoding="utf-8")
            # Check if it contains the AIME section we want
            if f"## {dataset}" in content:
                passk = parse_passk_from_report(per_dataset_report)

        # If not found, try the combined report by extracting the right AIME section
        if not passk and combined_report.exists():
            content = combined_report.read_text(encoding="utf-8")
            # Find the section for this dataset
            section_pattern = re.compile(
                rf"## {re.escape(dataset)}\s*\n(.*?)(?=\n## |\Z)",
                re.DOTALL,
            )
            section_match = section_pattern.search(content)
            if section_match:
                section = section_match.group(1)
                # Parse the Pass@K table from this section
                table_pattern = re.compile(
                    r"### Pass@K \(Unbiased Estimator\).*?\n((?:\| \d+ \| [\d.]+ \|.*?\n)+)",
                    re.DOTALL,
                )
                table_match = table_pattern.search(section)
                if table_match:
                    table = table_match.group(1)
                    row_pattern = re.compile(r"\| (\d+) \| ([\d.]+) \|")
                    for row_match in row_pattern.finditer(table):
                        k = int(row_match.group(1))
                        value = float(row_match.group(2))
                        passk[k] = value

        result[dataset] = passk

    return result


def plot_passk_comparison(
    model_data: Dict[str, Dict[str, Dict[int, float]]],
    out_path: Path,
    title_suffix: str = "",
):
    """
    Plot Pass@K comparison for the 3 models on AIME 2023/2024/2025.

    Args:
        model_data: {model_name: {dataset: {k: pass@k}}}
        out_path: Output file path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        ax.set_title(dataset.replace("AIME", "AIME "), fontsize=16, fontweight="bold")

        for model in MODELS:
            name = model["name"]
            passk = model_data.get(name, {}).get(dataset, {})
            if not passk:
                continue

            # Sort by k
            ks = sorted(passk.keys())
            values = [passk[k] for k in ks]

            ax.plot(
                ks,
                values,
                color=model["color"],
                marker=model["marker"],
                markersize=8,
                linewidth=2,
                label=name,
                markerfacecolor="white",
                markeredgewidth=2,
                markeredgecolor=model["color"],
            )

            # Annotate only at k = 1, 8, 64, 256 (original style)
            for k, v in zip(ks, values):
                if k in (1, 8, 64, 256):
                    ax.annotate(
                        f"{v:.3f}",
                        xy=(k, v),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=10,
                        color=model["color"],
                        fontweight="bold",
                    )

        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 64, 256])
        ax.set_xticklabels(["1", "8", "64", "256"], fontsize=12)
        ax.set_ylim(-0.02, 0.6)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 0:
            ax.set_ylabel("pass@k", fontsize=12)
        ax.set_xlabel("Number of Samples k", fontsize=12)

        # Only show k values 1, 8, 64, 256 to match original style
        ks_to_show = [k for k in ks if k in (1, 8, 64, 256)]
        # Filter for clean plotting (full curve still drawn, but only key points annotated)

    # Combined legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        fontsize=11,
        frameon=True,
    )

    # Figure caption
    caption = "Figure: pass@k Results of PAV-distribution / PAV-scalar-c2 / Qwen2.5-1.5B-Instruct on AIME 2023 / 2024 / 2025"
    if title_suffix:
        caption += f"  ({title_suffix})"
    fig.text(0.5, -0.02, caption, ha="center", fontsize=11, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Pass@K comparison plots.")
    parser.add_argument(
        "--out_path",
        type=str,
        default="runs/passk_comparison.png",
        help="Output path for the plot image.",
    )
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data for each model
    model_data = {}
    for model in MODELS:
        if model.get("hardcoded") is not None:
            # Use hardcoded data (e.g., from an image)
            model_data[model["name"]] = model["hardcoded"]
            print(f"Loaded {model['name']} (hardcoded from image): {model['hardcoded']}")
            continue

        model_dir = Path(model["dir"])
        if not model_dir.exists():
            print(f"Warning: Model directory not found: {model_dir}")
            continue
        model_data[model["name"]] = parse_passk_per_dataset(model_dir)
        print(f"Loaded {model['name']}: {model_data[model['name']]}")

    # Generate the plot
    plot_passk_comparison(model_data, out_path)


if __name__ == "__main__":
    main()