#!/usr/bin/env python3
"""
Pass@K Benchmark for Mathematical Reasoning Models.

This script evaluates a model's mathematical reasoning performance using
Pass@K, Majority@K, and Oracle metrics.

Usage:
    python -m bench_passatk.run_bench \
        --model_path ./PAV-distribution-test-1/checkpoint-200 \
        --backend vllm \
        --datasets gsm8k,MATH \
        --k 256 --micro_n 32 \
        --temperature 0.7 --top_p 0.95 --max_new_tokens 2048 \
        --seed 42 \
        --out_dir runs/ckpt200_passat256/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# wandb support (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_passatk.datasets import get_dataset_loader
from bench_passatk.eval import compute_all_metrics, grade_answer
from bench_passatk.samplers import get_sampler
from bench_passatk.utils import (
    get_completed_ids,
    save_config,
    save_results,
    set_global_seed,
    get_problem_seed,
)

# Few-shot examples for step-by-step reasoning
FEW_SHOT_EXAMPLES = [
    {
        "user": "A box has 2 dozen pens. 5 are given away. How many remain?",
        "assistant": "Step 1: One dozen is 12, so 2 dozen is 2 × 12 = 24 pens.\nStep 2: 5 pens are given away.\nStep 3: Remaining pens = 24 − 5 = 19.\nAnswer: 19"
    },
    {
        "user": "Adam, Andrew and Ahmed all raise goats. Adam has 7 goats. Andrew has 5 more than twice as many goats as Adam. Ahmed has 6 fewer goats than Andrew. How many goats does Ahmed have?",
        "assistant": "Step 1: Adam has 7 goats.\nStep 2: Andrew has 5 more than twice Adam's goats, so Andrew has 2 × 7 + 5 = 19 goats.\nStep 3: Ahmed has 6 fewer goats than Andrew, so Ahmed has 19 − 6 = 13 goats.\nAnswer: 13"
    }
]

# System message for step-by-step reasoning
STEP_SYSTEM_MESSAGE = """You solve math problems using natural-language steps only.
Rules:
- Output exactly one reasoning step per line.
- Start every line with "Step k:" (k = 1,2,3,...).
- Each line must contain a SINGLE calculation or deduction. Never put two on one line.
- Do NOT write any code or use Python.
- Do NOT write an introduction or a summary sentence.
- The last line must be "Answer: <number>"."""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pass@K Benchmark for Mathematical Reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf"],
        default="vllm",
        help="Backend to use for generation.",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k",
        help="Comma-separated list of datasets to evaluate. "
             "Options: gsm8k, MATH, AIME2023, AIME2024, AIME2025, AIME2026, OlympiadBench. "
             "For AIME2026+, provide a custom data file via AIME_CUSTOM_PATH env var "
             "or place data/aime_2026.json(l) in the working directory.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use.",
    )
    
    # Sampling arguments
    parser.add_argument(
        "--k",
        type=int,
        default=256,
        help="Total number of samples per problem.",
    )
    parser.add_argument(
        "--micro_n",
        type=int,
        default=32,
        help="Number of samples per micro-batch (for memory efficiency).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate.",
    )
    
    # Reproducibility arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    
    # Output arguments
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/default/",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed problems).",
    )
    
    # System message
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a careful mathematical reasoner. Think step by step and put the final answer in \\boxed{}.",
        help="System message for chat template.",
    )
    
    # Few-shot examples
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Use few-shot examples for step-by-step reasoning format.",
    )
    
    # GPU arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM only).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (vLLM only).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "half", "bf16", "fp16"],
        help="Data type for model weights. Use 'bfloat16' or 'bf16' for bf16.",
    )
    
    # wandb arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="passatk-benchmark",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity (username or team name).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb run name. If not specified, auto-generated from model and timestamp.",
    )
    
    return parser.parse_args()


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        
        return {
            "commit": commit,
            "branch": branch,
        }
    except Exception:
        return {
            "commit": "unknown",
            "branch": "unknown",
        }


def get_gpu_info() -> Dict[str, str]:
    """Get GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "gpu_name": gpu_name,
            "gpu_memory_gb": f"{gpu_memory:.1f}",
        }
    return {
        "gpu_name": "N/A",
        "gpu_memory_gb": "N/A",
    }


def evaluate_problem(
    sampler,
    problem: Dict,
    k: int,
    micro_n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    base_seed: int,
    problem_idx: int,
    system_message: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    dataset_type: str = "math",
) -> Dict:
    """
    Evaluate a single problem.
    
    Args:
        sampler: The model sampler.
        problem: Problem dictionary with 'id', 'problem', 'gold'.
        k: Total number of samples.
        micro_n: Micro-batch size.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_new_tokens: Maximum tokens to generate.
        base_seed: Base random seed.
        problem_idx: Index of the problem.
        system_message: System message for chat template.
        few_shot_examples: Optional few-shot examples.
        dataset_type: Type of dataset for grading.
        
    Returns:
        Result dictionary with samples and metrics.
    """
    # Get problem-specific seed
    problem_seed = get_problem_seed(base_seed, problem_idx)
    
    # Apply chat template
    prompt = sampler.apply_chat_template(
        problem["problem"],
        system_message=system_message,
        few_shot_examples=few_shot_examples,
    )
    
    # Generate samples
    samples_raw = sampler.generate_with_micro_batch(
        prompts=[prompt],
        n=k,
        micro_n=micro_n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=problem_seed,
    )
    
    # Process samples
    samples = []
    for sample in samples_raw[0]:
        text = sample["text"]
        is_correct, pred, _ = grade_answer(text, problem["gold"], dataset_type)
        
        samples.append({
            "text": text,
            "pred": pred,
            "is_correct": is_correct,
        })
    
    # Compute metrics
    metrics = compute_all_metrics(samples)
    
    return {
        "problem_id": problem["id"],
        "problem": problem["problem"],
        "gold": problem["gold"],
        "samples": samples,
        "per_problem": metrics,
    }


def run_benchmark(args):
    """Run the benchmark."""
    # Set global seed
    set_global_seed(args.seed)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get git info and GPU info early for wandb
    git_info = get_git_info()
    gpu_info = get_gpu_info()
    
    # Save configuration
    config = {
        "model_path": args.model_path,
        "backend": args.backend,
        "datasets": args.datasets,
        "k": args.k,
        "micro_n": args.micro_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "system_message": args.system_message,
        "dtype": args.dtype,
        "git": git_info,
        "gpu": gpu_info,
        "timestamp": datetime.now().isoformat(),
    }
    save_config(config, out_dir)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling wandb logging.")
            args.use_wandb = False
        else:
            # Generate run name if not specified
            if args.wandb_run_name is None:
                model_name = Path(args.model_path).name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{model_name}_{timestamp}"
            else:
                run_name = args.wandb_run_name
            
            # Initialize wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=config,
                tags=["passatk", "benchmark"],
            )
            print(f"wandb initialized: {wandb_run.url}")
    
    # Initialize sampler
    print(f"Loading model from {args.model_path}...")
    sampler = get_sampler(
        backend=args.backend,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    # Results storage
    all_results = {}
    
    # Start timer
    start_time = time.time()
    
    try:
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            # Load dataset
            loader = get_dataset_loader(dataset_name)
            problems = loader.load()
            
            if not problems:
                print(f"Warning: No problems loaded for {dataset_name}")
                continue
            
            print(f"Loaded {len(problems)} problems")
            
            # Determine dataset type for grading
            dataset_type = dataset_name.lower()
            if dataset_type.startswith("aime"):
                dataset_type = "aime"
            elif dataset_type.startswith("olympiad"):
                dataset_type = "olympiad"
            
            # Output file for this dataset
            output_file = out_dir / f"{dataset_name}.jsonl"
            
            # Get completed problems if resuming
            completed_ids = set()
            if args.resume and output_file.exists():
                completed_ids = get_completed_ids(output_file)
                print(f"Resuming: {len(completed_ids)} problems already completed")
            
            # Filter out completed problems
            problems_to_eval = [
                p for p in problems if p["id"] not in completed_ids
            ]
            
            if not problems_to_eval:
                print(f"All problems already completed for {dataset_name}")
                continue
            
            # Evaluate problems
            for idx, problem in enumerate(tqdm(
                problems_to_eval,
                desc=f"Evaluating {dataset_name}",
                unit="problem",
            )):
                # Get original index
                orig_idx = problems.index(problem)
                
                # Prepare few-shot examples if requested
                few_shot_examples = None
                if args.few_shot:
                    few_shot_examples = FEW_SHOT_EXAMPLES
                
                # Use step-by-step system message if few-shot is enabled
                system_msg = STEP_SYSTEM_MESSAGE if args.few_shot else args.system_message
                
                try:
                    result = evaluate_problem(
                        sampler=sampler,
                        problem=problem,
                        k=args.k,
                        micro_n=args.micro_n,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        base_seed=args.seed,
                        problem_idx=orig_idx,
                        system_message=system_msg,
                        few_shot_examples=few_shot_examples,
                        dataset_type=dataset_type,
                    )
                    
                    # Save result
                    save_results(
                        output_path=output_file,
                        problem_id=result["problem_id"],
                        problem=result["problem"],
                        gold=result["gold"],
                        samples=result["samples"],
                        per_problem=result["per_problem"],
                    )
                    
                    # Log to wandb (per-problem metrics)
                    if args.use_wandb and wandb_run is not None:
                        wandb.log({
                            f"{dataset_name}/problem_idx": len(completed_ids) + idx + 1,
                            f"{dataset_name}/pass@1": result["per_problem"].get("pass@1", 0),
                            f"{dataset_name}/correct": result["per_problem"].get("correct", 0),
                        })
                    
                except Exception as e:
                    print(f"\nError evaluating problem {problem['id']}: {e}")
                    # Save error result
                    save_results(
                        output_path=output_file,
                        problem_id=problem["id"],
                        problem=problem["problem"],
                        gold=problem["gold"],
                        samples=[],
                        per_problem={"error": str(e)},
                    )
            
            # Load all results for this dataset
            from bench_passatk.utils.io import load_results
            dataset_results = load_results(output_file)
            all_results[dataset_name] = dataset_results
            
            # Log dataset summary to wandb
            if args.use_wandb and wandb_run is not None:
                from bench_passatk.eval.metrics import aggregate_metrics
                agg = aggregate_metrics(dataset_results)
                
                # Log Pass@K metrics
                for k, stats in agg["pass@k"].items():
                    wandb.log({
                        f"{dataset_name}/pass@{k}": stats["mean"],
                        f"{dataset_name}/pass@{k}_ci_low": stats["ci_low"],
                        f"{dataset_name}/pass@{k}_ci_high": stats["ci_high"],
                    })
                
                # Log Best-of-N metrics
                for n, stats in agg["bon@n"].items():
                    wandb.log({
                        f"{dataset_name}/bon@{n}": stats["accuracy"],
                    })
                
                # Log Majority@K metrics
                for k, stats in agg["maj@k"].items():
                    wandb.log({
                        f"{dataset_name}/maj@{k}": stats["accuracy"],
                    })
                
                # Log Oracle
                wandb.log({
                    f"{dataset_name}/oracle": agg["oracle"]["accuracy"],
                })
            
            # Generate per-dataset report
            generate_report(
                {dataset_name: dataset_results},
                out_dir,
                config,
                wall_clock=0,  # Per-dataset time not tracked separately
                report_name=f"{dataset_name}_report.md",
            )
    
    finally:
        # Cleanup
        sampler.cleanup()
        
        # Finish wandb run
        if args.use_wandb and wandb_run is not None:
            wandb.finish()
    
    # End timer
    end_time = time.time()
    wall_clock = end_time - start_time
    
    # Generate report
    generate_report(all_results, out_dir, config, wall_clock)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Results saved to: {out_dir}")
    print(f"Wall-clock time: {wall_clock:.2f}s")
    print(f"{'='*60}")


def generate_report(
    all_results: Dict[str, List],
    out_dir: Path,
    config: Dict,
    wall_clock: float,
    report_name: str = "report.md",
):
    """
    Generate a markdown report with results.
    
    Args:
        all_results: Dictionary mapping dataset names to result lists.
        out_dir: Output directory.
        config: Configuration dictionary.
        wall_clock: Total wall-clock time in seconds.
        report_name: Name of the report file.
    """
    from bench_passatk.eval.metrics import aggregate_metrics, wilson_confidence_interval
    
    report_lines = []
    
    # Header
    report_lines.append("# Pass@K Benchmark Report\n")
    report_lines.append(f"\nGenerated: {datetime.now().isoformat()}\n")
    
    # Metadata
    report_lines.append("\n## Configuration\n")
    report_lines.append(f"\n| Parameter | Value |")
    report_lines.append(f"\n|-----------|-------|")
    report_lines.append(f"\n| Model Path | `{config['model_path']}` |")
    report_lines.append(f"\n| Backend | {config['backend']} |")
    report_lines.append(f"\n| K (samples) | {config['k']} |")
    report_lines.append(f"\n| Temperature | {config['temperature']} |")
    report_lines.append(f"\n| Top-p | {config['top_p']} |")
    report_lines.append(f"\n| Max Tokens | {config['max_new_tokens']} |")
    report_lines.append(f"\n| Seed | {config['seed']} |")
    report_lines.append(f"\n| GPU | {config['gpu']['gpu_name']} ({config['gpu']['gpu_memory_gb']} GB) |")
    report_lines.append(f"\n| Git Commit | `{config['git']['commit'][:8]}` |")
    report_lines.append(f"\n| Wall-clock | {wall_clock:.2f}s |")
    
    # Results per dataset
    for dataset_name, results in all_results.items():
        if not results:
            continue
        
        report_lines.append(f"\n\n## {dataset_name}\n")
        
        # Aggregate metrics
        agg = aggregate_metrics(results)
        
        # Summary statistics
        report_lines.append(f"\n\n**Total Problems:** {agg['n_problems']}\n")
        
        # Pass@1 explicit (first sample accuracy)
        report_lines.append("\n### Pass@1 (First Sample Accuracy)\n")
        pass1 = agg["pass@1_explicit"]
        report_lines.append(f"\n| Metric | Value | 95% CI |")
        report_lines.append(f"\n|--------|-------|--------|")
        report_lines.append(f"\n| Pass@1 | {pass1['accuracy']:.4f} | [{pass1['ci_low']:.4f}, {pass1['ci_high']:.4f}] |")
        
        # Pass@k table
        report_lines.append("\n\n### Pass@K (Unbiased Estimator)\n")
        report_lines.append("\n| k | Pass@k | 95% CI |")
        report_lines.append("\n|---|--------|--------|")
        
        for k, stats in agg["pass@k"].items():
            mean = stats["mean"]
            ci_low = stats["ci_low"]
            ci_high = stats["ci_high"]
            report_lines.append(f"\n| {k} | {mean:.4f} | [{ci_low:.4f}, {ci_high:.4f}] |")
        
        # Best-of-N table
        report_lines.append("\n\n### Best-of-N (BoN)\n")
        report_lines.append("\n| N | Accuracy | 95% CI |")
        report_lines.append("\n|---|----------|--------|")
        
        for n, stats in agg["bon@n"].items():
            acc = stats["accuracy"]
            ci_low = stats["ci_low"]
            ci_high = stats["ci_high"]
            report_lines.append(f"\n| {n} | {acc:.4f} | [{ci_low:.4f}, {ci_high:.4f}] |")
        
        # Majority@k table
        report_lines.append("\n\n### Majority@K (Self-Consistency)\n")
        report_lines.append("\n| k | Accuracy | 95% CI |")
        report_lines.append("\n|---|----------|--------|")
        
        for k, stats in agg["maj@k"].items():
            acc = stats["accuracy"]
            ci_low = stats["ci_low"]
            ci_high = stats["ci_high"]
            report_lines.append(f"\n| {k} | {acc:.4f} | [{ci_low:.4f}, {ci_high:.4f}] |")
        
        # Oracle
        report_lines.append("\n\n### Oracle\n")
        oracle = agg["oracle"]
        report_lines.append(f"\n| Metric | Value | 95% CI |")
        report_lines.append(f"\n|--------|-------|--------|")
        report_lines.append(f"\n| Oracle@{config['k']} | {oracle['accuracy']:.4f} | [{oracle['ci_low']:.4f}, {oracle['ci_high']:.4f}] |")
        
        # Level breakdown for MATH
        if dataset_name == "MATH":
            # Group by level
            levels = {}
            for r in results:
                level = r.get("level", "unknown")
                if level not in levels:
                    levels[level] = []
                levels[level].append(r)
            
            if len(levels) > 1:
                report_lines.append("\n\n### Results by Level\n")
                report_lines.append("\n| Level | n | Pass@1 | Pass@16 | Pass@256 |")
                report_lines.append("\n|-------|---|--------|---------|----------|")
                
                for level in sorted(levels.keys()):
                    level_results = levels[level]
                    level_agg = aggregate_metrics(level_results)
                    n = level_agg["n_problems"]
                    p1 = level_agg["pass@k"].get(1, {}).get("mean", float("nan"))
                    p16 = level_agg["pass@k"].get(16, {}).get("mean", float("nan"))
                    p256 = level_agg["pass@k"].get(256, {}).get("mean", float("nan"))
                    report_lines.append(f"\n| {level} | {n} | {p1:.4f} | {p16:.4f} | {p256:.4f} |")
    
    # Write report
    report_path = out_dir / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)


def main():
    """Entry point for the module."""
    args = parse_args()
    run_benchmark(args)