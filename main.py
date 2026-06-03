"""
AIME 2023/2024/2025 evaluation with Qwen2.5-1.5B-Instruct + Wandb dashboard.

Usage:
    python main.py                                # defaults: all three years
    python main.py --years 2024 2025             # specific years
    python main.py --batch-size 8 --dtype float16
    python main.py --wandb-entity my-team
"""

import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from config import parse_args
from dataset import AIMEProblem, load_aime_problems
from evaluate import (
    ProblemResult,
    extract_answer,
    init_wandb,
    log_to_wandb,
    save_results,
    score_results,
)
from model import build_chat_prompt, generate_batch, load_model_and_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU (will be slow).")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}: {props.name}  |  "
            f"VRAM: {props.total_memory / 1e9:.1f} GB  |  "
            f"sm_{props.major}{props.minor}"
        )


def run_eval(config):
    set_seed(config.seed)

    print("=" * 60)
    print("AIME Evaluation  |  Model:", config.model_name)
    print("=" * 60)
    check_gpu()

    # 1. Load datasets
    year_data = load_aime_problems(config.years)
    all_problems: list[AIMEProblem] = [p for y in config.years for p in year_data[y]]

    if not all_problems:
        print("No problems loaded. Exiting.")
        sys.exit(1)

    print(f"\nTotal problems to evaluate: {len(all_problems)}")

    # 2. Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # 3. Build prompts
    prompts = [build_chat_prompt(p.problem, tokenizer) for p in all_problems]

    # 4. Run inference with progress bar
    print(f"\nRunning inference (batch_size={config.batch_size}) ...")
    raw_outputs: list[str] = []
    t0 = time.time()

    with tqdm(total=len(all_problems), unit="problem") as pbar:
        for start in range(0, len(prompts), config.batch_size):
            batch_prompts   = prompts[start : start + config.batch_size]
            batch_outputs   = generate_batch(model, tokenizer, batch_prompts, config)
            raw_outputs.extend(batch_outputs)
            pbar.update(len(batch_prompts))

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s  ({elapsed / len(all_problems):.1f}s/problem)")

    # 5. Score
    results: list[ProblemResult] = []
    for problem, raw in zip(all_problems, raw_outputs):
        predicted = extract_answer(raw)
        correct   = predicted is not None and predicted == problem.answer
        results.append(ProblemResult(
            problem=problem,
            raw_output=raw,
            predicted=predicted,
            correct=correct,
        ))

    metrics = score_results(results)

    # 6. Print console summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"  {key:<40} {val:.1%}")
        else:
            print(f"  {key:<40} {val}")

    # 7. Log to Wandb
    run = init_wandb(config)
    log_to_wandb(results, metrics, config)
    run.finish()

    # 8. Save JSON
    save_results(results, metrics, config)

    print("\nDone.")


if __name__ == "__main__":
    config = parse_args()
    run_eval(config)
