"""
AIME 2023/2024/2025 — combined pass@k evaluation (single wandb run).

Metrics reported:
  pass@256        : unbiased estimator, averaged over all combined problems
  greedy_pass@1   : greedy decode accuracy, averaged over all combined problems
  per-year breakdown of both metrics in the same run

Wandb run created:
  <base>-combined

Usage:
    python main.py                              # all three years combined
    python main.py --num-samples 32            # quick smoke-test
    python main.py --years 2024 2025           # subset of years
    python main.py --wandb-entity my-team
"""

import os
import random
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch
from tqdm import tqdm

from config import EvalConfig, parse_args
from dataset import AIMEProblem, load_aime_problems
from evaluate import (
    ProblemResult,
    extract_answer,
    init_wandb,
    log_combined_to_wandb,
    save_combined_results,
    score_combined,
)
from model import build_chat_prompt, generate_greedy, generate_samples, load_model_and_tokenizer, warmup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU (will be very slow).")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}: {props.name}  |  "
            f"VRAM: {props.total_memory / 1e9:.1f} GB  |  "
            f"sm_{props.major}{props.minor}"
        )


def eval_all(
    all_problems: list[AIMEProblem],
    model,
    tokenizer,
    config: EvalConfig,
) -> list[ProblemResult]:
    """Two-phase evaluation over all problems at once.

    Phase 1 — greedy pass@1 (temperature=0, single decode per problem)
    Phase 2 — sampling pass@k (config.num_samples draws per problem)
    """
    n = len(all_problems)
    print(f"\n[Phase 1/2] greedy pass@1  ({n} problems total)")

    greedy_outputs: dict[str, str] = {}
    for problem in tqdm(all_problems, desc="greedy", unit="problem"):
        prompt = build_chat_prompt(problem.problem, tokenizer)
        greedy_outputs[problem.problem_id] = generate_greedy(model, prompt, config)

    print(f"\n[Phase 2/2] sampling pass@{config.num_samples}  ({n} problems total)")

    results: list[ProblemResult] = []
    for problem in tqdm(all_problems, desc="sample", unit="problem"):
        prompt      = build_chat_prompt(problem.problem, tokenizer)
        raw_outputs = generate_samples(model, tokenizer, prompt, config)
        predicted   = [extract_answer(o) for o in raw_outputs]
        n_correct   = sum(1 for p in predicted if p == problem.answer)

        greedy_out  = greedy_outputs[problem.problem_id]
        greedy_pred = extract_answer(greedy_out)

        results.append(ProblemResult(
            problem=problem,
            raw_outputs=raw_outputs,
            predicted_answers=predicted,
            n_correct=n_correct,
            n_samples=config.num_samples,
            greedy_output=greedy_out,
            greedy_predicted=greedy_pred,
            greedy_correct=(greedy_pred == problem.answer),
        ))

    return results


def print_combined_summary(metrics: dict, elapsed: float):
    print(f"\n{'='*60}")
    print(f"Combined AIME Results  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<44} {v:.1%}")
        else:
            print(f"  {k:<44} {v}")


def run_eval(config: EvalConfig):
    set_seed(config.seed)

    print("=" * 60)
    print(f"AIME Combined Evaluation  |  Model: {config.model_name}")
    print(f"  years={config.years}  num_samples={config.num_samples}")
    print("=" * 60)
    check_gpu()

    year_data    = load_aime_problems(config.years)
    all_problems = [p for year in config.years for p in year_data.get(year, [])]

    if not all_problems:
        print("No problems loaded. Exiting.")
        sys.exit(1)

    print(f"\nTotal problems: {len(all_problems)}  "
          f"({', '.join(f'AIME {y}: {len(year_data.get(y, []))}' for y in config.years)})")

    model, tokenizer = load_model_and_tokenizer(config)

    print("\nWarm-up pass ...")
    t_warmup = time.time()
    warmup(model)
    print(f"Warm-up done in {time.time() - t_warmup:.1f}s — starting evaluation.\n")

    t0      = time.time()
    results = eval_all(all_problems, model, tokenizer, config)
    elapsed = time.time() - t0

    metrics = score_combined(results, config)
    print_combined_summary(metrics, elapsed)

    run = init_wandb(config, suffix="combined")
    log_combined_to_wandb(results, metrics, config)
    run.finish()

    save_combined_results(results, metrics, config)

    print(f"\nTotal wall-clock time: {elapsed / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    config = parse_args()
    run_eval(config)
