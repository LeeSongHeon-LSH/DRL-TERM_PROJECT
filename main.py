"""
AIME 2023/2024/2025 — pass@k evaluation with per-year wandb runs + comparison dashboard.

Wandb runs created:
  <base>-2023        : AIME 2023 results (pass@k metrics, per-problem table, difficulty curve)
  <base>-2024        : AIME 2024 results
  <base>-2025        : AIME 2025 results
  <base>-comparison  : cross-year comparison table and bar charts

Usage:
    python main.py                              # pass@256, all three years
    python main.py --num-samples 32            # quick smoke-test
    python main.py --years 2024 2025           # specific years only
    python main.py --no-sample                 # greedy pass@1
    python main.py --wandb-entity my-team
"""

import os
import random
import sys
import time

# FlashInfer requires sm75+ (Turing); disable it for older GPUs.
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
    log_year_to_wandb,
    log_comparison_to_wandb,
    save_results,
    score_results,
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


def eval_year(
    year: int,
    problems: list[AIMEProblem],
    model,
    tokenizer,
    config: EvalConfig,
) -> list[ProblemResult]:
    """Two-phase evaluation per year.

    Phase 1 — greedy pass@1:
        Each problem is decoded once with temperature=0.  Gives exact
        single-attempt accuracy (no estimator bias).

    Phase 2 — sampling pass@k:
        Each problem is sampled config.num_samples times.  Used to
        compute pass@1/8/64/256 via the unbiased estimator.
    """
    n = len(problems)
    print(f"\n[AIME {year}] Phase 1/2 — greedy pass@1  ({n} problems)")

    greedy_outputs: dict[str, str] = {}
    for problem in tqdm(problems, desc=f"AIME {year} [greedy]", unit="problem"):
        prompt = build_chat_prompt(problem.problem, tokenizer)
        greedy_outputs[problem.problem_id] = generate_greedy(model, prompt, config)

    print(f"\n[AIME {year}] Phase 2/2 — sampling pass@{config.num_samples}  ({n} problems)")

    results: list[ProblemResult] = []
    for problem in tqdm(problems, desc=f"AIME {year} [sample]", unit="problem"):
        prompt = build_chat_prompt(problem.problem, tokenizer)
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


def print_year_summary(year: int, metrics: dict, elapsed: float):
    print(f"\n{'='*60}")
    print(f"AIME {year} — Results  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    # greedy first, then sampled, then totals
    priority = ["greedy_pass@1", "sampled_pass@1", "pass@"]
    def _sort_key(item):
        k = item[0]
        for i, p in enumerate(priority):
            if p in k:
                return (i, k)
        return (len(priority), k)
    for k, v in sorted(metrics.items(), key=_sort_key):
        if isinstance(v, float):
            print(f"  {k:<44} {v:.1%}")
        else:
            print(f"  {k:<44} {v}")


def run_eval(config: EvalConfig):
    set_seed(config.seed)

    print("=" * 60)
    print(f"AIME pass@k Evaluation  |  Model: {config.model_name}")
    print(f"  num_samples={config.num_samples}  "
          f"sample_batch_size={config.sample_batch_size}  "
          f"pass_k={config.pass_k_values}")
    print("=" * 60)
    check_gpu()

    year_data = load_aime_problems(config.years)

    if not any(year_data.values()):
        print("No problems loaded. Exiting.")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(config)

    # Warm-up: trigger engine/kernel initialisation before the main loop so the
    # first real problem isn't penalised by one-time compilation costs.
    print("\nWarm-up pass (first-run CUDA kernel compilation may take a while)...")
    t_warmup = time.time()
    warmup(model)
    print(f"Warm-up done in {time.time() - t_warmup:.1f}s — starting evaluation.\n")

    year_metrics: dict[int, dict] = {}
    t_total = time.time()

    # -----------------------------------------------------------------------
    # Per-year loop: infer → score → wandb run → save JSON
    # -----------------------------------------------------------------------
    for year in config.years:
        problems = year_data.get(year, [])
        if not problems:
            print(f"\nNo problems for AIME {year}, skipping.")
            continue

        t0 = time.time()
        results = eval_year(year, problems, model, tokenizer, config)
        elapsed = time.time() - t0

        metrics = score_results(results, config)
        year_metrics[year] = metrics

        print_year_summary(year, metrics, elapsed)

        run = init_wandb(config, suffix=str(year))
        log_year_to_wandb(results, metrics, config, year)
        run.finish()

        save_results(results, metrics, config, year)

    # -----------------------------------------------------------------------
    # Cross-year comparison dashboard
    # -----------------------------------------------------------------------
    if len(year_metrics) > 1:
        print(f"\n{'='*60}")
        print("Logging cross-year comparison dashboard ...")
        run = init_wandb(config, suffix="comparison")
        log_comparison_to_wandb(year_metrics, config)
        run.finish()

    print(f"\nTotal wall-clock time: {(time.time() - t_total) / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    config = parse_args()
    run_eval(config)
