"""
Answer extraction, pass@k scoring, and Wandb logging.

Wandb structure produced per run:
  - <base>-<year>      : per-problem table + difficulty curve chart for that year
  - <base>-comparison  : cross-year comparison table and bar charts
"""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import wandb

from config import EvalConfig
from dataset import AIMEProblem


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE     = re.compile(r"\\boxed\{(\d+)\}")
_ANSWER_IS_RE = re.compile(r"(?:the\s+answer\s+is|answer\s*[:=])\s*(\d+)", re.IGNORECASE)
_TRAILING_INT = re.compile(r"\b(\d{1,3})\b")


def extract_answer(text: str) -> Optional[int]:
    """Parse a model completion and return the predicted integer answer (0-999)."""
    matches = _BOXED_RE.findall(text)
    if matches:
        val = int(matches[-1])
        if 0 <= val <= 999:
            return val

    m = _ANSWER_IS_RE.search(text[::-1])
    if m:
        val = int(m.group(1)[::-1])
        if 0 <= val <= 999:
            return val

    all_ints = [int(x) for x in _TRAILING_INT.findall(text) if 0 <= int(x) <= 999]
    if all_ints:
        return all_ints[-1]

    return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    problem: AIMEProblem
    raw_outputs: List[str]                  # N sampled completions
    predicted_answers: List[Optional[int]]  # N extracted predictions
    n_correct: int                          # how many of the N are correct
    n_samples: int                          # N (= config.num_samples)
    # greedy pass@1 fields (populated in two-phase eval)
    greedy_output: Optional[str] = None
    greedy_predicted: Optional[int] = None
    greedy_correct: bool = False

    @property
    def year(self) -> int:           return self.problem.year
    @property
    def competition(self) -> str:    return self.problem.competition
    @property
    def problem_number(self) -> int: return self.problem.problem_number

    def pass_at(self, k: int) -> float:
        return _pass_at_k(self.n_samples, self.n_correct, k)


# ---------------------------------------------------------------------------
# pass@k unbiased estimator
# ---------------------------------------------------------------------------

def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k: 1 - C(n-c, k) / C(n, k).
    Computed via log-products to avoid integer overflow for large n.
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # log( product_{i=0}^{k-1} (n-c-i)/(n-i) )
    log_prob_all_wrong = sum(
        math.log(n - c - i) - math.log(n - i) for i in range(k)
    )
    return 1.0 - math.exp(log_prob_all_wrong)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results: List[ProblemResult], config: EvalConfig) -> Dict:
    """Compute pass@k metrics, grouped by AIME competition split (I / II)."""
    valid_k = [k for k in config.pass_k_values if k <= config.num_samples]
    metrics: Dict = {}

    has_greedy = any(r.greedy_output is not None for r in results)

    for comp in ["I", "II"]:
        comp_results = [r for r in results if r.competition == comp]
        if not comp_results:
            continue
        for k in valid_k:
            metrics[f"pass@{k}/AIME_{comp}"] = (
                sum(r.pass_at(k) for r in comp_results) / len(comp_results)
            )
        if has_greedy:
            metrics[f"greedy_pass@1/AIME_{comp}"] = (
                sum(r.greedy_correct for r in comp_results) / len(comp_results)
            )

    for k in valid_k:
        metrics[f"pass@{k}/overall"] = (
            sum(r.pass_at(k) for r in results) / len(results) if results else 0.0
        )
    if has_greedy:
        metrics["greedy_pass@1/overall"] = (
            sum(r.greedy_correct for r in results) / len(results) if results else 0.0
        )

    metrics["total_problems"]  = len(results)
    metrics["total_samples"]   = sum(r.n_samples for r in results)
    metrics["total_n_correct"] = sum(r.n_correct for r in results)
    return metrics


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------

def _run_name(config: EvalConfig, suffix: str) -> str:
    base = config.wandb_run_name or config.model_name.split("/")[-1]
    return f"{base}-{suffix}"


def init_wandb(config: EvalConfig, suffix: str):
    """Initialise a wandb run for a given year or for the comparison dashboard."""
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=_run_name(config, suffix),
        config={
            "model":             config.model_name,
            "dtype":             config.dtype,
            "max_new_tokens":    config.max_new_tokens,
            "temperature":       config.temperature,
            "do_sample":         config.do_sample,
            "num_samples":       config.num_samples,
            "sample_batch_size": config.sample_batch_size,
            "pass_k_values":     config.pass_k_values,
            "years":             config.years,
        },
    )


def log_year_to_wandb(
    results: List[ProblemResult],
    metrics: Dict,
    config: EvalConfig,
    year: int,
):
    """
    Log one year's results to the currently active wandb run.

    Uploads:
      - Scalar pass@k metrics
      - Per-problem results table
      - pass@k by problem-number bar chart (difficulty curve)
    """
    wandb.log(metrics)

    valid_k   = [k for k in config.pass_k_values if k <= config.num_samples]
    has_greedy = any(r.greedy_output is not None for r in results)
    pk_col     = f"pass@{config.num_samples}"

    # Per-problem table — one column per valid k value
    columns = [
        "problem_id", "competition", "problem_number",
        "ground_truth", "n_correct", "n_samples",
    ]
    if has_greedy:
        columns += ["greedy_pass@1", "greedy_predicted"]
    columns += [f"pass@{k}" for k in valid_k]
    columns += ["best_predicted", "problem_snippet"]

    table = wandb.Table(columns=columns)
    for r in results:
        best_pred = next(
            (p for p in r.predicted_answers if p == r.problem.answer),
            r.predicted_answers[0] if r.predicted_answers else None,
        )
        row = [
            r.problem.problem_id,
            r.competition,
            r.problem_number,
            r.problem.answer,
            r.n_correct,
            r.n_samples,
        ]
        if has_greedy:
            row += [
                float(r.greedy_correct),
                r.greedy_predicted if r.greedy_predicted is not None else -1,
            ]
        row += [r.pass_at(k) for k in valid_k]
        row += [
            best_pred if best_pred is not None else -1,
            r.problem.problem[:300] + ("..." if len(r.problem.problem) > 300 else ""),
        ]
        table.add_data(*row)
    wandb.log({f"results/{year}/per_problem": table})

    # Difficulty curve: all pass@k values by problem number
    by_num: Dict[int, List[ProblemResult]] = {}
    for r in results:
        by_num.setdefault(r.problem_number, []).append(r)

    curve_cols = ["problem_number"]
    if has_greedy:
        curve_cols.append("greedy_pass@1")
    curve_cols += [f"pass@{k}" for k in valid_k]

    num_table = wandb.Table(columns=curve_cols)
    for num in sorted(by_num):
        rs  = by_num[num]
        row = [num]
        if has_greedy:
            row.append(sum(r.greedy_correct for r in rs) / len(rs))
        row += [sum(r.pass_at(k) for r in rs) / len(rs) for k in valid_k]
        num_table.add_data(*row)

    wandb.log({
        f"charts/{year}/pass_by_problem_number_table": num_table,
        f"charts/{year}/pass_by_problem_number": wandb.plot.bar(
            num_table, "problem_number", pk_col,
            title=f"AIME {year} — {pk_col} by Problem Number (1=easiest, 15=hardest)",
        ),
    })


def log_comparison_to_wandb(
    year_metrics: Dict[int, Dict],
    config: EvalConfig,
):
    """
    Log a cross-year comparison to the currently active wandb run.

    Uploads:
      - Flat scalar metrics per (year, k)
      - Summary table (rows = years, cols = pass@k values)
      - Bar charts comparing each pass@k value across years
    """
    valid_k = [k for k in config.pass_k_values if k <= config.num_samples]

    # Flat scalars
    flat: Dict = {}
    for year, m in year_metrics.items():
        for k in valid_k:
            flat[f"comparison/{year}/pass@{k}"] = m.get(f"pass@{k}/overall", 0.0)
    wandb.log(flat)

    # Summary table
    cols = ["year"] + [f"pass@{k}" for k in valid_k] + ["total_problems"]
    table = wandb.Table(columns=cols)
    for year, m in sorted(year_metrics.items()):
        row = [year] + [m.get(f"pass@{k}/overall", 0.0) for k in valid_k] + [m["total_problems"]]
        table.add_data(*row)
    wandb.log({"comparison/year_summary": table})

    # Bar chart per k
    for k in valid_k:
        kt = wandb.Table(columns=["year", f"pass@{k}"])
        for year, m in sorted(year_metrics.items()):
            kt.add_data(str(year), m.get(f"pass@{k}/overall", 0.0))
        wandb.log({
            f"comparison/pass@{k}_by_year": wandb.plot.bar(
                kt, "year", f"pass@{k}",
                title=f"pass@{k} — AIME 2023 vs 2024 vs 2025",
            )
        })


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def save_results(
    results: List[ProblemResult],
    metrics: Dict,
    config: EvalConfig,
    year: int,
):
    os.makedirs(config.output_dir, exist_ok=True)
    valid_k = [k for k in config.pass_k_values if k <= config.num_samples]
    output = {
        "year":    year,
        "metrics": metrics,
        "results": [
            {
                "problem_id":        r.problem.problem_id,
                "competition":       r.competition,
                "problem_number":    r.problem_number,
                "ground_truth":      r.problem.answer,
                "n_correct":         r.n_correct,
                "n_samples":         r.n_samples,
                "greedy_correct":    r.greedy_correct,
                "greedy_predicted":  r.greedy_predicted,
                "greedy_pass@1":     float(r.greedy_correct),
                **{f"pass@{k}": r.pass_at(k) for k in valid_k},
                "predicted_answers": [p if p is not None else -1 for p in r.predicted_answers],
                "problem":           r.problem.problem,
                "raw_outputs":       r.raw_outputs,
            }
            for r in results
        ],
    }
    path = os.path.join(config.output_dir, f"aime_{year}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Combined (multi-year) scoring, wandb logging, and JSON save
# ---------------------------------------------------------------------------

def score_combined(results: List[ProblemResult], config: EvalConfig) -> Dict:
    """Compute pass@256 and greedy_pass@1 across all combined problems.

    Also includes per-year breakdowns of both metrics.
    """
    n = len(results)
    if n == 0:
        return {}

    num_samples = config.num_samples
    metrics: Dict = {
        f"pass@{num_samples}/combined": sum(r.pass_at(num_samples) for r in results) / n,
        "greedy_pass@1/combined":       sum(r.greedy_correct for r in results) / n,
        "total_problems":               n,
        "total_samples":                sum(r.n_samples for r in results),
        "total_n_correct":              sum(r.n_correct for r in results),
    }

    for year in sorted(set(r.year for r in results)):
        yr = [r for r in results if r.year == year]
        metrics[f"pass@{num_samples}/{year}"] = (
            sum(r.pass_at(num_samples) for r in yr) / len(yr)
        )
        metrics[f"greedy_pass@1/{year}"] = (
            sum(r.greedy_correct for r in yr) / len(yr)
        )

    return metrics


def log_combined_to_wandb(
    results: List[ProblemResult],
    metrics: Dict,
    config: EvalConfig,
):
    """Log combined results to the currently active wandb run.

    Uploads:
      - Scalar metrics (combined + per-year)
      - Per-problem table (all years, year column included)
      - Bar charts: pass@256 and greedy_pass@1 by year
    """
    wandb.log(metrics)

    num_samples = config.num_samples
    pk_col      = f"pass@{num_samples}"

    # Per-problem table
    columns = [
        "year", "problem_id", "competition", "problem_number",
        "ground_truth", "n_correct", "n_samples",
        "greedy_pass@1", "greedy_predicted", pk_col, "problem_snippet",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        table.add_data(
            r.year,
            r.problem.problem_id,
            r.competition,
            r.problem_number,
            r.problem.answer,
            r.n_correct,
            r.n_samples,
            float(r.greedy_correct),
            r.greedy_predicted if r.greedy_predicted is not None else -1,
            r.pass_at(num_samples),
            r.problem.problem[:300] + ("..." if len(r.problem.problem) > 300 else ""),
        )
    wandb.log({"combined/per_problem": table})

    # Bar charts: both metrics by year
    years = sorted(set(r.year for r in results))
    year_table = wandb.Table(columns=["year", pk_col, "greedy_pass@1"])
    for year in years:
        year_table.add_data(
            str(year),
            metrics.get(f"{pk_col}/{year}", 0.0),
            metrics.get(f"greedy_pass@1/{year}", 0.0),
        )
    wandb.log({"combined/by_year_summary": year_table})
    wandb.log({
        f"charts/{pk_col}_by_year": wandb.plot.bar(
            year_table, "year", pk_col,
            title=f"{pk_col} — AIME {'/'.join(str(y) for y in years)}",
        ),
        "charts/greedy_pass1_by_year": wandb.plot.bar(
            year_table, "year", "greedy_pass@1",
            title=f"greedy pass@1 — AIME {'/'.join(str(y) for y in years)}",
        ),
    })


def save_combined_results(
    results: List[ProblemResult],
    metrics: Dict,
    config: EvalConfig,
):
    os.makedirs(config.output_dir, exist_ok=True)
    num_samples = config.num_samples
    output = {
        "years":   sorted(set(r.year for r in results)),
        "metrics": metrics,
        "results": [
            {
                "year":              r.year,
                "problem_id":        r.problem.problem_id,
                "competition":       r.competition,
                "problem_number":    r.problem_number,
                "ground_truth":      r.problem.answer,
                "n_correct":         r.n_correct,
                "n_samples":         r.n_samples,
                "greedy_correct":    r.greedy_correct,
                "greedy_predicted":  r.greedy_predicted,
                "greedy_pass@1":     float(r.greedy_correct),
                f"pass@{num_samples}": r.pass_at(num_samples),
                "predicted_answers": [p if p is not None else -1 for p in r.predicted_answers],
                "problem":           r.problem.problem,
                "raw_outputs":       r.raw_outputs,
            }
            for r in results
        ],
    }
    years_str = "_".join(str(y) for y in output["years"])
    path = os.path.join(config.output_dir, f"aime_{years_str}_combined_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {path}")
