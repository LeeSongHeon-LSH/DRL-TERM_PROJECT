"""
Answer extraction, scoring, and Wandb dashboard logging.

Answer extraction priority:
  1. \\boxed{N}   — LaTeX box (our requested format)
  2. "The answer is N" / "answer: N" — natural language
  3. Last integer in [0, 999] found in the output — last-resort fallback
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import wandb

from config import EvalConfig
from dataset import AIMEProblem


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE      = re.compile(r"\\boxed\{(\d+)\}")
_ANSWER_IS_RE  = re.compile(r"(?:the\s+answer\s+is|answer\s*[:=])\s*(\d+)", re.IGNORECASE)
_TRAILING_INT  = re.compile(r"\b(\d{1,3})\b")


def extract_answer(text: str) -> Optional[int]:
    """Parse the model output and return the predicted integer answer (0-999)."""
    # 1. \boxed{N}
    matches = _BOXED_RE.findall(text)
    if matches:
        val = int(matches[-1])
        if 0 <= val <= 999:
            return val

    # 2. "The answer is N"
    m = _ANSWER_IS_RE.search(text[::-1])   # search from the end
    if m:
        val = int(m.group(1)[::-1])
        if 0 <= val <= 999:
            return val

    # 3. Last integer in valid range
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
    raw_output: str
    predicted: Optional[int]
    correct: bool

    @property
    def year(self) -> int:          return self.problem.year
    @property
    def competition(self) -> str:   return self.problem.competition
    @property
    def problem_number(self) -> int: return self.problem.problem_number


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results: List[ProblemResult]) -> Dict:
    """Compute accuracy metrics grouped by year and competition."""
    metrics: Dict = {}

    years = sorted(set(r.year for r in results))
    total_correct = 0
    total_count = 0

    for year in years:
        year_results = [r for r in results if r.year == year]
        for comp in ["I", "II"]:
            comp_results = [r for r in year_results if r.competition == comp]
            if not comp_results:
                continue
            n_correct = sum(r.correct for r in comp_results)
            metrics[f"accuracy/{year}/AIME_{comp}"] = n_correct / len(comp_results)

        n_correct_year = sum(r.correct for r in year_results)
        metrics[f"accuracy/{year}/overall"] = (
            n_correct_year / len(year_results) if year_results else 0.0
        )
        total_correct += n_correct_year
        total_count   += len(year_results)

    metrics["accuracy/overall"] = total_correct / total_count if total_count else 0.0
    metrics["total_correct"]    = total_correct
    metrics["total_problems"]   = total_count

    return metrics


# ---------------------------------------------------------------------------
# Wandb logging
# ---------------------------------------------------------------------------

def init_wandb(config: EvalConfig):
    """Initialise a Wandb run."""
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config={
            "model":          config.model_name,
            "dtype":          config.dtype,
            "max_new_tokens": config.max_new_tokens,
            "temperature":    config.temperature,
            "do_sample":      config.do_sample,
            "batch_size":     config.batch_size,
            "years":          config.years,
        },
    )
    return run


def log_to_wandb(results: List[ProblemResult], metrics: Dict, config: EvalConfig):
    """Log scalar metrics, per-problem table, and accuracy bar charts to Wandb."""

    # --- Scalar metrics ---
    wandb.log(metrics)

    # --- Per-problem results table ---
    table = wandb.Table(columns=[
        "problem_id", "year", "competition", "problem_number",
        "problem", "ground_truth", "predicted", "correct", "raw_output",
    ])
    for r in results:
        table.add_data(
            r.problem.problem_id,
            r.year,
            r.competition,
            r.problem_number,
            r.problem.problem[:500] + ("..." if len(r.problem.problem) > 500 else ""),
            r.problem.answer,
            r.predicted if r.predicted is not None else -1,
            int(r.correct),
            r.raw_output[:1000] + ("..." if len(r.raw_output) > 1000 else ""),
        )
    wandb.log({"results/per_problem": table})

    # --- Accuracy by problem number (difficulty curve) ---
    by_num: Dict[int, List[bool]] = {}
    for r in results:
        by_num.setdefault(r.problem_number, []).append(r.correct)

    num_table = wandb.Table(columns=["problem_number", "accuracy", "n_problems"])
    for num in sorted(by_num):
        vals = by_num[num]
        num_table.add_data(num, sum(vals) / len(vals), len(vals))
    wandb.log({
        "charts/accuracy_by_problem_number": wandb.plot.bar(
            num_table, "problem_number", "accuracy",
            title="Accuracy by Problem Number (1=easiest, 15=hardest)",
        )
    })

    # --- Accuracy by year ---
    years = sorted(set(r.year for r in results))
    year_table = wandb.Table(columns=["year", "accuracy"])
    for y in years:
        yr = [r for r in results if r.year == y]
        year_table.add_data(y, sum(r.correct for r in yr) / len(yr) if yr else 0)
    wandb.log({
        "charts/accuracy_by_year": wandb.plot.bar(
            year_table, "year", "accuracy",
            title="Overall Accuracy by Year",
        )
    })

    print("\n--- Wandb metrics ---")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Save raw results to JSON
# ---------------------------------------------------------------------------

def save_results(results: List[ProblemResult], metrics: Dict, config: EvalConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    output = {
        "metrics": metrics,
        "results": [
            {
                "problem_id":     r.problem.problem_id,
                "year":           r.year,
                "competition":    r.competition,
                "problem_number": r.problem_number,
                "problem":        r.problem.problem,
                "ground_truth":   r.problem.answer,
                "predicted":      r.predicted,
                "correct":        r.correct,
                "raw_output":     r.raw_output,
            }
            for r in results
        ],
    }
    path = os.path.join(config.output_dir, "aime_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {path}")
