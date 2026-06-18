"""Evaluation modules for bench_passatk."""

from .grader import extract_answer, grade_answer
from .metrics import (
    compute_pass_at_k_unbiased,
    compute_pass_at_k_values,
    compute_majority_at_k,
    compute_majority_at_k_values,
    compute_oracle,
    compute_pass_at_1,
    compute_best_of_n,
    compute_best_of_n_values,
    compute_all_metrics,
    aggregate_metrics,
    wilson_confidence_interval,
)

# Alias for convenience
compute_pass_at_k = compute_pass_at_k_unbiased

__all__ = [
    "extract_answer",
    "grade_answer",
    "compute_pass_at_k",
    "compute_pass_at_k_unbiased",
    "compute_pass_at_k_values",
    "compute_majority_at_k",
    "compute_majority_at_k_values",
    "compute_oracle",
    "compute_pass_at_1",
    "compute_best_of_n",
    "compute_best_of_n_values",
    "compute_all_metrics",
    "aggregate_metrics",
    "wilson_confidence_interval",
]