"""BoN-PAV — Best-of-N selection with PAV scoring.

G0 / G1 통과 기준에 사용:
  G0: BoN-PAV ≥ BoN-PRM       (Phase 0 차분 PAV가 단순 PRM 합산보다 낫거나 같음)
  G1: BoN-PAV(분포) ≥ BoN-PAV(스칼라)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..pav.base import PAVMethod
from ..pav.reduce import reduce_advantage
from ..rollout.parser import split_steps


@dataclass
class BoNExample:
    problem: str
    candidates: list[str]
    correctness: list[bool]


def _score_with_pav(
    pav: PAVMethod,
    problem: str,
    completion: str,
    *,
    mode: str = "Q1",
    lam: float = -0.5,
    aggregate: str = "sum",
) -> float:
    """한 trajectory에 대해 step-wise PAV → trajectory score (sum / mean / min)."""
    steps = split_steps(completion)
    if not steps:
        return 0.0
    prefix = ""
    vals: list[float] = []
    for step in steps:
        out = pav(problem, prefix, step)
        vals.append(reduce_advantage(out, mode=mode, lam=lam))
        prefix = prefix + step
        if not prefix.endswith("\n\n"):
            prefix = prefix + "\n\n"
    if aggregate == "sum":
        return sum(vals)
    if aggregate == "mean":
        return sum(vals) / len(vals)
    if aggregate == "min":
        return min(vals)
    raise ValueError(aggregate)


def _score_with_prm_sum(prm, problem: str, completion: str) -> float:
    """BoN-PRM baseline — step별 PRM 점수 합산 (PAV 차분 X)."""
    steps = split_steps(completion)
    prefix = ""
    total = 0.0
    for step in steps:
        prefix = prefix + step
        if not prefix.endswith("\n\n"):
            prefix = prefix + "\n\n"
        total += float(prm.score(problem, prefix))
    return total


def bon_pav(
    examples: Sequence[BoNExample],
    pav: PAVMethod,
    *,
    mode: str = "Q1",
    lam: float = -0.5,
    aggregate: str = "sum",
) -> float:
    """BoN-PAV 정확도 (선택된 후보가 정답인 비율)."""
    correct = 0
    for ex in examples:
        scores = [
            _score_with_pav(pav, ex.problem, c, mode=mode, lam=lam, aggregate=aggregate)
            for c in ex.candidates
        ]
        best = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(ex.correctness[best])
    return correct / max(1, len(examples))


def bon_prm(examples: Sequence[BoNExample], prm) -> float:
    """비교용 BoN-PRM 정확도."""
    correct = 0
    for ex in examples:
        scores = [_score_with_prm_sum(prm, ex.problem, c) for c in ex.candidates]
        best = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(ex.correctness[best])
    return correct / max(1, len(examples))
