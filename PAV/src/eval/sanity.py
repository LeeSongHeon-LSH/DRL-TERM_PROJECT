"""Phase 0 통과 기준 — Sanity check S1~S4.

라벨이 붙은 (problem, prefix, step, label) 데이터 위에서 PAVMethod를 돌려
advantage의 부호 / 크기 분포를 검증.

S1. 정답 step에서 A > 0 비율 ≥ 70%
S2. 무의미한 filler step ("Let me think...")에서 |A| < 0.05 비율 ≥ 60%
S3. 오답 step에서 A < 0 비율 ≥ 60%
S4. p_v (prefix-only) 점수 분포가 단봉이 아닐 것
    — PRM이 prefix를 제대로 평가하는지 확인. 실패 시 Phase 1 MC rollout으로 자연 전환.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from ..pav.base import PAVMethod
from ..pav.reduce import reduce_advantage

StepLabel = Literal["correct", "wrong", "filler"]


@dataclass
class SanityItem:
    problem: str
    prefix: str
    step: str
    label: StepLabel


@dataclass
class SanityResult:
    s1_correct_pos_rate: float
    s2_filler_small_rate: float
    s3_wrong_neg_rate: float
    s4_pv_multimodal: bool
    n_correct: int
    n_wrong: int
    n_filler: int
    pass_all: bool


def run_sanity_checks(
    pav: PAVMethod,
    items: Iterable[SanityItem],
    *,
    s1_threshold: float = 0.70,
    s2_threshold: float = 0.60,
    s3_threshold: float = 0.60,
    filler_eps: float = 0.05,
) -> SanityResult:
    items = list(items)
    advs: dict[StepLabel, list[float]] = {"correct": [], "wrong": [], "filler": []}
    pv_scores: list[float] = []

    for it in items:
        out = pav(it.problem, it.prefix, it.step)
        a = reduce_advantage(out, mode="Q1")     # mean — 부호/크기 평가용
        advs[it.label].append(a)

        # p_v 수집 — DifferentialPAV는 p_v, MCRolloutPAV는 p_v_samples 평균
        if "p_v" in out and out["p_v"] is not None:
            pv_scores.append(float(out["p_v"]))
        elif "p_v_samples" in out and out["p_v_samples"] is not None:
            pv_scores.append(float(out["p_v_samples"].mean()))

    s1 = _rate(advs["correct"], lambda x: x > 0)
    s2 = _rate(advs["filler"], lambda x: abs(x) < filler_eps)
    s3 = _rate(advs["wrong"], lambda x: x < 0)
    s4 = _is_multimodal(pv_scores)

    pass_all = (
        s1 >= s1_threshold
        and s2 >= s2_threshold
        and s3 >= s3_threshold
        and s4
    )
    return SanityResult(
        s1_correct_pos_rate=s1,
        s2_filler_small_rate=s2,
        s3_wrong_neg_rate=s3,
        s4_pv_multimodal=s4,
        n_correct=len(advs["correct"]),
        n_wrong=len(advs["wrong"]),
        n_filler=len(advs["filler"]),
        pass_all=pass_all,
    )


def _rate(xs, predicate) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if predicate(x)) / len(xs)


def _is_multimodal(values: list[float], n_bins: int = 20, min_modes: int = 2) -> bool:
    """단순 히스토그램 mode 카운트 — 정밀하지 않으나 S4 게이트 용도로 충분."""
    if len(values) < 30:
        return False
    t = torch.tensor(values)
    hist = torch.histogram(t, bins=n_bins).hist
    # local maxima 카운트 (이웃보다 크고 0 아님)
    modes = 0
    for i in range(1, n_bins - 1):
        if hist[i] > 0 and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            # 평탄 구간 중복 카운트 방지
            if hist[i] > hist[i - 1] or hist[i] > hist[i + 1]:
                modes += 1
    return modes >= min_modes
