"""PAVMethod Protocol — advantage 추출 방식의 통일 인터페이스.

새 추출 방식 (BetaPosterior, Lookahead, Ensemble, LogitDifferential …)을 추가할 때
이 Protocol만 만족하면 RL 조립 코드(`PAVRewardFn`, GRPO trainer)는 수정 불필요.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PAVMethod(Protocol):
    """advantage 추출 방식의 통일 인터페이스.

    구현체는 (problem, prefix, step) → dict 반환.
      필수: advantage_scalar     (단일 값, scalar fallback용; 0-d tensor 또는 float)
      선택: advantage_samples    [K] 분포일 때만 — 없으면 reducer가 scalar로 fallback
      디버깅: p_q, p_v, p_v_samples 등
    """

    name: str

    def __call__(
        self,
        problem: str,
        prefix: str,
        step: str,
    ) -> dict[str, torch.Tensor | None]: ...


def is_distributional(out: dict) -> bool:
    """out["advantage_samples"]가 비어있지 않으면 분포형."""
    samples = out.get("advantage_samples")
    return samples is not None and getattr(samples, "numel", lambda: 0)() > 0
