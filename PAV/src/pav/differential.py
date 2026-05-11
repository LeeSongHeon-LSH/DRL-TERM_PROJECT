"""Phase 0 — DifferentialPAV.

A = PRM(s + a) − PRM(s)  (scalar 1개)
추가 학습 0, smoke test / S1~S4 sanity / G0 baseline 용.
"""
from __future__ import annotations

import torch

from .base import PAVMethod  # noqa: F401  (Protocol 만족 표시)


class DifferentialPAV:
    """A = PRM(s+a) − PRM(s) — scalar 1개. PAVMethod 만족."""

    name = "differential"

    def __init__(self, prm):
        self.prm = prm

    @torch.no_grad()
    def __call__(self, problem: str, prefix: str, step: str) -> dict:
        p_q = self.prm.score(problem, prefix + step)
        p_v = self.prm.score(problem, prefix)
        # tensor로 통일 — reducer가 .item() 호출
        p_q_t = _as_tensor(p_q)
        p_v_t = _as_tensor(p_v)
        return {
            "advantage_scalar": p_q_t - p_v_t,
            "advantage_samples": None,   # 분포 없음 → reducer가 scalar fallback
            "p_q": p_q_t,
            "p_v": p_v_t,
        }


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    return torch.tensor(float(x))
