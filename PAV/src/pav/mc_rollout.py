"""Phase 1 ⭐ — MCRolloutPAV (메인).

V를 K개 μ-rollout으로 추정 → advantage가 K개 sample 분포:
    A_k = PRM(s + a) − PRM(s + a_k),   a_k ~ μ(·|s),   k = 1..K

장점:
  - 진짜 V 분포 (Phase 0의 prefix-only PRM 근사 없음)
  - μ ≠ π 명시적 (Theorem 3.1 자연 만족)
  - vLLM prefix caching으로 K rollout 비용 ≪ K × 단일 forward
"""
from __future__ import annotations

import torch

from .base import PAVMethod  # noqa: F401


class MCRolloutPAV:
    """V를 K개 μ-rollout으로 추정해 advantage 분포 생성. PAVMethod 만족."""

    name = "mc_rollout"

    def __init__(self, prm, mu, K: int = 16):
        """
        Args:
            prm: src.prm.PRM — score(), score_batch() 제공
            mu:  src.rollout.MuSampler — sample_step(problem, prefix) → step_str
            K:   per-step μ-rollout 수 (16 권장)
        """
        self.prm = prm
        self.mu = mu
        self.K = K

    @torch.no_grad()
    def __call__(self, problem: str, prefix: str, step: str) -> dict:
        # Q deterministic — π가 실제로 둔 step
        p_q = self.prm.score(problem, prefix + step)
        p_q_t = _as_tensor(p_q).reshape(())  # 0-d

        # K개 alternative step from μ — vLLM prefix cache 활용
        alt_steps = self.mu.sample_step_batch(problem, prefix, n=self.K)
        contexts = [prefix + a for a in alt_steps]
        p_v_samples = self.prm.score_batch(problem, contexts)   # [K]
        p_v_t = _as_tensor(p_v_samples).reshape(-1)

        A_samples = p_q_t - p_v_t   # [K]
        return {
            "advantage_scalar": A_samples.mean(),
            "advantage_samples": A_samples,
            "p_q": p_q_t,
            "p_v_samples": p_v_t,
        }


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    return torch.tensor(x, dtype=torch.float32)
