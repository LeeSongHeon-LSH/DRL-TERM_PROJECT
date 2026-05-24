"""SwapAwareRewardFn — PAVRewardFn의 swap 버전.

기존 PAVRewardFn은 PRM/μ가 항상 GPU에 있다고 가정.
이 클래스는 매 호출마다 SwapOrchestrator로 GPU 모델 교체.

trajectory 1개 reward 계산 흐름:
  1. swap_to("mu") — μ만 GPU
  2. K=16 alternative 생성 (각 step별)
  3. swap_to("prm") — PRM만 GPU
  4. 모든 alternative + actual에 대해 score
  5. PAV reduce → step rewards 합산
  6. reward 반환 → trainer가 다음 forward를 위해 swap_to("pi")
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import torch

from ..pav.differential import DifferentialPAV
from ..pav.mc_rollout import MCRolloutPAV
from ..pav.reduce import reduce_advantage

log = logging.getLogger(__name__)


class SwapAwareRewardFn:
    """PAVRewardFn + SwapOrchestrator 통합.

    호출 시 (problem, trajectory) → trajectory scalar reward.
    내부에서 orchestrator로 PRM/μ swap.
    """

    def __init__(
        self,
        orchestrator,           # SwapOrchestrator
        pav,                    # DifferentialPAV 또는 MCRolloutPAV (orchestrator의 prm/mu 참조)
        reducer_mode: str = "Q3",
        alpha: float = 3.0,
        lam: float = -0.5,
        cvar_alpha: float = 0.2,
        correct_bonus: float = 0.0,
    ):
        self.orchestrator = orchestrator
        self.pav = pav
        self.mode = reducer_mode
        self.alpha = alpha
        self.lam = lam
        self.cvar_alpha = cvar_alpha
        self.correct_bonus = correct_bonus
        # PAVMonitorCallback 호환 buffer (옵션)
        self.stats_buffer: Optional[deque] = None
        self.sample_buffer: Optional[deque] = None

    # ------------------------------------------------------------------ api
    @torch.no_grad()
    def __call__(self, problem: str, traj: list[str], final_correct: bool = False) -> list[float]:
        """traj의 각 step에 대해 PAV reward 계산 → step rewards list 반환.

        is_mc_rollout = (type(self.pav).__name__ == "MCRolloutPAV")
        - mc_rollout: μ로 K개 alternative 생성 → PRM score → PAV reduce
        - differential: μ 안 씀, PRM만 호출
        """
        is_mc = isinstance(self.pav, MCRolloutPAV)

        step_rewards: list[float] = []
        prefix = ""
        for i, step in enumerate(traj):
            # ---- 분포형 PAV (Phase 1) ----
            if is_mc:
                # μ로 K개 alternative
                self.orchestrator.swap_to("mu")
                alt_steps = self.orchestrator.mu.sample_step_batch(
                    problem, prefix, n=self.pav.K
                )
                # PRM score (현재 step + K개 alternative = K+1 호출 또는 batch)
                self.orchestrator.swap_to("prm")
                p_q = self.orchestrator.prm.score(problem, prefix + step)
                contexts = [prefix + a for a in alt_steps]
                p_v_samples = self.orchestrator.prm.score_batch(problem, contexts)
                # advantage 분포
                p_q_t = _as_tensor(p_q).reshape(())
                p_v_t = _as_tensor(p_v_samples).reshape(-1)
                A_samples = p_q_t - p_v_t   # [K]
                # reduce (mean / Q1 / Q3 / Q4) — reduce_advantage는 dict 받음
                out_dict = {
                    "advantage_scalar": A_samples.mean(),
                    "advantage_samples": A_samples,
                    "p_q": p_q_t,
                    "p_v_samples": p_v_t,
                }
                scalar = reduce_advantage(
                    out_dict, mode=self.mode, lam=self.lam, cvar_alpha=self.cvar_alpha
                )
                # PAVMonitor 호환 stats
                if self.stats_buffer is not None:
                    self.stats_buffer.append({
                        "A_mean": float(A_samples.mean()),
                        "A_std": float(A_samples.std()),
                        "A_q05": float(A_samples.quantile(0.05)),
                        "A_q95": float(A_samples.quantile(0.95)),
                        "p_q": float(p_q_t),
                        "p_v": float(p_v_t.mean()),
                        "Q1": scalar if self.mode == "Q1" else None,
                        "Q3": scalar if self.mode == "Q3" else None,
                    })

            # ---- 차분 PAV (Phase 0) ----
            else:
                self.orchestrator.swap_to("prm")
                p_after = self.orchestrator.prm.score(problem, prefix + step)
                p_before = self.orchestrator.prm.score(problem, prefix) if prefix else 0.5
                scalar = float(p_after - p_before)

            reward = self.alpha * scalar
            if i == len(traj) - 1 and final_correct:
                reward += self.correct_bonus
            step_rewards.append(reward)
            prefix = prefix + step + "\n"

        # sample buffer 호환
        if self.sample_buffer is not None:
            self.sample_buffer.append((problem, traj, step_rewards))

        return step_rewards


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    return torch.tensor(x, dtype=torch.float32)
