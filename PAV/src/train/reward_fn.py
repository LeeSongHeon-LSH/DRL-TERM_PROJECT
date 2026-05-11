"""PAVRewardFn — PAV 메인수식 eq (5).

    r_h = R_ex · 𝟙[h = H] + α · Ã_h

`PAVMethod` 인터페이스 하나만 받음 — Phase 0 (DifferentialPAV) ↔ Phase 1 (MCRolloutPAV)
swap은 `pav` 객체 교체로 끝. 이 클래스 / GRPO trainer는 수정 불필요.

추가:
  - stats_buffer/sample_buffer attach 가능 (PAVMonitorCallback이 wandb로 push)
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import torch

from ..pav.base import PAVMethod
from ..pav.reduce import reduce_advantage


class PAVRewardFn:
    def __init__(
        self,
        pav: PAVMethod,
        alpha: float = 3.0,
        mode: str = "Q3",
        lam: float = -0.5,
        cvar_alpha: float = 0.2,
    ):
        """
        Args:
            pav: PAVMethod (DifferentialPAV / MCRolloutPAV / …)
            alpha: PAV 가중치 (1.5 ~ 5.5)
            mode:  B1 | Q1 | Q3 | Q4
            lam:   Q3 λ (메인 = -0.5)
            cvar_alpha: Q4 분위수
        """
        self.pav = pav
        self.alpha = alpha
        self.mode = mode
        self.lam = lam
        self.cvar_alpha = cvar_alpha
        # Callback이 attach (없으면 무동작)
        self.stats_buffer: Optional[deque] = None
        self.sample_buffer: Optional[deque] = None

    def __call__(
        self,
        problem: str,
        trajectory: Sequence[str],
        final_correct: bool,
    ) -> list[float]:
        """trajectory: List[step_str]. 길이 H의 step-wise reward [H] 반환."""
        rewards: list[float] = []
        prefix = ""
        H = len(trajectory)
        for h, step in enumerate(trajectory):
            out = self.pav(problem, prefix, step)
            a_tilde = reduce_advantage(
                out, mode=self.mode, lam=self.lam, cvar_alpha=self.cvar_alpha
            )
            r_ex = float(final_correct) if h == H - 1 else 0.0
            rewards.append(r_ex + self.alpha * a_tilde)
            self._push_stats(out)
            prefix = prefix + step
            if not prefix.endswith("\n"):
                prefix = prefix + "\n"

        if self.sample_buffer is not None:
            self.sample_buffer.append((problem, list(trajectory), list(rewards)))
        return rewards

    # ------------------------------------------------------------- internals
    def _push_stats(self, out: dict) -> None:
        if self.stats_buffer is None:
            return
        a_samples = out.get("advantage_samples")
        record: dict[str, float | None] = {
            "p_q": _to_float(out.get("p_q")),
            "p_v": _to_float(out.get("p_v")) or _to_float(_mean(out.get("p_v_samples"))),
        }
        if isinstance(a_samples, torch.Tensor) and a_samples.numel() > 0:
            record.update(
                A_mean=float(a_samples.mean().item()),
                A_std=float(a_samples.std(unbiased=False).item()),
                A_q05=float(a_samples.quantile(0.05).item()),
                A_q95=float(a_samples.quantile(0.95).item()),
                Q1=float(a_samples.mean().item()),
                Q3=float((a_samples.mean() - self.lam * a_samples.std(unbiased=False)).item()),
            )
        else:
            scalar = out.get("advantage_scalar")
            v = _to_float(scalar)
            record.update(A_mean=v, A_std=0.0, A_q05=v, A_q95=v, Q1=v, Q3=v)
        self.stats_buffer.append(record)


# ------------------------------------------------------------------ helpers
def _to_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item()) if x.numel() == 1 else float(x.float().mean().item())
    return float(x)


def _mean(x) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor) and x.numel() > 0:
        return x.mean()
    return None


# ---------------------------------------------------------------- factory
def build_pav_from_config(cfg: dict, prm, mu=None) -> PAVMethod:
    """rl_q3.yaml의 pav: 섹션에서 PAVMethod 인스턴스 생성.

    Args:
        cfg: {"method": "differential" | "mc_rollout", "K": int}
        prm: src.prm.PRM
        mu:  src.rollout.MuSampler — mc_rollout에서 필수

    Returns:
        PAVMethod (DifferentialPAV 또는 MCRolloutPAV)
    """
    method = cfg.get("method", "mc_rollout")
    if method == "differential":
        from ..pav.differential import DifferentialPAV
        return DifferentialPAV(prm)
    if method == "mc_rollout":
        from ..pav.mc_rollout import MCRolloutPAV
        if mu is None:
            raise ValueError("mc_rollout 방식은 MuSampler가 필요합니다.")
        return MCRolloutPAV(prm, mu, K=cfg.get("K", 16))
    raise ValueError(f"Unknown PAV method: {method!r}")
