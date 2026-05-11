"""분포 ↔ 스칼라 통합 reducer.

DifferentialPAV (스칼라) ↔ MCRolloutPAV (분포) 모두 동일 인터페이스로 처리.
B1 / Q1 / Q3 / Q4 모드 지원 — Phase 0 ↔ Phase 1 swap 비용 0.
"""
from __future__ import annotations

import torch


def reduce_advantage(
    out: dict,
    mode: str = "Q3",
    lam: float = -0.5,
    cvar_alpha: float = 0.2,
) -> float:
    """PAVMethod 출력 dict → scalar reward 항.

    Args:
        out: PAVMethod.__call__ 반환값
            - advantage_scalar: 0-d tensor (필수)
            - advantage_samples: [K] tensor 또는 None
        mode:
            B1 — sign(A) ∈ {0, 1}
            Q1 — mean
            Q3 — mean − λ·std  (메인, λ=−0.5 risk-seeking) ⭐
            Q4 — CVaR_α (lower tail mean)
        lam: Q3의 λ (음수면 risk-seeking — std에 보너스)
        cvar_alpha: Q4의 분위수 (0.2 → 하위 20%)

    Returns:
        scalar float — PAVRewardFn에서 r_h += α·Ã 로 가산됨
    """
    A = out.get("advantage_samples")

    # 분포 없음 (Phase 0 DifferentialPAV) → scalar fallback
    if A is None or (isinstance(A, torch.Tensor) and A.numel() == 0):
        scalar = out["advantage_scalar"]
        if mode == "B1":
            return float((scalar > 0).item())
        return float(scalar.item())

    # 분포 있음 (Phase 1 MCRolloutPAV)
    if mode == "B1":
        return float((A.mean() > 0).item())
    if mode == "Q1":
        return float(A.mean().item())
    if mode == "Q3":
        return float((A.mean() - lam * A.std(unbiased=False)).item())
    if mode == "Q4":
        k = max(1, int(A.numel() * cvar_alpha))
        sorted_A, _ = A.sort()
        return float(sorted_A[:k].mean().item())
    raise ValueError(f"Unknown reducer mode: {mode!r}")
