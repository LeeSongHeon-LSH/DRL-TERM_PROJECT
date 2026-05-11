"""Ray Actor for μ (Prover) serving.

μ = Qwen2.5-Math-7B base, vLLM bf16. 별도 GPU 1대 (예: PC B, RTX 3090 24GB).
MCRolloutPAV가 K=16 alternative step을 한 호출(`sample_step_batch`)에 묶어 보내므로
single actor + vLLM 내부 batching이 자연스러움 (replica pool 불필요).

`@ray.remote(num_gpus=1)`로 GPU 1장 점유.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MuHandler:
    """transport-무관 μ 처리 로직. RayMuActor와 단위 테스트가 동일하게 사용."""

    def __init__(self, mu: Any):
        self.mu = mu

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        return list(self.mu.sample_step_batch(problem, prefix, n))

    def sample_step(self, problem: str, prefix: str) -> str:
        return self.mu.sample_step(problem, prefix)

    def health(self) -> dict:
        return {
            "ok": True,
            "model_id": getattr(self.mu.cfg, "model_id", "unknown"),
        }


def _build_actor_cls():
    """lazy build — ray 미설치 환경(테스트)에서 import-time 실패 회피."""
    import ray

    @ray.remote(num_gpus=1)
    class RayMuActor:
        """μ를 적재하고 sample_step_batch / sample_step / health 노출."""

        def __init__(self, policy_yaml: str):
            from .mu_sampler import build_mu_from_policy_yaml
            # actor 내부에서는 force_local=True로 강제 (yaml의 mode='ray'여도 local vLLM 적재)
            self.mu = build_mu_from_policy_yaml(policy_yaml, force_local=True)
            self.mu._ensure_loaded()
            self._handler = MuHandler(self.mu)
            logger.info(f"RayMuActor ready: {self.mu.cfg.model_id}")

        def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
            return self._handler.sample_step_batch(problem, prefix, n)

        def sample_step(self, problem: str, prefix: str) -> str:
            return self._handler.sample_step(problem, prefix)

        def health(self) -> dict:
            return self._handler.health()

    return RayMuActor


def get_actor_cls() -> Any:
    return _build_actor_cls()
