"""RayMuClient — Ray actor handle을 기존 MuSampler 인터페이스로 wrap.

MCRolloutPAV는 이 객체를 그대로 받아 `sample_step_batch()` 호출. transport(Ray)는 투명.

μ는 단일 instance만 사용 (보통 K=16 alternative를 한 RPC에 묶어 보내고 vLLM 내부 batching) →
Pool 클래스 불필요.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RayMuConfig:
    actor_name: str = "mu-actor"
    namespace: str = "pav-rl"
    rpc_timeout: float = 180.0
    # sampling 힌트 (현재 actor 측 cfg가 우선)
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256


class RayMuClient:
    """단일 Ray actor handle wrapper. MuSampler와 동일 인터페이스."""

    def __init__(self, cfg: RayMuConfig, actor_handle: Any | None = None):
        import ray

        self.cfg = cfg
        if actor_handle is None:
            actor_handle = ray.get_actor(cfg.actor_name, namespace=cfg.namespace)
        self._actor = actor_handle

    # ------------------------------------------------------------------ api
    def sample_step(self, problem: str, prefix: str) -> str:
        import ray
        return ray.get(
            self._actor.sample_step.remote(problem, prefix),
            timeout=self.cfg.rpc_timeout,
        )

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        import ray
        steps = ray.get(
            self._actor.sample_step_batch.remote(problem, prefix, n),
            timeout=self.cfg.rpc_timeout,
        )
        return list(steps)

    def health(self) -> dict:
        import ray
        return ray.get(self._actor.health.remote(), timeout=10.0)

    def close(self):
        self._actor = None
