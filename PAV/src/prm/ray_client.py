"""RayPRMClient — Ray actor handle을 기존 PRM 인터페이스로 wrap.

PAV 코어(DifferentialPAV/MCRolloutPAV/PAVRewardFn)는 이 객체를 그대로 받아
`score()` / `score_batch()` / `score_per_step()`을 호출. transport(Ray)는 투명.

여러 replica를 사용하려면 RayPRMClientPool 사용 — Client 1개당 actor 1개,
Pool이 라운드로빈으로 분산.

설계 결정:
  - `ray.get()`로 동기 결과 받음 (기존 PRM 인터페이스가 동기).
  - asyncio 통합은 추후 필요 시 별도 client (호환성 우선).
"""
from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class RayPRMConfig:
    name: str = "ray-prm"
    model_id: str = "ray-actor"          # 표시용
    actor_name: str = "prm-actor"        # named actor 이름 (단일) 또는 prefix (multi-replica)
    namespace: str = "pav-rl"
    num_replicas: int = 1                # 1이면 단일 actor, 2+면 pool
    rpc_timeout: float = 120.0
    step_token: str = "\n"
    batch_size: int = 16


class RayPRMClient:
    """단일 Ray actor handle wrapper. PRM과 동일 인터페이스."""

    def __init__(self, cfg: RayPRMConfig, actor_handle: Any | None = None):
        """
        Args:
            cfg: 표시/메타 정보
            actor_handle: 이미 얻은 ray actor handle. None이면 ray.get_actor(cfg.actor_name)로 조회.
        """
        import ray

        self.cfg = cfg
        if actor_handle is None:
            actor_handle = ray.get_actor(cfg.actor_name, namespace=cfg.namespace)
        self._actor = actor_handle

    # ------------------------------------------------------------------ api
    @torch.no_grad()
    def score(self, problem: str, solution_prefix: str) -> torch.Tensor:
        if not solution_prefix.strip():
            return torch.tensor(0.5)
        import ray
        s = ray.get(
            self._actor.score.remote(problem, solution_prefix),
            timeout=self.cfg.rpc_timeout,
        )
        return torch.tensor(float(s))

    @torch.no_grad()
    def score_batch(
        self, problem: str, solution_prefixes: Sequence[str]
    ) -> torch.Tensor:
        prefixes = list(solution_prefixes)
        if not prefixes:
            return torch.empty(0)
        import ray
        # 한 RPC에 모두 묶어 보냄 (한 actor가 내부 batch로 처리)
        out: list[float] = []
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(prefixes), bs):
            chunk = prefixes[i : i + bs]
            scores = ray.get(
                self._actor.score_batch.remote(problem, chunk),
                timeout=self.cfg.rpc_timeout,
            )
            out.extend(float(x) for x in scores)
        return torch.tensor(out, dtype=torch.float32)

    @torch.no_grad()
    def score_per_step(self, problem: str, solution: str) -> list[float]:
        import ray
        ps = ray.get(
            self._actor.score_per_step.remote(problem, solution),
            timeout=self.cfg.rpc_timeout,
        )
        return [float(x) for x in ps]

    def health(self) -> dict:
        import ray
        return ray.get(self._actor.health.remote(), timeout=10.0)

    def close(self):
        # actor lifecycle은 명시적으로 등록한 측이 책임 (예: serve_prm_ray.py)
        self._actor = None


class RayPRMClientPool:
    """N개 replica를 라운드로빈으로 분산. 단일 RayPRMClient와 같은 인터페이스."""

    def __init__(self, cfg: RayPRMConfig, actor_handles: list[Any] | None = None):
        import ray

        self.cfg = cfg
        if actor_handles is None:
            actor_handles = [
                ray.get_actor(f"{cfg.actor_name}-{i}", namespace=cfg.namespace)
                for i in range(cfg.num_replicas)
            ]
        if not actor_handles:
            raise ValueError("RayPRMClientPool: actor_handles 비어있음")
        self._handles = actor_handles
        self._cycle = itertools.cycle(self._handles)
        self._lock = threading.Lock()

    def _next(self) -> Any:
        with self._lock:
            return next(self._cycle)

    @torch.no_grad()
    def score(self, problem: str, solution_prefix: str) -> torch.Tensor:
        if not solution_prefix.strip():
            return torch.tensor(0.5)
        import ray
        h = self._next()
        s = ray.get(
            h.score.remote(problem, solution_prefix),
            timeout=self.cfg.rpc_timeout,
        )
        return torch.tensor(float(s))

    @torch.no_grad()
    def score_batch(
        self, problem: str, solution_prefixes: Sequence[str]
    ) -> torch.Tensor:
        prefixes = list(solution_prefixes)
        if not prefixes:
            return torch.empty(0)
        import ray

        # batch를 N개 chunk로 쪼개 N개 actor에 병렬 분산
        n = len(self._handles)
        chunk_size = (len(prefixes) + n - 1) // n
        chunks = [prefixes[i : i + chunk_size] for i in range(0, len(prefixes), chunk_size)]
        futures = [
            self._handles[i].score_batch.remote(problem, chunk)
            for i, chunk in enumerate(chunks)
        ]
        results = ray.get(futures, timeout=self.cfg.rpc_timeout)
        out: list[float] = []
        for r in results:
            out.extend(float(x) for x in r)
        return torch.tensor(out, dtype=torch.float32)

    @torch.no_grad()
    def score_per_step(self, problem: str, solution: str) -> list[float]:
        import ray
        h = self._next()
        ps = ray.get(
            h.score_per_step.remote(problem, solution),
            timeout=self.cfg.rpc_timeout,
        )
        return [float(x) for x in ps]

    def health(self) -> list[dict]:
        import ray
        return ray.get([h.health.remote() for h in self._handles], timeout=10.0)

    def close(self):
        self._handles = []
