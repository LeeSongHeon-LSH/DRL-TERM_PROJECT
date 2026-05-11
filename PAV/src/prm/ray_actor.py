"""Ray Actor for PRM serving.

각 워커 PC(5070급)에서 Ray Worker로 등록되어 PRM 1.5B를 GPU에 적재.
본체의 RayPRMClient가 `actor.score.remote(...)` 형태로 호출 → ray.get()으로 응답.

`@ray.remote(num_gpus=1)` 데코레이터로 Ray 스케줄러가 자동으로 GPU 1장 할당.
N개 replica를 띄우면 Ray가 라운드로빈 분산 (Client 측에서 N개 handle 받음).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class PRMHandler:
    """transport-무관 PRM 처리 로직. RayPRMActor 와 단위 테스트가 동일하게 사용."""

    def __init__(self, prm: Any):
        self.prm = prm

    def score(self, problem: str, solution_prefix: str) -> float:
        s = self.prm.score(problem, solution_prefix)
        return float(s.item() if hasattr(s, "item") else s)

    def score_batch(self, problem: str, solution_prefixes: Sequence[str]) -> list[float]:
        ss = self.prm.score_batch(problem, list(solution_prefixes))
        return [float(x) for x in ss.tolist()]

    def score_per_step(self, problem: str, solution: str) -> list[float]:
        return [float(x) for x in self.prm.score_per_step(problem, solution)]

    def health(self) -> dict:
        return {
            "ok": True,
            "name": getattr(self.prm.cfg, "name", "prm"),
            "model_id": getattr(self.prm.cfg, "model_id", "unknown"),
        }


def _build_actor_cls():
    """ray가 미설치된 환경(테스트 등)에서 import-time 실패를 피하기 위해 lazy 빌드."""
    import ray

    @ray.remote(num_gpus=1)
    class RayPRMActor:
        """PRM 1.5B를 적재하고 score / score_batch / score_per_step / health를 노출."""

        def __init__(self, config: str | dict):
            from .loader import load_prm
            self.prm = load_prm(config, mode="local")  # actor 내부에서는 항상 local로 강제
            # actor 생성 시점에 GPU에 모델 로드 — 첫 score 호출 시 lazy load 회피
            self.prm._ensure_loaded()
            self._handler = PRMHandler(self.prm)
            logger.info(f"RayPRMActor ready: {self.prm.cfg.model_id}")

        def score(self, problem: str, solution_prefix: str) -> float:
            return self._handler.score(problem, solution_prefix)

        def score_batch(self, problem: str, solution_prefixes: Sequence[str]) -> list[float]:
            return self._handler.score_batch(problem, solution_prefixes)

        def score_per_step(self, problem: str, solution: str) -> list[float]:
            return self._handler.score_per_step(problem, solution)

        def health(self) -> dict:
            return self._handler.health()

    return RayPRMActor


def get_actor_cls() -> Any:
    """RayPRMActor 클래스 핸들 (lazy 생성)."""
    return _build_actor_cls()
