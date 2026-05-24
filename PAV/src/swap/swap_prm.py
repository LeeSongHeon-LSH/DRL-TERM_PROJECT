"""SwapPRM — PRM wrapper. CPU/GPU 사이 swap 지원.

기존 src.prm.PRM을 감싸서 .to_cuda(), .to_cpu() 추가.
score/score_batch는 PRM 위임 — 단, GPU에 있을 때만 호출 가능 (orchestrator가 보장).
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch

log = logging.getLogger(__name__)


class SwapPRM:
    """PRM CPU/GPU swap wrapper.

    초기에 CPU에 로드 (메모리만 차지). SwapOrchestrator가 score 호출 직전 .to_cuda(),
    호출 끝나면 .to_cpu()로 GPU 양보.
    """

    def __init__(self, prm):
        """prm: src.prm.PRM 인스턴스 (이미 load됨). 처음엔 GPU에 있다고 가정 → CPU로 옮김."""
        self.prm = prm
        self._device = "cuda"
        # PRM은 _ensure_loaded()로 가중치 로드. 처음에 GPU에 있음.
        # 시작 시 CPU로 swap (학습 PC GPU 양보)
        self.to_cpu()

    @property
    def cfg(self):
        return self.prm.cfg

    def to_cuda(self):
        if self._device == "cuda":
            return
        log.debug("SwapPRM: CPU → CUDA")
        self.prm._ensure_loaded()   # 가중치 로드 보장
        if hasattr(self.prm, "model") and self.prm._model is not None:
            self.prm._model.to("cuda")
        self._device = "cuda"
        torch.cuda.synchronize()

    def to_cpu(self):
        if self._device == "cpu":
            return
        log.debug("SwapPRM: CUDA → CPU")
        if hasattr(self.prm, "model") and self.prm._model is not None:
            self.prm._model.to("cpu")
        self._device = "cpu"
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ api
    @torch.no_grad()
    def score(self, problem: str, solution_prefix: str) -> torch.Tensor:
        if self._device != "cuda":
            raise RuntimeError("SwapPRM.score 호출 시 GPU에 있어야 함. orchestrator.swap_to('prm') 호출 후 사용.")
        return self.prm.score(problem, solution_prefix)

    @torch.no_grad()
    def score_batch(self, problem: str, solution_prefixes: Sequence[str]) -> torch.Tensor:
        if self._device != "cuda":
            raise RuntimeError("SwapPRM.score_batch 호출 시 GPU에 있어야 함.")
        return self.prm.score_batch(problem, solution_prefixes)

    @torch.no_grad()
    def score_per_step(self, problem: str, solution: str) -> list[float]:
        if self._device != "cuda":
            raise RuntimeError("SwapPRM.score_per_step 호출 시 GPU에 있어야 함.")
        return self.prm.score_per_step(problem, solution)
