"""SwapOrchestrator — 단일 GPU에서 모델 (π vLLM / PRM / μ HF) 동적 swap.

목적: 단일 PC 24GB GPU에서 1.5B Full FT + Phase 1 (K=16 μ rollout) 가능하게.

동작:
  - 어느 한 시점에 한 모델만 GPU에 활성 ("current")
  - swap_to(target): current 모델 → CPU, target 모델 → GPU
  - vLLM은 sleep(level=1)로 KV cache만 비움 (weight는 GPU 유지)
  - PRM/μ는 model.to('cuda' / 'cpu')로 수동 swap

메모리 시나리오 (24 GB GPU, 1.5B Full FT):
  학습 모델 (vLLM share)         ~3 GB (항상)
  학습 grad bf16                 ~3 GB (학습 단계만)
  GaLore state                   ~1.5 GB (학습 단계만)
  vLLM KV cache (wake_up 시)     ~3-5 GB
  PRM 8bit (swap)                ~4 GB (PRM scoring 시만)
  μ HF (swap)                    ~4 GB (μ rollout 시만)
  activations                    ~3 GB (학습 forward 시만)
  ────────────────────────────────────────────
  peak (학습 단계):              ~14 GB
  peak (PRM 단계):               ~7 GB (학습 grad 활성, PRM 추가)
  peak (μ 단계):                 ~7 GB
  → 모두 24 GB 안 안전 ✅
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import torch

log = logging.getLogger(__name__)

ModelState = Literal["pi", "prm", "mu", "none"]


class SwapOrchestrator:
    """π vLLM / PRM / μ HF 모델의 GPU↔CPU swap 관리.

    π vLLM (TRL GRPOTrainer가 관리)은 sleep mode로 KV cache 비움.
    PRM, μ는 nn.Module이라 .to('cuda'/'cpu')로 직접 swap.
    """

    def __init__(
        self,
        pi_vllm=None,    # vLLM LLM 인스턴스 (None이면 vLLM 안 씀)
        prm=None,        # SwapPRM 또는 None
        mu=None,         # SwapMu 또는 None
    ):
        self.pi_vllm = pi_vllm
        self.prm = prm
        self.mu = mu
        self.current: ModelState = "pi"   # 시작 시 π가 GPU 점유 (학습 모델)

    # ------------------------------------------------------------------ core
    def swap_to(self, target: ModelState):
        """현재 GPU 모델을 offload하고 target을 GPU로 load."""
        if self.current == target:
            return
        log.debug(f"SwapOrchestrator: {self.current} → {target}")

        # 1) 현재 모델 offload
        self._offload_current()

        # 2) target 모델 load
        self._load_target(target)

        self.current = target
        torch.cuda.synchronize()   # swap 완료 대기

    def _offload_current(self):
        if self.current == "pi" and self.pi_vllm is not None:
            try:
                # vLLM sleep level 1 — KV cache 비움, weight는 GPU 유지
                # (level 2: weight도 CPU. 다만 wake_up 시 GPU copy 시간 + 메모리 hop)
                self.pi_vllm.sleep(level=1)
            except Exception as e:
                log.warning(f"vLLM sleep 실패: {e}")
        elif self.current == "prm" and self.prm is not None:
            self.prm.to_cpu()
        elif self.current == "mu" and self.mu is not None:
            self.mu.to_cpu()
        # 메모리 fragmentation 회피
        torch.cuda.empty_cache()

    def _load_target(self, target: ModelState):
        if target == "pi" and self.pi_vllm is not None:
            try:
                self.pi_vllm.wake_up()
            except Exception as e:
                log.warning(f"vLLM wake_up 실패: {e}")
        elif target == "prm" and self.prm is not None:
            self.prm.to_cuda()
        elif target == "mu" and self.mu is not None:
            self.mu.to_cuda()

    # ------------------------------------------------------------------ helpers
    def gpu_memory_used_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def report(self) -> str:
        return f"[Swap current={self.current} GPU={self.gpu_memory_used_gb():.1f}GB]"
