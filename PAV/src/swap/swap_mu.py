"""SwapMu — μ HF model. CPU/GPU swap 지원.

vLLM 안 씀 (vLLM은 sleep mode로 weight CPU offload 가능하지만 별도 인스턴스라 메모리 추가 차지).
대신 HF AutoModelForCausalLM + .generate() — vLLM보다 느리지만 swap 단순.

sample_step_batch는 MuSampler / RemoteMuSampler 인터페이스와 동일.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

log = logging.getLogger(__name__)


@dataclass
class SwapMuConfig:
    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    step_stop: tuple[str, ...] = ("\n\n",)
    dtype: str = "bfloat16"


class SwapMu:
    """μ HF model wrapper. CPU/GPU swap 지원.

    초기에 CPU에 로드. SwapOrchestrator가 generate 직전 .to_cuda(), 후 .to_cpu().
    """

    def __init__(self, cfg: SwapMuConfig):
        self.cfg = cfg
        self._device = "cpu"
        self._model = None
        self._tokenizer = None
        self._load_to_cpu()

    def _load_to_cpu(self):
        log.info(f"SwapMu: loading {self.cfg.model_id} → CPU")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = getattr(torch, self.cfg.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        self._model.eval()
        self._device = "cpu"
        log.info(f"SwapMu: ready on CPU ({self.cfg.model_id})")

    # ------------------------------------------------------------------ swap
    def to_cuda(self):
        if self._device == "cuda":
            return
        log.debug("SwapMu: CPU → CUDA")
        self._model.to("cuda")
        self._device = "cuda"
        torch.cuda.synchronize()

    def to_cpu(self):
        if self._device == "cpu":
            return
        log.debug("SwapMu: CUDA → CPU")
        self._model.to("cpu")
        self._device = "cpu"
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ api
    @staticmethod
    def _build_prompt(problem: str, prefix: str) -> str:
        return (
            "<|im_start|>system\nYou solve math step by step. "
            "Number each step on its own line.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"
        )

    @torch.no_grad()
    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        if self._device != "cuda":
            raise RuntimeError("SwapMu.sample_step_batch 호출 시 GPU에 있어야 함.")
        prompt = self._build_prompt(problem, prefix)
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        # num_return_sequences로 K개 alternative 생성
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        # prompt 부분 자르고 step boundary에서 truncate
        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        for out in outputs:
            text = self._tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for stop in self.cfg.step_stop:
                idx = text.find(stop)
                if idx >= 0:
                    text = text[:idx]
            completions.append(text.strip())
        return completions

    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]
