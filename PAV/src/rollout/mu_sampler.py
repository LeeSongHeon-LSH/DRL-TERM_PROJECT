"""Prover μ — frozen base model, single-step sampler.

MCRolloutPAV에서 V를 추정하기 위해 매 prefix 위치마다 K개의 single-step rollout이 필요.
요구사항:
    - μ ≠ π (μ는 frozen base, π는 LoRA로 학습 중)
    - vLLM prefix caching으로 K-rollout 비용 ≪ K × 단일 forward
    - 한 step (\\n\\n 또는 헤더 패턴)에서 stop
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class MuConfig:
    model_id: str
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    # Skywork는 단일 \n 경계, Qwen2.5-Math는 \n\n로 step 출력 — 둘 다 허용
    step_stop: tuple[str, ...] = ("\n\n",)
    gpu_memory_utilization: float = 0.30
    max_model_len: int = 4096        # KV cache 메모리 제한 (default 32k는 OOM 유발)
    dtype: str = "bfloat16"


class MuSampler:
    """μ.sample_step / sample_step_batch — vLLM 백엔드, lazy load."""

    def __init__(self, cfg: MuConfig):
        self.cfg = cfg
        self._llm = None

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(
            model=self.cfg.model_id,
            enable_prefix_caching=True,
            dtype=self.cfg.dtype,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_model_len=self.cfg.max_model_len,
        )

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build_prompt(problem: str, prefix: str) -> str:
        # Qwen2.5 chat template 직접 작성 — μ가 base 모델이라 chat fine-tune이 약할 수 있어
        # 단순 system + user 형식이 안전. prefix는 assistant 답변 진행 중인 상태로 주입.
        return (
            "<|im_start|>system\nYou solve math step by step. Number each step on its own line.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"
        )

    # ------------------------------------------------------------------ api
    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        """K=n개 alternative single-step. vLLM prefix cache hit 기대."""
        self._ensure_loaded()
        from vllm import SamplingParams

        prompt = self._build_prompt(problem, prefix)
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
            stop=list(self.cfg.step_stop),
            n=n,
        )
        outputs = self._llm.generate([prompt], sp)
        completions = outputs[0].outputs
        return [c.text.strip() for c in completions]


def build_mu_from_policy_yaml(path: str | Path):
    """policy.yaml의 mu: 섹션에서 MuSampler 또는 RemoteMuSampler 생성.

    mode == "remote" 이면 remote_urls의 μ 서버에 HTTP로 sample 요청.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mu_cfg = data.get("mu", {})

    mode = mu_cfg.get("mode", "local")
    if mode == "remote":
        from .remote_mu import RemoteMuConfig, RemoteMuSampler
        return RemoteMuSampler(
            RemoteMuConfig(
                amqp_url=mu_cfg.get("amqp_url", "amqp://guest:guest@localhost:5672/"),
                request_queue=mu_cfg.get("request_queue", "mu.requests"),
                rpc_timeout=mu_cfg.get("rpc_timeout", 180.0),
                temperature=mu_cfg.get("temperature", 1.0),
                top_p=mu_cfg.get("top_p", 0.95),
                max_new_tokens=mu_cfg.get("max_new_tokens", 256),
            )
        )

    return MuSampler(
        MuConfig(
            model_id=mu_cfg["model_id"],
            temperature=mu_cfg.get("temperature", 1.0),
            top_p=mu_cfg.get("top_p", 0.95),
            max_new_tokens=mu_cfg.get("max_new_tokens", 256),
            step_stop=tuple(mu_cfg.get("step_stop", ["\n\n"])),
            gpu_memory_utilization=mu_cfg.get("gpu_memory_utilization", 0.30),
            max_model_len=mu_cfg.get("max_model_len", 4096),
            dtype=mu_cfg.get("dtype", "bfloat16"),
        )
    )
