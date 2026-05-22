"""RemoteMuSampler — vLLM의 OpenAI 호환 API 클라이언트.

추론 PC에서 vLLM stock 서버를 띄우면 됨 (별도 서버 코드 X):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-Math-7B-Instruct \
        --gpu-memory-utilization 0.65 \
        --max-model-len 4096 --port 8001

기존 MuSampler와 동일한 (sample_step, sample_step_batch) 인터페이스.
MCRolloutPAV는 그대로 받음.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RemoteMuConfig:
    endpoint: str = "http://localhost:8001"
    model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    timeout: float = 600.0   # K=16 batch generation은 추론 서버에서 ~120-180초 소요 가능 → 여유 600초
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    step_stop: tuple[str, ...] = ("\n\n",)


class _CfgShim:
    """vLLM 어떤 코드도 cfg.model_id를 참조할 수 있게 — 호환용."""

    def __init__(self, model_id: str):
        self.model_id = model_id


class RemoteMuSampler:
    """MuSampler와 동일 인터페이스 — sample_step / sample_step_batch (HTTP transport)."""

    def __init__(self, cfg: RemoteMuConfig):
        self.cfg = cfg
        self._client = None
        self._endpoint = cfg.endpoint.rstrip("/")
        # MuSampler처럼 .cfg.model_id 접근 호환
        # (이미 RemoteMuConfig에 model_id 있어 추가 작업 불필요)

    # ------------------------------------------------------------------ http
    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.cfg.timeout)
        return self._client

    @staticmethod
    def _build_prompt(problem: str, prefix: str) -> str:
        # MuSampler와 동일 prompt 포맷 (Qwen2.5 chat template)
        return (
            "<|im_start|>system\nYou solve math step by step. "
            "Number each step on its own line.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"
        )

    def _call_completions(self, prompt: str, n: int) -> list[str]:
        """vLLM OpenAI /v1/completions 호출 — n개 alternative 생성."""
        client = self._get_client()
        payload = {
            "model": self.cfg.model_id,
            "prompt": prompt,
            "n": n,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_new_tokens,
            "stop": list(self.cfg.step_stop),
        }
        resp = client.post(f"{self._endpoint}/v1/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"μ HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        # OpenAI 포맷: {"choices": [{"text": "...", "index": 0}, ...]}
        return [c["text"].strip() for c in data["choices"]]

    # ------------------------------------------------------------------ api
    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        prompt = self._build_prompt(problem, prefix)
        return self._call_completions(prompt, n=n)

    # ------------------------------------------------------------------ admin
    def health(self) -> dict:
        """vLLM은 /health 가 200/empty 반환."""
        client = self._get_client()
        resp = client.get(f"{self._endpoint}/health")
        return {"ok": resp.status_code == 200, "model_id": self.cfg.model_id}

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
