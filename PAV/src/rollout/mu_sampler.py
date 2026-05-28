"""Prover μ — frozen base model, single-step sampler.

MCRolloutPAV에서 V를 추정하기 위해 매 prefix 위치마다 K개의 single-step rollout이 필요.
요구사항:
    - μ ≠ π (μ는 frozen base, π는 LoRA로 학습 중)
    - vLLM prefix caching으로 K-rollout 비용 ≪ K × 단일 forward
    - 한 step (\\n\\n 또는 헤더 패턴)에서 stop

두 가지 백엔드:
    - MuSampler         : 별도 vLLM 인스턴스 (base 추가 copy. 빠름.)
    - SharedHFMuSampler : trainer의 PEFT 모델을 disable_adapter()로 재활용 (1-copy 진정 공유)
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
    # CPU offload — 가중치 일부를 호스트 RAM에 두고 forward 시 PCIe로 끌어옴.
    # 0이면 비활성. μ 7B(15 GB)를 24GB GPU에 올릴 때 예: 10 → GPU 점유 ~5 GB로 압축.
    # decode 2.5–4× 느려짐. PCIe 4.0 x16 기준.
    cpu_offload_gb: float = 0.0


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

        llm_kwargs = dict(
            model=self.cfg.model_id,
            enable_prefix_caching=True,
            dtype=self.cfg.dtype,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_model_len=self.cfg.max_model_len,
        )
        if self.cfg.cpu_offload_gb and self.cfg.cpu_offload_gb > 0:
            llm_kwargs["cpu_offload_gb"] = float(self.cfg.cpu_offload_gb)
        self._llm = LLM(**llm_kwargs)

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


class SharedHFMuSampler:
    """μ가 trainer의 PEFT 모델을 disable_adapter()로 재활용 — 1-copy 진정 공유.

    GPU에 base 가중치 1 copy만 존재 (trainer가 보유). μ rollout은 같은 모델에서
    LoRA를 끄고 generate() — frozen base에서 sampling하는 효과.

    Sequence:
        1. build_policy() 가 base 모델 반환
        2. build_mu_from_policy_yaml(mode='shared') 가 unbound SharedHFMuSampler 반환
        3. build_grpo_trainer() 가 PEFT wrap한 모델을 trainer.model에 저장
        4. mu.bind(trainer.model) 로 PEFT 모델 주입 (학습 직전)
        5. 학습 중 μ 호출 시 disable_adapter() 컨텍스트로 base만 사용

    장단점 vs MuSampler(vLLM):
        + GPU 메모리 절감 ~9 GB (7B 기준) — vLLM duplicate copy 제거
        − HF.generate가 vLLM 대비 ~K× 느림 (prefix caching 없음, K-sampling 비효율)
    """

    def __init__(self, cfg: MuConfig, tokenizer=None):
        self.cfg = cfg
        self._tokenizer = tokenizer
        self._model = None     # bind()으로 주입됨

    def bind(self, model, tokenizer=None) -> None:
        """trainer의 PEFT 모델 (또는 임의의 disable_adapter() 지원 model) 연결."""
        self._model = model
        if tokenizer is not None:
            self._tokenizer = tokenizer

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "SharedHFMuSampler가 모델에 bind되지 않았습니다. "
                "build_grpo_trainer() 후 mu.bind(trainer.model) 호출 필요."
            )
        if self._tokenizer is None:
            raise RuntimeError("SharedHFMuSampler에 tokenizer가 없습니다.")

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build_prompt(problem: str, prefix: str) -> str:
        return (
            "<|im_start|>system\nYou solve math step by step. Number each step on its own line.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"
        )

    @staticmethod
    def _unwrap(model):
        """accelerate/DeepSpeed wrapper 제거 — disable_adapter()는 PeftModel에 정의."""
        while hasattr(model, "module"):
            model = model.module
        return model

    def _trim_at_stop(self, text: str) -> str:
        text = text.strip()
        cut = len(text)
        for stop in self.cfg.step_stop:
            idx = text.find(stop)
            if 0 <= idx < cut:
                cut = idx
        return text[:cut]

    # ------------------------------------------------------------------ api
    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        """K=n개 alternative single-step — trainer 모델의 disable_adapter() 컨텍스트."""
        import torch

        self._ensure_loaded()
        prompt = self._build_prompt(problem, prefix)
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)

        peft_model = self._unwrap(self._model)
        # peft_model.disable_adapter() 컨텍스트 내에서 base만 사용
        # stop_strings — step boundary에서 sequence-별로 조기 종료 (max_new_tokens 낭비 방지)
        with torch.no_grad(), peft_model.disable_adapter():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.pad_token_id,
                stop_strings=list(self.cfg.step_stop),
                tokenizer=self._tokenizer,
            )

        # 생성된 부분만 디코드 + step boundary에서 자르기
        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        texts = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [self._trim_at_stop(t) for t in texts]


def build_mu_from_policy_yaml(path: str | Path):
    """policy.yaml의 mu: 섹션에서 μ 샘플러 인스턴스 생성.

    mu.mode:
        "vllm"   (default) → 같은 GPU에 별도 vLLM 인스턴스 (MuSampler)
        "shared"           → trainer 모델 재활용 (SharedHFMuSampler, 1-copy 공유)
                             build_grpo_trainer() 후 mu.bind(trainer.model)을 호출해야 함.
        "remote"           → 다른 PC의 vLLM OpenAI API 호출 (RemoteMuSampler)
                             서버: `python -m vllm.entrypoints.openai.api_server`
                             환경변수 MU_ENDPOINT로 yaml 값 override 가능.

    예시 (remote):
        mu:
          mode: remote
          model_id: Qwen/Qwen2.5-Math-7B-Instruct
          remote:
            endpoint: http://192.168.1.10:8001
            timeout: 180
    """
    import os

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mu_cfg = data.get("mu", {})
    mode = (mu_cfg.get("mode") or "vllm").lower()

    if mode == "remote":
        from .remote_mu import RemoteMuConfig, RemoteMuSampler

        remote = dict(mu_cfg.get("remote") or {})
        env_endpoint = os.environ.get("MU_ENDPOINT")
        if env_endpoint:
            remote["endpoint"] = env_endpoint
        env_replicas = os.environ.get("MU_REPLICAS")
        replicas = int(env_replicas) if env_replicas else int(remote.get("num_replicas", 1))
        env_frps = os.environ.get("FRPS_DASHBOARD_URL")
        frps_url = env_frps or remote.get("frps_dashboard_url", "http://frps:7500")
        return RemoteMuSampler(
            RemoteMuConfig(
                endpoint=remote.get("endpoint", "http://localhost:8001"),
                model_id=mu_cfg.get("model_id", "Qwen/Qwen2.5-Math-7B-Instruct"),
                timeout=float(remote.get("timeout", 180.0)),
                temperature=mu_cfg.get("temperature", 1.0),
                top_p=mu_cfg.get("top_p", 0.95),
                max_new_tokens=mu_cfg.get("max_new_tokens", 256),
                step_stop=tuple(mu_cfg.get("step_stop", ["\n\n"])),
                num_replicas=replicas,
                frps_dashboard_url=frps_url,
            )
        )

    base_cfg = MuConfig(
        model_id=mu_cfg.get("model_id", "shared"),
        temperature=mu_cfg.get("temperature", 1.0),
        top_p=mu_cfg.get("top_p", 0.95),
        max_new_tokens=mu_cfg.get("max_new_tokens", 256),
        step_stop=tuple(mu_cfg.get("step_stop", ["\n\n"])),
        gpu_memory_utilization=mu_cfg.get("gpu_memory_utilization", 0.30),
        max_model_len=mu_cfg.get("max_model_len", 4096),
        dtype=mu_cfg.get("dtype", "bfloat16"),
        cpu_offload_gb=float(mu_cfg.get("cpu_offload_gb", 0) or 0),
    )

    if mode == "shared":
        return SharedHFMuSampler(base_cfg)

    return MuSampler(base_cfg)
