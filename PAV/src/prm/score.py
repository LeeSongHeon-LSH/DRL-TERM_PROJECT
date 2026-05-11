"""PRM scoring wrapper — Skywork-o1-Open-PRM-Qwen-2.5-{1.5B, 7B} 호환.

천공 PRM은 TRL의 ValueHead 패턴을 사용 (Linear(hidden, 1) head per token).
- step boundary: 단일 newline (\n)
- prepare_input: bos + problem + "\n" + (각 step + step_token), reward_flag=1 at last token of each step
- forward(return_probs=True) → (lm_logits, loss, sigmoid(value)) — value는 [B, T]

본 wrapper는 PAV가 호출하는 단순 인터페이스:
    score(problem, solution_prefix)        → 0-d tensor (마지막 step의 sigmoid 보상)
    score_batch(problem, [prefix1, ...])   → tensor [N]   (각 prefix의 마지막 step 보상)
    score_per_step(problem, solution)      → list[float] (모든 step 보상 — sanity 시각화용)

solution_prefix가 \n\n로 split된 step들을 \n로 정규화하므로, 정책이 \n\n 출력해도 호환.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

import torch

from .skywork import (
    PRM_MODEL,
    derive_step_rewards,
    prepare_batch_input_for_model,
    prepare_input,
)

if TYPE_CHECKING:
    from .loader import PRMConfig


# 정책이 출력하는 \n\n 구분 step → Skywork PRM의 \n 구분 형식으로 정규화
_DOUBLE_NL = re.compile(r"\n\s*\n+")


def _normalize_for_prm(text: str) -> str:
    """모든 multi-newline 구간을 단일 \n으로 압축. trailing \n 보장."""
    if not text:
        return ""
    s = _DOUBLE_NL.sub("\n", text).strip()
    return s + "\n"


class PRM:
    """Skywork PRM의 step-level scorer."""

    def __init__(self, cfg: "PRMConfig"):
        self.cfg = cfg
        self._model: PRM_MODEL | None = None
        self._tokenizer = None
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoConfig, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        load_kwargs: dict = {"trust_remote_code": True}
        if self.cfg.quantization == "awq":
            # AWQ 양자화 가중치 — torch_dtype은 fp16 권장 (Skywork README와 동일)
            load_kwargs["torch_dtype"] = torch.float16
        else:
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.cfg.dtype]
            load_kwargs["torch_dtype"] = dtype

        # device_map="auto" — accelerate가 layer-wise sharding (multi-GPU 자동 처리)
        load_kwargs["device_map"] = "auto"

        # 1.5B (Qwen2ForPrmModel / Qwen2ForRewardModel)와 7B (Qwen2ForCausalLM + ValueHead) 분기.
        # 1.5B는 reward head 내장 → AutoModel로 직접 로드 후 SkyworkRMWrapper로 wrap.
        # 7B는 외부 ValueHead 패턴 → PRM_MODEL wrapper 사용.
        cfg = AutoConfig.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        archs = list(getattr(cfg, "architectures", None) or [])
        is_internal_rm = any(("PrmModel" in a) or ("RewardModel" in a) for a in archs)

        if is_internal_rm:
            from transformers import AutoModel

            from .skywork_rm import SkyworkRMWrapper

            base = AutoModel.from_pretrained(self.cfg.model_id, **load_kwargs)
            self._model = SkyworkRMWrapper(base).eval()
        else:
            self._model = PRM_MODEL.from_pretrained(
                self.cfg.model_id, **load_kwargs
            ).eval()

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def score(self, problem: str, solution_prefix: str) -> torch.Tensor:
        """단일 (problem, prefix) → 0-d tensor (sigmoid 확률, [0, 1]).

        prefix가 비어 있으면 0.5 반환 (uninformative prior — score_batch 호환).
        """
        if not solution_prefix.strip():
            return torch.tensor(0.5)
        return self.score_batch(problem, [solution_prefix])[0]

    @torch.no_grad()
    def score_batch(
        self,
        problem: str,
        solution_prefixes: Sequence[str],
    ) -> torch.Tensor:
        """N개 prefix를 같은 problem에 대해 batch 채점 → tensor [N].

        반환값은 각 prefix의 *마지막 step* 위치의 sigmoid 확률.
        """
        self._ensure_loaded()

        # mini-batch — OOM 방지
        scores: list[float] = []
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(solution_prefixes), bs):
            chunk = list(solution_prefixes[i : i + bs])
            chunk_scores = self._score_chunk(problem, chunk)
            scores.extend(chunk_scores)
        return torch.tensor(scores, dtype=torch.float32)

    @torch.no_grad()
    def score_per_step(
        self,
        problem: str,
        solution: str,
    ) -> list[float]:
        """전체 solution의 step별 sigmoid 보상 리스트 (sanity 시각화 / Phase 0 visualization)."""
        self._ensure_loaded()
        normalized = _normalize_for_prm(solution)
        if not normalized.strip():
            return []
        input_ids, _steps, reward_flags = prepare_input(
            problem, normalized, self._tokenizer, self.cfg.step_token
        )
        padded_ids, padded_attn, padded_flags = prepare_batch_input_for_model(
            [input_ids], [reward_flags], self._tokenizer.pad_token_id
        )
        padded_ids = padded_ids.to(self._device)
        padded_attn = padded_attn.to(self._device)
        _, _, rewards = self._model(
            input_ids=padded_ids, attention_mask=padded_attn, return_probs=True
        )  # rewards: [1, T]
        per_step = derive_step_rewards(rewards.cpu(), padded_flags)
        return per_step[0]

    # ------------------------------------------------------------------ internals
    def _score_chunk(self, problem: str, prefixes: list[str]) -> list[float]:
        """한 batch의 마지막-step 보상만 모아 반환."""
        normalized = [_normalize_for_prm(p) for p in prefixes]
        # 빈 prefix는 prior 0.5로 — 위치를 보존하기 위해 마스크 처리
        nonempty_idx = [i for i, n in enumerate(normalized) if n.strip()]
        if not nonempty_idx:
            return [0.5] * len(prefixes)

        prepared = [
            prepare_input(problem, normalized[i], self._tokenizer, self.cfg.step_token)
            for i in nonempty_idx
        ]
        input_ids = [p[0] for p in prepared]
        reward_flags = [p[2] for p in prepared]

        padded_ids, padded_attn, padded_flags = prepare_batch_input_for_model(
            input_ids, reward_flags, self._tokenizer.pad_token_id
        )
        padded_ids = padded_ids.to(self._device)
        padded_attn = padded_attn.to(self._device)

        _, _, rewards = self._model(
            input_ids=padded_ids, attention_mask=padded_attn, return_probs=True
        )  # rewards: [B, T]
        per_step = derive_step_rewards(rewards.cpu(), padded_flags)  # list[list[float]]

        # 각 sequence의 *마지막* step 보상만 채택
        last_per_seq = [r[-1] if r else 0.5 for r in per_step]

        # 빈 prefix 위치는 0.5로 채워 원래 순서 복원
        out = [0.5] * len(prefixes)
        for j, idx in enumerate(nonempty_idx):
            out[idx] = last_per_seq[j]
        return out
