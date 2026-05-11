"""1.5B Skywork PRM (Qwen2ForRewardModel) wrapper — 7B PRM_MODEL과 동일 인터페이스.

7B (Skywork-o1-Open-PRM-Qwen-2.5-7B):
    - 표준 AutoModelForCausalLM + 외부 ValueHead (PRM_MODEL wrapper)
    - forward → (lm_logits, loss, value[B,T])

1.5B (Skywork-o1-Open-PRM-Qwen-2.5-1.5B):
    - 자체 Qwen2ForRewardModel (reward head v_head 내장)
    - 표준 forward는 pooled_logits만 반환 → per-step reward 추출 불가
    - 본 wrapper가 base model의 hidden_states를 v_head에 직접 통과시켜 [B,T] sigmoid 반환

score.py는 둘 모두 동일하게 `_, _, value = self._model(input_ids=..., return_probs=True)` 형태로 호출.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SkyworkRMWrapper(nn.Module):
    """Qwen2ForRewardModel의 .model + .v_head를 직접 호출하여 per-token reward 분포를 반환."""

    def __init__(self, model):
        super().__init__()
        self.model = model  # transformers의 Qwen2ForRewardModel 인스턴스 (auto_map 통해 trust_remote_code 로드)

    @property
    def is_skywork_rm(self) -> bool:
        return True

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_probs: bool = False,
        **kwargs,
    ):
        """7B PRM_MODEL.forward와 동일 시그니처로 (None, None, value[B,T]) 반환."""
        # 1) base 트랜스포머 forward — 마지막 hidden_states만 필요
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state  # [B, T, H]

        # 2) v_head는 내장된 reward head — 가중치는 from_pretrained가 이미 로드
        value = self.model.v_head(hidden).squeeze(-1)  # [B, T]

        if return_probs:
            value = torch.sigmoid(value)

        # 7B와 호환되는 (lm_logits, loss, value) 형태
        return None, None, value
