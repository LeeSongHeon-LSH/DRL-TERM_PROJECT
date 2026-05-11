"""Vendored from https://github.com/SkyworkAI/skywork-o1-prm-inference (Apache 2.0).

Skywork-o1-Open-PRM-Qwen-2.5-{1.5B, 7B} 추론에 필요한 PRM_MODEL + io 유틸을 PAV-RL 안에 동봉.
원본을 수정하지 않고 그대로 가져와 추후 upstream 변경에 맞춰 갱신하기 쉽게 분리.
"""
from .prm_model import PRM_MODEL
from .io_utils import (
    derive_step_rewards,
    derive_step_rewards_vllm,
    prepare_batch_input_for_model,
    prepare_input,
)

__all__ = [
    "PRM_MODEL",
    "prepare_input",
    "prepare_batch_input_for_model",
    "derive_step_rewards",
    "derive_step_rewards_vllm",
]
