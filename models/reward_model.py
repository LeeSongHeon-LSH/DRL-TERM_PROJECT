import torch
from .base import BaseLLM


class RewardModel(BaseLLM):
    """
    Judge LLM — evaluates Policy outputs and provides reward signals.
    TBD: Concrete judge model to be plugged in here.

    The judge's output probability distribution over tokens (e.g. "good"/"bad",
    or a scoring token) is used as the reward signal for PPO.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._device = None

    def load(self, model_name: str, dtype: torch.dtype = torch.bfloat16, **kwargs) -> None:
        # TODO: load judge LLM
        raise NotImplementedError(f"RewardModel.load() — model '{model_name}' not configured yet.")

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        # TODO: judge generation (may not always be needed — reward comes from logits)
        raise NotImplementedError

    def get_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def get_token_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def get_reward_logits(self, judge_prompts: list[str]) -> torch.Tensor:
        """
        Return raw logits from the judge over the vocabulary.
        Used by LLMJudgeReward to extract probability-based reward.
        Shape: (batch, vocab_size)
        """
        # TODO: forward pass through judge, return last-token logits
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self._device
