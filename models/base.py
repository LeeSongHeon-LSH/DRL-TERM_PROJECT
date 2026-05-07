from abc import ABC, abstractmethod
from typing import Any
import torch


class BaseLLM(ABC):
    """Abstract interface for all LLMs in the pipeline."""

    @abstractmethod
    def load(self, model_name: str, **kwargs) -> None:
        """Load model and tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts."""
        raise NotImplementedError

    @abstractmethod
    def get_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """Return log probabilities of responses given prompts. Shape: (batch,)"""
        raise NotImplementedError

    @abstractmethod
    def get_token_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """Return per-token log probabilities. Shape: (batch, seq_len)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError
