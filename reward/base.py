from abc import ABC, abstractmethod
import torch


class BaseReward(ABC):
    """Abstract interface for reward computation."""

    @abstractmethod
    def compute(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """
        Compute scalar reward for each (prompt, response) pair.
        Returns: Tensor of shape (batch,)
        """
        raise NotImplementedError
