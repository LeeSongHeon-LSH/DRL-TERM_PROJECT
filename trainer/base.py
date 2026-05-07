from abc import ABC, abstractmethod
from models.policy import PolicyModel
from reward.base import BaseReward


class BaseTrainer(ABC):
    """Abstract trainer interface."""

    def __init__(self, policy: PolicyModel, reward_fn: BaseReward, config: dict):
        self.policy = policy
        self.reward_fn = reward_fn
        self.config = config

    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        """
        Run a single training step on a batch.
        Returns: dict of logged metrics (loss, reward, kl, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, dataloader) -> None:
        """Full training loop."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError
