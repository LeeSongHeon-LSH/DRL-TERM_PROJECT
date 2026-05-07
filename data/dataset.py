from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract dataset interface. Subclass for each concrete dataset/benchmark."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with at minimum:
          - "prompt": str
        Optionally:
          - "reference": str  (ground-truth answer, if available)
          - "metadata": dict  (task-specific info)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseDataset":
        """Instantiate from config dict (loaded from base_config.yaml)."""
        raise NotImplementedError
