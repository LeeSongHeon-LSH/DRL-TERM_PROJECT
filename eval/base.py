from abc import ABC, abstractmethod


class BaseBenchmark(ABC):
    """
    Abstract benchmark interface.
    TBD: Concrete benchmarks to be plugged in once confirmed.
    """

    @abstractmethod
    def load(self) -> None:
        """Load benchmark data."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, responses: list[str], references: list[str]) -> dict:
        """
        Score model responses against references (or gold labels).
        Returns: dict of metric_name → score
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
