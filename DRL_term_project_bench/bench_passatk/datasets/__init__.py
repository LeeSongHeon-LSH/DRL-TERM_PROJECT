"""Dataset loaders for bench_passatk."""

from typing import Dict, List, Tuple
from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """Base class for dataset loaders."""
    
    @abstractmethod
    def load(self) -> List[Dict]:
        """
        Load the dataset.
        
        Returns:
            List of problem dictionaries with 'id', 'problem', 'gold', 'level' (optional).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass


def get_dataset_loader(dataset_name: str) -> DatasetLoader:
    """
    Get a dataset loader by name.
    
    Args:
        dataset_name: Name of the dataset ('MATH', 'AIME2023', 'AIME2024', 'OlympiadBench', 'test').
        
    Returns:
        DatasetLoader instance.
    """
    from .math import MATHLoader
    from .aime import AIMELoader
    from .olympiad import OlympiadLoader
    from .test import TestLoader
    
    loaders = {
        "MATH": MATHLoader,
        "AIME2023": AIMELoader,
        "AIME2024": AIMELoader,
        "AIME2025": AIMELoader,
        "AIME2026": AIMELoader,
        "OlympiadBench": OlympiadLoader,
        "test": TestLoader,
    }
    
    # Handle AIME with year
    if dataset_name.startswith("AIME"):
        return AIMELoader(year=dataset_name[4:] if len(dataset_name) > 4 else None)
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name]()


__all__ = [
    "DatasetLoader",
    "get_dataset_loader",
]