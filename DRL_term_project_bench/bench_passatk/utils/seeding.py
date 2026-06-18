"""Seeding utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Sets seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variable for vLLM
    
    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for vLLM
    os.environ["VLLM_SEED"] = str(seed)


def get_problem_seed(base_seed: int, problem_idx: int) -> int:
    """
    Generate a deterministic seed for a specific problem.
    
    This ensures that each problem gets a unique but reproducible seed,
    while maintaining overall reproducibility.
    
    Args:
        base_seed: The base random seed.
        problem_idx: The index of the problem in the dataset.
        
    Returns:
        A deterministic seed for the specific problem.
    """
    # Use a simple hash-like combination that's deterministic
    # and unlikely to collide for reasonable problem counts
    return base_seed + problem_idx * 1000