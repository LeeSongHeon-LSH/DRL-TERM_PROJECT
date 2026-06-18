"""Sampler backends for bench_passatk."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for model samplers."""
    
    @abstractmethod
    def __init__(self, model_path: str, **kwargs):
        """Initialize the sampler with a model path."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate samples for each prompt.
        
        Args:
            prompts: List of prompt strings.
            n: Number of samples per prompt.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            List of lists of samples, one list per prompt.
            Each sample is a dict with 'text' and 'tokens' keys.
        """
        pass
    
    @abstractmethod
    def get_tokenizer(self):
        """Return the tokenizer."""
        pass
    
    @abstractmethod
    def apply_chat_template(
        self,
        problem: str,
        system_message: str = "You are a careful mathematical reasoner. Think step by step and put the final answer in \\boxed{}.",
    ) -> str:
        """
        Apply the model's chat template to a problem.
        
        Args:
            problem: The problem text.
            system_message: System message to prepend.
            
        Returns:
            Formatted prompt string.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources (free GPU memory)."""
        pass


def get_sampler(backend: str, model_path: str, **kwargs) -> BaseSampler:
    """
    Get a sampler by backend name.
    
    Args:
        backend: Backend name ('vllm' or 'hf').
        model_path: Path to the model checkpoint.
        **kwargs: Additional arguments passed to the sampler.
        
    Returns:
        BaseSampler instance.
    """
    if backend == "vllm":
        from .vllm_sampler import VLLMSampler
        return VLLMSampler(model_path, **kwargs)
    elif backend == "hf":
        from .hf_sampler import HFSampler
        # Filter out vLLM-specific arguments
        hf_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['tensor_parallel_size', 'gpu_memory_utilization', 
                                   'max_model_len', 'enforce_eager']}
        return HFSampler(model_path, **hf_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'hf'.")


__all__ = [
    "BaseSampler",
    "get_sampler",
]