"""vLLM-based sampler for efficient batch generation."""

import gc
import os
from typing import List, Dict, Any, Optional

import torch

# Disable FlashInfer completely to avoid JIT compilation issues with CUDA version mismatch
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


class VLLMSampler:
    """Sampler using vLLM backend for efficient batch generation."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        **kwargs,
    ):
        """
        Initialize vLLM sampler.
        
        Args:
            model_path: Path to the model checkpoint directory.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: GPU memory utilization ratio.
            trust_remote_code: Whether to trust remote code.
            dtype: Data type for model weights.
            **kwargs: Additional arguments passed to LLM.
        """
        from vllm import LLM
        from transformers import AutoTokenizer
        
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Load model with vLLM
        # Use enforce_eager=True to avoid JIT compilation issues
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enforce_eager=True,
            **kwargs,
        )
    
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
        Generate samples using vLLM.
        
        This method efficiently generates n samples per prompt in a single batch,
        leveraging vLLM's continuous batching.
        
        Args:
            prompts: List of prompt strings.
            n: Number of samples per prompt.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            List of lists of samples, one list per prompt.
        """
        from vllm import SamplingParams
        
        # Create sampling params
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        
        # Generate all samples in one batch
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Organize results by prompt
        results = []
        for output in outputs:
            prompt_samples = []
            for sample in output.outputs:
                prompt_samples.append({
                    "text": sample.text,
                    "tokens": sample.token_ids if hasattr(sample, "token_ids") else None,
                    "finish_reason": sample.finish_reason,
                })
            results.append(prompt_samples)
        
        return results
    
    def generate_with_micro_batch(
        self,
        prompts: List[str],
        n: int = 256,
        micro_n: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        progress_callback=None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate samples with micro-batching for memory efficiency.
        
        When n is large (e.g., 256), generating all samples at once may cause OOM.
        This method splits the generation into micro-batches and aggregates results.
        
        Args:
            prompts: List of prompt strings.
            n: Total number of samples per prompt.
            micro_n: Number of samples per micro-batch.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
            seed: Random seed for reproducibility.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            List of lists of samples, one list per prompt.
        """
        from vllm import SamplingParams
        
        # Initialize results
        all_results = [[] for _ in prompts]
        
        # Calculate number of micro-batches
        n_batches = (n + micro_n - 1) // micro_n
        
        for batch_idx in range(n_batches):
            # Calculate samples for this batch
            start_n = batch_idx * micro_n
            end_n = min(start_n + micro_n, n)
            current_n = end_n - start_n
            
            # Use different seed for each batch to ensure diversity
            batch_seed = None
            if seed is not None:
                batch_seed = seed + batch_idx
            
            # Create sampling params
            sampling_params = SamplingParams(
                n=current_n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=batch_seed,
            )
            
            # Generate
            outputs = self.llm.generate(prompts, sampling_params)
            
            # Aggregate results
            for i, output in enumerate(outputs):
                for sample in output.outputs:
                    all_results[i].append({
                        "text": sample.text,
                        "tokens": sample.token_ids if hasattr(sample, "token_ids") else None,
                        "finish_reason": sample.finish_reason,
                    })
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_idx + 1, n_batches)
        
        return all_results
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer
    
    def apply_chat_template(
        self,
        problem: str,
        system_message: str = "You are a careful mathematical reasoner. Think step by step and put the final answer in \\boxed{}.",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Apply the model's chat template to a problem.
        
        Args:
            problem: The problem text.
            system_message: System message to prepend.
            few_shot_examples: Optional list of few-shot examples. Each example is a dict
                              with 'user' and 'assistant' keys.
            
        Returns:
            Formatted prompt string.
        """
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Add the actual problem
        messages.append({"role": "user", "content": problem})
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return prompt
    
    def cleanup(self):
        """Clean up resources (free GPU memory)."""
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()