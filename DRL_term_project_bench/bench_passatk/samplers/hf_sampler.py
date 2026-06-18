"""HuggingFace transformers-based sampler (fallback backend)."""

import gc
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFSampler:
    """Sampler using HuggingFace transformers backend."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """
        Initialize HuggingFace sampler.
        
        Args:
            model_path: Path to the model checkpoint directory.
            device: Device to load model on ('auto', 'cuda', 'cpu').
            torch_dtype: Data type for model weights.
            trust_remote_code: Whether to trust remote code.
            **kwargs: Additional arguments passed to AutoModelForCausalLM.
        """
        self.model_path = model_path
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.model.eval()
    
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
        Generate samples using HuggingFace transformers.
        
        Note: This is slower than vLLM for large n values.
        For efficiency, we generate n samples per prompt sequentially.
        
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
        if seed is not None:
            torch.manual_seed(seed)
        
        results = []
        
        for prompt in prompts:
            prompt_samples = []
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            for _ in range(n):
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                prompt_samples.append({
                    "text": generated_text,
                    "tokens": outputs[0].tolist(),
                    "finish_reason": "stop",
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
        Generate samples with micro-batching.
        
        For HF backend, this is equivalent to generate() since we already
        generate samples one at a time.
        
        Args:
            prompts: List of prompt strings.
            n: Total number of samples per prompt.
            micro_n: Ignored for HF backend.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
            seed: Random seed for reproducibility.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            List of lists of samples, one list per prompt.
        """
        return self.generate(
            prompts=prompts,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
    
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
        del self.model
        gc.collect()
        torch.cuda.empty_cache()