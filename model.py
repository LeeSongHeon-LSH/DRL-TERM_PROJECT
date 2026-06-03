"""
Qwen2.5-1.5B-Instruct loader and batch inference.
Optimised for RTX 5060 (Blackwell, sm_120):
  - bfloat16  → native Blackwell precision
  - SDPA      → PyTorch built-in scaled-dot-product attention (safe on all CUDA GPUs)
  - left-pad  → correct batch generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from config import EvalConfig


SYSTEM_PROMPT = """\
You are an expert competition mathematician. Solve the following AIME problem carefully.

Rules:
- AIME answers are non-negative integers from 0 to 999.
- Work through the problem step by step, showing all reasoning.
- At the very end of your response write your final answer on its own line in the exact format:
  \\boxed{N}
where N is your integer answer (e.g. \\boxed{42})."""


def build_chat_prompt(problem: str, tokenizer) -> str:
    """Format a single problem as a Qwen2.5-Instruct chat prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(config: EvalConfig):
    """Load Qwen2.5-1.5B-Instruct with settings tuned for the RTX 5060."""
    print(f"Loading model: {config.model_name}")

    dtype = getattr(torch, config.dtype)

    # SDPA is always available in PyTorch 2.x and works on Blackwell without
    # needing a compiled flash-attention wheel.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Left-pad so that all sequences in a batch end at the same position
    # (required for correct greedy / sampling generation on padded batches)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    print(f"  Model on device: {device}  |  dtype: {dtype}")
    print(f"  VRAM allocated:  {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    config: EvalConfig,
) -> List[str]:
    """
    Run inference on a list of prompts in mini-batches.
    Returns decoded new-token strings (input tokens stripped).
    """
    all_outputs: List[str] = []

    for start in range(0, len(prompts), config.batch_size):
        batch = prompts[start : start + config.batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,       # hard cap on input length
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if config.do_sample and config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        # Strip the echoed input tokens — only decode newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)

    return all_outputs
