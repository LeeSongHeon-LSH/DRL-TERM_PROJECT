"""
Qwen2.5-1.5B-Instruct loader and sampling-based inference for pass@k evaluation.

Key design:
  - generate_samples() produces N completions for a single prompt
  - sub-batching via num_return_sequences keeps VRAM usage bounded
  - bfloat16 + SDPA for Blackwell (RTX 50xx) compatibility
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
    print(f"Loading model: {config.model_name}")
    dtype = getattr(torch, config.dtype)

    # Try flash_attention_2 first (faster on Ampere+); fall back to sdpa for Blackwell
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("  Using flash_attention_2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        print("  Using sdpa (flash_attention_2 not available)")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    vram_used = torch.cuda.memory_allocated(device) / 1e9
    vram_total = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"  Model on device: {device}  |  dtype: {dtype}")
    print(f"  VRAM: {vram_used:.2f} GB used / {vram_total:.1f} GB total")
    if vram_used > vram_total * 0.9:
        raise RuntimeError(
            f"Model uses {vram_used:.1f}/{vram_total:.1f} GB VRAM. "
            "Try --dtype float16 or a smaller model."
        )
    return model, tokenizer


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    config: EvalConfig,
) -> List[str]:
    """
    Generate config.num_samples completions for a single prompt.

    Splits the request into sub-batches of config.sample_batch_size
    (num_return_sequences per call) to keep VRAM usage bounded.
    Returns a flat list of decoded strings (input tokens stripped).
    """
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id":   tokenizer.pad_token_id,
        "eos_token_id":   tokenizer.eos_token_id,
    }
    if config.do_sample and config.temperature > 0:
        gen_kwargs["do_sample"]   = True
        gen_kwargs["temperature"] = config.temperature
    else:
        gen_kwargs["do_sample"] = False

    all_outputs: List[str] = []
    remaining = config.num_samples

    while remaining > 0:
        n = min(config.sample_batch_size, remaining)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                num_return_sequences=n,
                **gen_kwargs,
            )
        new_ids = generated_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)
        remaining -= n

    return all_outputs
