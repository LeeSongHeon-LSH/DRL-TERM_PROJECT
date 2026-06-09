"""
Qwen2.5-1.5B-Instruct loader and sampling-based inference for pass@k evaluation.

Backend: vLLM (continuous batching) — generates all N completions for a prompt
in a single request via SamplingParams(n=N), which is dramatically faster than
sequential HuggingFace `generate()` sub-batching.

Key design:
  - load_model_and_tokenizer() builds a vLLM engine + returns its tokenizer
  - generate_samples() produces N completions for a single prompt in one call
"""

from typing import List

from vllm import LLM, SamplingParams

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
    """Build a vLLM engine and return (llm, tokenizer)."""
    print(f"Loading model with vLLM: {config.model_name}")

    llm = LLM(
        model=config.model_name,
        dtype=config.dtype,                 # "bfloat16" / "float16" / "float32"
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
        seed=config.seed,
    )

    tokenizer = llm.get_tokenizer()
    print(f"  vLLM engine ready  |  dtype: {config.dtype}  "
          f"|  gpu_mem_util: {config.gpu_memory_utilization}  "
          f"|  max_model_len: {config.max_model_len}")
    return llm, tokenizer


def _sampling_params(config: EvalConfig, n: int) -> SamplingParams:
    if config.do_sample and config.temperature > 0:
        return SamplingParams(
            n=n,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
        )
    # Greedy decoding
    return SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=config.max_new_tokens,
    )


def warmup(model) -> None:
    """Trigger engine/kernel initialisation with a tiny request."""
    model.generate(
        ["warmup"],
        SamplingParams(n=1, temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )


def generate_greedy(model, prompt: str, config: EvalConfig) -> str:
    """Single greedy (temperature=0) decoding pass for exact pass@1 accuracy."""
    params = SamplingParams(n=1, temperature=0.0, max_tokens=config.max_new_tokens)
    outputs = model.generate([prompt], params, use_tqdm=False)
    return outputs[0].outputs[0].text


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    config: EvalConfig,
) -> List[str]:
    """
    Generate config.num_samples completions for a single prompt.

    vLLM batches all N samples internally via SamplingParams(n=N), so this is a
    single engine call. Returns a flat list of decoded completion strings.
    """
    sampling_params = _sampling_params(config, n=config.num_samples)
    outputs = model.generate([prompt], sampling_params, use_tqdm=False)
    return [completion.text for completion in outputs[0].outputs]
