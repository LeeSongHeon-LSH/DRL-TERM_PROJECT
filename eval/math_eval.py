"""
MathNet evaluator.

MathNet is interpreted as the MATH benchmark (Hendrycks et al., 2021).
HuggingFace dataset: hendrycks/competition_math
  Fields: problem, solution (contains \\boxed{answer}), level, type
"""

import logging
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

_MATH_DATASET = "hendrycks/competition_math"


def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip() if m else None


def _normalize(s: str) -> str:
    return s.replace(",", "").replace(" ", "").lower()


def _is_correct(pred: str | None, ref: str) -> bool:
    if pred is None:
        return False
    p, r = _normalize(pred), _normalize(ref)
    if p == r:
        return True
    try:
        return abs(float(p) - float(r)) < 1e-6
    except ValueError:
        return False


def _make_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Show your work step by step "
        "and put your final answer inside \\boxed{}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def evaluate(model_path: str, cfg: dict) -> dict:
    """
    Load model from model_path via vLLM and evaluate on the MATH benchmark.
    Returns overall accuracy and per-level accuracy.
    """
    eval_cfg = cfg["eval"]
    vllm_cfg = cfg["vllm"]

    dataset = load_dataset(_MATH_DATASET, split="test")
    n = eval_cfg["num_samples"]
    if 0 < n < len(dataset):
        dataset = dataset.select(range(n))

    llm = LLM(
        model=model_path,
        dtype=cfg["model"]["dtype"],
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.8),
        max_model_len=vllm_cfg["max_model_len"],
    )
    params = SamplingParams(
        max_tokens=eval_cfg["max_new_tokens"],
        temperature=eval_cfg["temperature"],
        n=1,
    )

    prompts = [_make_prompt(item["problem"]) for item in dataset]
    outputs = llm.generate(prompts, params)
    responses = [out.outputs[0].text for out in outputs]

    correct = 0
    by_level: dict[str, dict] = {}

    for item, response in zip(dataset, responses):
        level = item.get("level", "unknown")
        ref_ans = _extract_boxed(item["solution"]) or item["solution"].strip()
        pred_ans = _extract_boxed(response)
        ok = _is_correct(pred_ans, ref_ans)
        correct += ok
        by_level.setdefault(level, {"correct": 0, "total": 0})
        by_level[level]["correct"] += ok
        by_level[level]["total"] += 1

    results = {
        "overall_accuracy": correct / len(dataset),
        "by_level": {
            k: v["correct"] / v["total"] for k, v in by_level.items()
        },
    }
    logger.info(f"MATH accuracy: {results['overall_accuracy']:.4f}")
    for lvl, acc in sorted(results["by_level"].items()):
        logger.info(f"  {lvl}: {acc:.4f}")
    return results
