"""W4 — MathNet 평가 (pass@1, pass@N).

학습된 LoRA 어댑터(또는 base 정책)를 MathNet eval subset에서 평가.
출력:
    - pass@1                  (greedy / temperature 0)
    - pass@N (sampling)       (계획서 §7-2의 pass@256 — Q3가 우세 기대)
    - 결과 jsonl 저장 (per-problem prediction + correctness)

사용 예:
    uv run python scripts/20_eval_mathnet.py \
        --rl-config configs/rl_q3.yaml \
        --policy-config configs/policy.yaml \
        --lora ./outputs/q3_lambda-0.5_K16/checkpoint-5000 \
        --N 64 \
        --out eval_q3_step5000.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.policy_data import build_eval_dataset

log = logging.getLogger("eval")


def is_correct(pred: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
        return bool(verify(parse(str(gold)), parse(str(pred))))
    except Exception:
        return str(gold).strip() in str(pred)


def extract_answer(text: str) -> str:
    import re
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    last = text.strip().splitlines()[-1] if text.strip() else ""
    return last.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-config", default="configs/rl_q3.yaml")
    ap.add_argument("--policy-config", default="configs/policy.yaml")
    ap.add_argument("--lora", default=None, help="LoRA 어댑터 경로 (없으면 base)")
    ap.add_argument("--N", type=int, default=64, help="pass@N의 N (sampling 횟수)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--out", default="data/eval_mathnet.jsonl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ---- eval dataset
    eval_ds = build_eval_dataset(args.rl_config, tokenizer=None)
    log.info(f"Eval dataset: {len(eval_ds)} problems")

    # ---- 정책 로드 (vLLM, optional LoRA)
    import yaml
    with open(args.policy_config, "r", encoding="utf-8") as f:
        pol_cfg = yaml.safe_load(f)

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=pol_cfg["model_id"],
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
    )
    if args.lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = pol_cfg.get("lora", {}).get("r", 64)
    llm = LLM(**llm_kwargs)
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("eval-lora", 1, args.lora)

    def _prompt(problem: str) -> str:
        return (
            "<|im_start|>system\nYou solve math step by step. Number each step on its own line. "
            "Put the final answer in \\boxed{}.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    prompts = [_prompt(ex["prompt"]) for ex in eval_ds]
    golds = [ex["answer"] for ex in eval_ds]

    # ---- pass@1 (greedy)
    log.info("\n=== pass@1 (greedy) ===")
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, n=1)
    gen_kwargs = {"lora_request": lora_request} if lora_request else {}
    out_greedy = llm.generate(prompts, sp_greedy, **gen_kwargs)

    pass1_correct = 0
    rows: list[dict] = []
    for ex, out in zip(eval_ds, out_greedy):
        completion = out.outputs[0].text
        pred = extract_answer(completion)
        ok = is_correct(pred, ex["answer"])
        pass1_correct += int(ok)
        rows.append({
            "problem": ex["prompt"][:200],
            "gold": ex["answer"],
            "pred_greedy": pred,
            "completion_greedy": completion,
            "pass1": ok,
        })
    pass1 = pass1_correct / len(eval_ds)
    log.info(f"  pass@1 = {pass1:.3%}  ({pass1_correct}/{len(eval_ds)})")

    # ---- pass@N (sampling)
    log.info(f"\n=== pass@{args.N} (T={args.temperature}, top_p={args.top_p}) ===")
    sp_sample = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.N,
    )
    out_sample = llm.generate(prompts, sp_sample, **gen_kwargs)

    passN_correct = 0
    for row, out in zip(rows, out_sample):
        any_correct = False
        for cand in out.outputs:
            if is_correct(extract_answer(cand.text), row["gold"]):
                any_correct = True
                break
        row[f"pass{args.N}"] = any_correct
        passN_correct += int(any_correct)
    passN = passN_correct / len(eval_ds)
    log.info(f"  pass@{args.N} = {passN:.3%}  ({passN_correct}/{len(eval_ds)})")

    # ---- 결과 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"\nWrote {out_path}")
    log.info(f"  pass@1 = {pass1:.3%}, pass@{args.N} = {passN:.3%}")


if __name__ == "__main__":
    main()
