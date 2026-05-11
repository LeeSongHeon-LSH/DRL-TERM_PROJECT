"""W1 — sanity 라벨 데이터 자동 생성.

지정한 데이터셋 N문제에 대해:
  1) 정책 base 모델로 trajectory 1개 생성
  2) 각 step을 다음 규칙으로 자동 라벨:
        - filler: 짧거나 filler 패턴 (`Let me think`, `So`, `Now,` 등) 매칭
        - correct: 최종 답이 정답이면, 마지막 step만 'correct'
        - wrong: 최종 답이 틀리면 마지막 step만 'wrong'
        - 그 외 step은 라벨 안 함 — sanity 평가 대상에서 제외
  3) jsonl로 저장 → scripts/01_phase0_diff.py --items-jsonl 입력으로 사용

지원 dataset:
  - gsm8k    (default — 학습 데이터와 동일 분포라 정답/오답 sample이 균형있게 나옴)
  - mathnet  (Olympiad급 — 정답률 매우 낮음, S1 sample 부족 가능)
  - math500  (HuggingFaceH4/MATH-500 — 중간 난이도)

자동 라벨은 noise가 큼. G0 게이트의 신뢰성이 부족하면 LLM-judge / 수동 라벨 추가 권장.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rollout.parser import split_steps

log = logging.getLogger("label")

_FILLER_PATTERNS = [
    r"^let me think",
    r"^let'?s think",
    r"^okay,?\s",
    r"^so,?\s",
    r"^now,?\s",
    r"^well,?\s",
    r"^hmm",
    r"^first,?\s+(?!step)",
]
_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_GSM_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def is_filler(step: str) -> bool:
    s = step.strip()
    if len(s) < 30:
        return True
    return bool(_FILLER_RE.match(s))


def extract_answer(text: str) -> str:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    last = text.strip().splitlines()[-1] if text.strip() else ""
    return last.strip()


def is_correct(pred: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
        return bool(verify(parse(str(gold)), parse(str(pred))))
    except Exception:
        return str(gold).strip() in str(pred)


# ---------------------------------------------------------------- dataset adapters
def load_problems(name: str, n: int) -> list[dict[str, str]]:
    """{problem, gold} 리스트 반환."""
    from datasets import load_dataset

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for ex in ds.select(range(min(n, len(ds)))):
            m = _GSM_ANSWER_RE.search(ex["answer"])
            items.append({
                "problem": ex["question"],
                "gold": (m.group(1).strip() if m else ex["answer"].strip()),
            })
        return items

    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = []
        for ex in ds.select(range(min(n, len(ds)))):
            gold = ex.get("answer") or _extract_boxed(ex.get("solution", ""))
            items.append({"problem": ex["problem"], "gold": gold or ""})
        return items

    if name == "mathnet":
        ds = load_dataset("ShadenA/MathNet", split="train")
        # English + text-only + final_answer 있는 것만
        ds = ds.filter(lambda x: x.get("language") == "English")
        ds = ds.filter(lambda x: not x.get("images") or len(x["images"]) == 0)
        ds = ds.filter(lambda x: bool((x.get("final_answer") or "").strip()))
        ds = ds.select(range(min(n, len(ds))))
        return [
            {
                "problem": (ex.get("problem_markdown") or "").strip(),
                "gold": (ex.get("final_answer") or "").strip(),
            }
            for ex in ds
        ]

    raise ValueError(f"Unknown dataset: {name}")


def _extract_boxed(s: str) -> str | None:
    m = _BOXED_RE.findall(s or "")
    return m[-1].strip() if m else None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-config", default="configs/policy.yaml")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "mathnet"])
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--out", default="data/sanity_items.jsonl")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    items = load_problems(args.dataset, args.n_problems)
    log.info(f"Loaded {len(items)} problems from {args.dataset}")

    # ---- 정책 (base, no LoRA — 단일 generation)
    import yaml
    with open(args.policy_config, "r", encoding="utf-8") as f:
        pol_cfg = yaml.safe_load(f)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=pol_cfg["model_id"],
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        n=1,
    )

    def _prompt(problem: str) -> str:
        return (
            "<|im_start|>system\nYou solve math step by step. Number each step on its own line.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    prompts = [_prompt(it["problem"]) for it in items]
    outputs = llm.generate(prompts, sp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct = n_wrong = n_filler = 0
    with out_path.open("w", encoding="utf-8") as f:
        for it, out in zip(items, outputs):
            problem = it["problem"]
            gold = it["gold"]
            comp = out.outputs[0].text
            steps = split_steps(comp)
            if not steps:
                continue
            final_correct = is_correct(extract_answer(comp), gold)
            prefix_acc = ""
            for h, step in enumerate(steps):
                if is_filler(step):
                    label = "filler"
                    n_filler += 1
                elif h == len(steps) - 1:
                    label = "correct" if final_correct else "wrong"
                    if final_correct:
                        n_correct += 1
                    else:
                        n_wrong += 1
                else:
                    continue
                f.write(json.dumps({
                    "problem": problem,
                    "prefix": prefix_acc,
                    "step": step + "\n",
                    "label": label,
                }, ensure_ascii=False) + "\n")
                prefix_acc = prefix_acc + step + "\n"

    log.info(
        f"Wrote {out_path}: correct={n_correct}, wrong={n_wrong}, filler={n_filler}"
    )
    if n_correct < 30 or n_wrong < 30:
        log.warning(
            "S1/S3 평가에 sample이 부족할 수 있습니다 (각 ≥30 권장). "
            "dataset/n-problems/temperature 조정을 고려하세요."
        )


if __name__ == "__main__":
    main()
