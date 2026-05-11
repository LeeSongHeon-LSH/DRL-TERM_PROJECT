"""정책 모델 + LoRA + 데이터셋 빌더.

학습:    GSM8K (openai/gsm8k, "main" config) — 단일.
검증/평가: MathNet (ShadenA/MathNet) — English + text-only + final_answer 있는 것만.

GRPOTrainer가 기대하는 {prompt, answer} 포맷으로 변환.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- policy + LoRA
def build_policy(policy_yaml: str | Path, *, gradient_checkpointing: bool = True):
    """AutoModelForCausalLM + AutoTokenizer (padding_side='left') 로드.

    LoraConfig는 별도 반환 — TRL GRPOTrainer가 peft_config로 wrap.
    """
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _load_yaml(policy_yaml)

    dtype_str = cfg.get("dtype", "bfloat16")
    import torch
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules", "all-linear")
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return model, tokenizer, peft_config


# ---------------------------------------------------------------- train
def build_train_dataset(rl_yaml: str | Path, tokenizer):
    """rl_q3.yaml의 data.train 항목을 합쳐 {prompt, answer} HF Dataset 반환.

    현재 지원:
      - gsm8k  (openai/gsm8k, "main" config) — main loader
      - hendrycks_math (DigitalLearningGmbH/MATH-lighteval) — legacy 호환용

    GRPOTrainer는 dataset의 추가 컬럼을 reward_func에 kwargs로 전달.
    "answer" 컬럼은 정답 문자열 (math_verify로 매칭).
    """
    from datasets import concatenate_datasets, load_dataset

    rl_cfg = _load_yaml(rl_yaml)
    data_cfg = rl_cfg.get("data", {}).get("train", [])
    if not data_cfg:
        raise ValueError(f"{rl_yaml}의 data.train이 비었습니다.")

    parts = []
    for entry in data_cfg:
        ds_name = entry["dataset"]
        split = entry.get("split", "train")
        if ds_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split=split)
            ds = ds.map(_format_gsm8k, remove_columns=ds.column_names)
        elif ds_name in ("hendrycks_math", "math", "lighteval/math"):
            ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
            ds = ds.map(_format_math, remove_columns=ds.column_names)
        else:
            raise ValueError(f"Unknown training dataset: {ds_name}")
        parts.append(ds)

    full = concatenate_datasets(parts).shuffle(seed=42)
    full = full.map(_make_chat_wrapper(tokenizer))
    return full


# ---------------------------------------------------------------- eval (MathNet)
def build_eval_dataset(rl_yaml: str | Path, tokenizer=None):
    """rl_q3.yaml의 data.eval 항목 → 평가용 {prompt, answer} Dataset.

    현재 지원:
      - mathnet (ShadenA/MathNet)
      - gsm8k   (검증 fallback)
      - hendrycks_math
    """
    from datasets import concatenate_datasets

    rl_cfg = _load_yaml(rl_yaml)
    eval_cfg = rl_cfg.get("data", {}).get("eval", [])
    if not eval_cfg:
        raise ValueError(f"{rl_yaml}의 data.eval이 비었습니다.")

    parts = []
    for entry in eval_cfg:
        ds_name = entry["dataset"]
        if ds_name == "mathnet":
            parts.append(_load_mathnet(entry))
        elif ds_name == "gsm8k":
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split=entry.get("split", "test"))
            ds = ds.map(_format_gsm8k, remove_columns=ds.column_names)
            n = entry.get("subset")
            if n:
                ds = ds.select(range(min(n, len(ds))))
            parts.append(ds)
        elif ds_name in ("hendrycks_math", "math"):
            from datasets import load_dataset
            ds = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval", split=entry.get("split", "test")
            )
            ds = ds.map(_format_math, remove_columns=ds.column_names)
            n = entry.get("subset")
            if n:
                ds = ds.select(range(min(n, len(ds))))
            parts.append(ds)
        else:
            raise ValueError(f"Unknown eval dataset: {ds_name}")

    full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if tokenizer is not None:
        full = full.map(_make_chat_wrapper(tokenizer))
    return full


def _load_mathnet(entry: dict[str, Any]):
    """ShadenA/MathNet 로드 + 필터 + subset.

    필터 옵션 (entry["filters"]):
      - language: "English"
      - text_only: True       (len(images) == 0)
      - require_final_answer: True   (final_answer != None and != "")
      - exclude_proof_only: True (problem_type != "proof only")
    """
    from datasets import load_dataset

    ds = load_dataset("ShadenA/MathNet", split=entry.get("split", "train"))
    filters = entry.get("filters", {}) or {}

    if filters.get("language"):
        target_lang = filters["language"]
        ds = ds.filter(lambda x: x.get("language") == target_lang)
    if filters.get("text_only"):
        ds = ds.filter(lambda x: not x.get("images") or len(x["images"]) == 0)
    if filters.get("require_final_answer"):
        ds = ds.filter(lambda x: bool((x.get("final_answer") or "").strip()))
    if filters.get("exclude_proof_only"):
        ds = ds.filter(lambda x: x.get("problem_type") != "proof only")

    n = entry.get("subset")
    if n:
        ds = ds.select(range(min(n, len(ds))))

    ds = ds.map(_format_mathnet, remove_columns=ds.column_names)
    log.info(f"MathNet loaded: {len(ds)} problems (after filters / subset)")
    return ds


# ---------------------------------------------------------------- formatters
def _format_math(ex: dict[str, Any]) -> dict[str, str]:
    sol = ex.get("solution", "")
    return {"prompt": ex["problem"], "answer": _extract_boxed(sol) or sol}


_GSM_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def _format_gsm8k(ex: dict[str, Any]) -> dict[str, str]:
    raw = ex["answer"]
    m = _GSM_ANSWER_RE.search(raw)
    return {"prompt": ex["question"], "answer": (m.group(1).strip() if m else raw.strip())}


def _format_mathnet(ex: dict[str, Any]) -> dict[str, str]:
    """MathNet markdown problem → {prompt, answer}."""
    return {
        "prompt": (ex.get("problem_markdown") or "").strip(),
        "answer": (ex.get("final_answer") or "").strip(),
    }


def _extract_boxed(s: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]+)\}", s)
    return m[-1].strip() if m else None


# ---------------------------------------------------------------- chat wrap
def _make_chat_wrapper(tokenizer):
    def _wrap(ex):
        msg = [
            {"role": "system",
             "content": "You solve math step by step. Number each step on its own line."},
            {"role": "user", "content": ex["prompt"]},
        ]
        ex["prompt"] = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        return ex

    return _wrap


# ---------------------------------------------------------------- helpers
def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
