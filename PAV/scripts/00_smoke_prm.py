"""W1 Day 1 — 천공 PRM smoke test.

PRM(default: Skywork-o1-Open-PRM-Qwen-2.5-1.5B)을 로드하고
toy 문제 + 정답/오답 prefix에 대해 점수가 분리되는지 확인.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prm import load_prm


TOY_PROBLEM = "What is 7 + 5?"
TOY_PREFIXES = [
    ("correct", "Step 1: 7 + 5 = 12.\n\n"),
    ("partial", "Step 1: Let me add these numbers.\n\n"),
    ("wrong",   "Step 1: 7 + 5 = 13.\n\n"),
    ("filler",  "Step 1: Let me think carefully about this.\n\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prm.yaml")
    args = ap.parse_args()

    prm = load_prm(args.config)
    print(f"Loaded PRM: {prm.cfg.name} ({prm.cfg.model_id}) [{prm.cfg.quantization}]")
    print()
    print(f"Problem: {TOY_PROBLEM}")
    print(f"{'label':10s}  {'score':>8s}   prefix")
    for label, prefix in TOY_PREFIXES:
        s = float(prm.score(TOY_PROBLEM, prefix))
        print(f"{label:10s}  {s:8.4f}   {prefix.strip()!r}")

    # batch sanity
    batch_scores = prm.score_batch(TOY_PROBLEM, [p for _, p in TOY_PREFIXES])
    print()
    print("score_batch:", batch_scores.tolist())


if __name__ == "__main__":
    main()
