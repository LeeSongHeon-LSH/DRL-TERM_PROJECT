"""W1 Day 2~4 — Phase 0 DifferentialPAV smoke + S1~S4 sanity.

작은 라벨 데이터 (correct / wrong / filler)로 advantage 부호/크기를 검증.
실제 데이터는 별도 라벨링 단계에서 작성 — 여기서는 toy 라벨로 파이프라인 검증.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.sanity import SanityItem, run_sanity_checks
from src.pav import DifferentialPAV
from src.prm import load_prm


# 실제로는 MATH 데이터 + step 라벨링으로 채울 것.
TOY_ITEMS = [
    SanityItem(
        problem="Solve x^2 = 9.",
        prefix="",
        step="Step 1: x^2 = 9 implies x = ±3.\n\n",
        label="correct",
    ),
    SanityItem(
        problem="Solve x^2 = 9.",
        prefix="",
        step="Step 1: Let me think carefully about this.\n\n",
        label="filler",
    ),
    SanityItem(
        problem="Solve x^2 = 9.",
        prefix="",
        step="Step 1: x^2 = 9 implies x = 4.\n\n",
        label="wrong",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prm-config", default="configs/prm.yaml")
    ap.add_argument("--items-jsonl", default=None,
                    help="라벨 데이터 jsonl: {problem, prefix, step, label} (없으면 toy)")
    args = ap.parse_args()

    items = TOY_ITEMS if args.items_jsonl is None else _load_jsonl(args.items_jsonl)

    prm = load_prm(args.prm_config)
    pav = DifferentialPAV(prm)
    print(f"PAV method: {pav.name}")

    res = run_sanity_checks(pav, items)
    print(f"\n=== Sanity (Phase 0) ===")
    print(f"  S1 correct A>0   : {res.s1_correct_pos_rate:.2%} (n={res.n_correct})  [≥70% pass]")
    print(f"  S2 filler |A|<eps: {res.s2_filler_small_rate:.2%} (n={res.n_filler})  [≥60% pass]")
    print(f"  S3 wrong A<0     : {res.s3_wrong_neg_rate:.2%} (n={res.n_wrong})  [≥60% pass]")
    print(f"  S4 p_v multimodal: {res.s4_pv_multimodal}  [True 필요 — 실패 시 Phase 1 전환]")
    print(f"  PASS_ALL         : {res.pass_all}")


def _load_jsonl(path: str) -> list[SanityItem]:
    import json
    out: list[SanityItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(
                SanityItem(
                    problem=d["problem"],
                    prefix=d.get("prefix", ""),
                    step=d["step"],
                    label=d["label"],
                )
            )
    return out


if __name__ == "__main__":
    main()
