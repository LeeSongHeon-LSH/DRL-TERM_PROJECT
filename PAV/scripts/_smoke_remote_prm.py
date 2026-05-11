"""실제 PRM 워커가 띄워진 상태에서 RemotePRM으로 score RPC 호출 검증.

전제:
    1) docker compose --profile broker up -d
    2) 별도 터미널/백그라운드:
        uv run python scripts/serve_prm.py --config configs/prm.yaml \
            --amqp-url amqp://guest:guest@localhost:5672/

이 스크립트는 RemotePRM 클라이언트를 만들어 toy 문제로 score / score_batch / score_per_step
모두 호출. 워커가 실제 PRM 모델 forward를 돌리고 응답이 정상으로 매칭되는지 검증.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prm.remote_client import RemotePRM, RemotePRMConfig


TOY = ("What is 7 + 5?", [
    ("correct", "Step 1: 7 + 5 = 12.\n"),
    ("partial", "Step 1: Let me add these numbers.\n"),
    ("wrong",   "Step 1: 7 + 5 = 13.\n"),
    ("filler",  "Step 1: Let me think carefully about this.\n"),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amqp-url", default="amqp://guest:guest@localhost:5672/")
    ap.add_argument("--queue", default="prm.requests")
    ap.add_argument("--rpc-timeout", type=float, default=120.0)
    args = ap.parse_args()

    cli = RemotePRM(RemotePRMConfig(
        amqp_url=args.amqp_url, request_queue=args.queue, rpc_timeout=args.rpc_timeout,
    ))

    problem, prefixes = TOY
    print(f"problem: {problem}\n")

    # health
    print("[health]")
    print(f"  {cli.health()}\n")

    # score (단일)
    print("[score] (단일)")
    for label, prefix in prefixes:
        t0 = time.time()
        s = cli.score(problem, prefix)
        print(f"  {label:8s} {s.item():.4f}  ({(time.time()-t0)*1000:.1f}ms)  prefix={prefix.strip()!r}")
    print()

    # score_batch (한 RPC에 묶음)
    print("[score_batch] (한 RPC, 4개)")
    t0 = time.time()
    bs = cli.score_batch(problem, [p for _, p in prefixes])
    dt = (time.time() - t0) * 1000
    for (label, _), s in zip(prefixes, bs.tolist()):
        print(f"  {label:8s} {s:.4f}")
    print(f"  total {dt:.1f}ms")
    print()

    # score_per_step (전체 solution)
    full = "".join(p for _, p in prefixes)
    print("[score_per_step] (full solution)")
    t0 = time.time()
    per = cli.score_per_step(problem, full)
    print(f"  per_step={[f'{x:.4f}' for x in per]}")
    print(f"  ({(time.time()-t0)*1000:.1f}ms)")

    cli.close()
    print("\n✅ 모든 RPC 정상")


if __name__ == "__main__":
    main()
