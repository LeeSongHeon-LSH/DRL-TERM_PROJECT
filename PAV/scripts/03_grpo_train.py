"""W3~W4 — GRPO + LoRA 학습 entry (Ray Train wrap).

기본은 단일 GPU(num_workers=1)지만 Ray Train으로 wrap되어 있어
multi-GPU/node 확장 시 `--num-workers` 인자만 변경하면 됨.

`configs/rl_q3.yaml`의 pav.method 키로 Phase 0 ↔ Phase 1 swap.
configs/prm.yaml + policy.yaml의 mode='ray'면 분산 워커 사용.

전제 (mode='ray'):
    - Ray Head가 떠 있어야 함 (`ray start --head` 또는 docker compose ray-head)
    - prm-actor + mu-actor가 등록되어 있어야 함 (serve_prm_ray.py / serve_mu_ray.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import os

import ray

from src.train.ray_train import build_ray_trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-config", default="configs/rl_q3.yaml")
    ap.add_argument("--prm-config", default="configs/prm.yaml")
    ap.add_argument("--policy-config", default="configs/policy.yaml")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="분산 학습 worker 수 (단일 GPU=1, multi-GPU 학습=N)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="GPU 없이 (디버깅용 dry-run)")
    args = ap.parse_args()

    # 이 컨테이너는 compose의 `ray start --address=...`로 이미 cluster의 worker로 join됨.
    # ray.init(address='auto')는 같은 노드의 raylet에 attach (새 cluster 시작 X).
    namespace = os.environ.get("RAY_NAMESPACE", "pav-rl")
    print(f"Attaching to local raylet (namespace={namespace!r})")
    ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)
    print(f"Connected. Nodes: {len(ray.nodes())}, Resources: {ray.cluster_resources()}")

    # config는 절대경로로 변환 (Ray worker가 다른 cwd에서 실행될 수 있음)
    rl_config = str((REPO_ROOT / args.rl_config).resolve())
    prm_config = str((REPO_ROOT / args.prm_config).resolve())
    policy_config = str((REPO_ROOT / args.policy_config).resolve())

    trainer = build_ray_trainer(
        rl_config=rl_config,
        prm_config=prm_config,
        policy_config=policy_config,
        repo_root=str(REPO_ROOT),
        num_workers=args.num_workers,
        use_gpu=not args.no_gpu,
    )
    result = trainer.fit()
    print(f"\nTraining finished. Metrics: {result.metrics}")
    if result.checkpoint:
        print(f"Checkpoint: {result.checkpoint}")


if __name__ == "__main__":
    main()
