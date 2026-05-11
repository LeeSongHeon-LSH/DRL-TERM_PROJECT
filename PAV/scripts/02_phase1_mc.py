"""W2 Day 1~5 — Phase 1 MCRolloutPAV smoke + K 비교.

K ∈ {4, 8, 16, 32}로 latency / std 비교, A_samples histogram dump.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pav import MCRolloutPAV
from src.prm import load_prm
from src.rollout.mu_sampler import build_mu_from_policy_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prm-config", default="configs/prm.yaml")
    ap.add_argument("--policy-config", default="configs/policy.yaml")
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()

    prm = load_prm(args.prm_config)
    mu = build_mu_from_policy_yaml(args.policy_config)

    problem = "Solve x^2 - 5x + 6 = 0."
    prefix = ""
    step = "Step 1: Factor the quadratic as (x - 2)(x - 3) = 0.\n\n"

    print(f"problem: {problem}")
    print(f"step   : {step.strip()}")
    print(f"\n{'K':>4s}  {'latency_ms':>10s}  {'A_mean':>8s}  {'A_std':>8s}  {'q05':>8s}  {'q95':>8s}")
    for K in args.ks:
        pav = MCRolloutPAV(prm, mu, K=K)
        t0 = time.time()
        out = pav(problem, prefix, step)
        dt = (time.time() - t0) * 1000
        A = out["advantage_samples"]
        print(
            f"{K:>4d}  {dt:>10.1f}  "
            f"{A.mean().item():>8.4f}  "
            f"{A.std().item():>8.4f}  "
            f"{A.quantile(0.05).item():>8.4f}  "
            f"{A.quantile(0.95).item():>8.4f}"
        )


if __name__ == "__main__":
    main()
