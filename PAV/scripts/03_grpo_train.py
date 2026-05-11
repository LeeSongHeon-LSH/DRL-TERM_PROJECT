"""W3~W4 — GRPO + LoRA 학습 entry.

`configs/rl_q3.yaml`의 pav.method 키만 바꾸면 Phase 0 ↔ Phase 1 swap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prm import load_prm
from src.rollout.mu_sampler import build_mu_from_policy_yaml
from src.train.callbacks import PAVMonitorCallback
from src.train.grpo_trainer import build_grpo_trainer, load_rl_config
from src.train.policy_data import build_policy, build_train_dataset
from src.train.reward_fn import PAVRewardFn, build_pav_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-config", default="configs/rl_q3.yaml")
    ap.add_argument("--prm-config", default="configs/prm.yaml")
    ap.add_argument("--policy-config", default="configs/policy.yaml")
    args = ap.parse_args()

    rl_cfg = load_rl_config(args.rl_config)

    # ---- PAV 인스턴스 (method에 따라 differential / mc_rollout)
    prm = load_prm(args.prm_config)
    mu = (
        build_mu_from_policy_yaml(args.policy_config)
        if rl_cfg["pav"]["method"] == "mc_rollout"
        else None
    )
    pav = build_pav_from_config(rl_cfg["pav"], prm, mu=mu)

    reward_fn = PAVRewardFn(
        pav=pav,
        alpha=rl_cfg["reward"]["alpha"],
        mode=rl_cfg["reward"]["mode"],
        lam=rl_cfg["reward"]["lam"],
        cvar_alpha=rl_cfg["reward"]["cvar_alpha"],
    )
    print(
        f"PAV: {pav.name} | mode: {reward_fn.mode} "
        f"| α={reward_fn.alpha} λ={reward_fn.lam}"
    )

    # ---- 정책 + 데이터
    policy_model, tokenizer, peft_config = build_policy(args.policy_config)
    train_dataset = build_train_dataset(args.rl_config, tokenizer)

    trainer = build_grpo_trainer(
        rl_cfg=rl_cfg,
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    # PAV 통계 + 함정 모니터링
    trainer.add_callback(
        PAVMonitorCallback(
            reward_fn=reward_fn,
            dump_every=rl_cfg.get("logging", {}).get("dump_samples_every", 1000),
        )
    )

    trainer.train()


if __name__ == "__main__":
    main()
