"""GRPO + Swap Pipeline (단일 PC 학습 entry point).

기존 03_grpo_train.py와 별도. 단일 GPU에서 π / PRM / μ를 dynamic swap.

사용:
    uv run python scripts/03_grpo_train_swap.py \\
        --rl-config configs/rl_q3_swap.yaml \\
        --prm-config configs/prm.yaml \\
        --policy-config configs/policy.yaml

핵심:
    - PRM은 SwapPRM으로 wrap → CPU에 대기, score 호출 시 GPU swap
    - μ는 SwapMu (HF model) → CPU에 대기, generate 호출 시 GPU swap
    - π vLLM은 colocate + sleep(level=1) (KV cache만 비움, weight는 GPU 유지)
    - reward_fn 안에서 orchestrator.swap_to("mu"/"prm") 호출
    - reward 끝나면 swap_to("pi") — trainer가 forward/backward 가능
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.pav.differential import DifferentialPAV
from src.pav.mc_rollout import MCRolloutPAV
from src.prm.loader import load_prm
from src.swap.orchestrator import SwapOrchestrator
from src.swap.swap_prm import SwapPRM
from src.swap.swap_mu import SwapMu, SwapMuConfig
from src.swap.reward_fn import SwapAwareRewardFn
from src.swap.trainer import build_grpo_trainer_swap
from src.train.grpo_trainer import load_rl_config
from src.train.policy_data import build_policy, build_train_dataset


log = logging.getLogger("swap-train")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rl-config", default="configs/rl_q3_swap.yaml")
    ap.add_argument("--prm-config", default="configs/prm.yaml")
    ap.add_argument("--policy-config", default="configs/policy.yaml")
    args = ap.parse_args()

    # ---- configs ----
    rl_cfg = load_rl_config(args.rl_config)
    with open(args.policy_config, "r", encoding="utf-8") as f:
        policy_cfg = yaml.safe_load(f)

    log.info(f"PAV: {rl_cfg['pav']['method']} | mode: {rl_cfg['reward']['mode']} | "
             f"α={rl_cfg['reward']['alpha']} λ={rl_cfg['reward']['lam']}")

    # ---- 1. PRM (local, swap wrap) ----
    raw_prm = load_prm(args.prm_config, mode="local")
    raw_prm._ensure_loaded()
    swap_prm = SwapPRM(raw_prm)   # init 후 CPU로 swap (GPU 양보)
    log.info(f"SwapPRM ready: {raw_prm.cfg.name} (CPU 대기)")

    # ---- 2. μ HF model (swap wrap) — Phase 1 mc_rollout이면 필요 ----
    swap_mu = None
    if rl_cfg["pav"]["method"] == "mc_rollout":
        mu_cfg = policy_cfg.get("mu", {})
        swap_mu = SwapMu(SwapMuConfig(
            model_id=mu_cfg.get("model_id", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
            temperature=mu_cfg.get("temperature", 1.0),
            top_p=mu_cfg.get("top_p", 0.95),
            max_new_tokens=mu_cfg.get("max_new_tokens", 256),
            step_stop=tuple(mu_cfg.get("step_stop", ["\n\n"])),
        ))
        log.info(f"SwapMu ready: {swap_mu.cfg.model_id} (CPU 대기)")

    # ---- 3. Orchestrator (π vLLM은 trainer 생성 후 연결) ----
    orchestrator = SwapOrchestrator(pi_vllm=None, prm=swap_prm, mu=swap_mu)

    # ---- 4. PAV method (orchestrator의 swap 모델 참조) ----
    pav_method = rl_cfg["pav"]["method"]
    if pav_method == "differential":
        pav = DifferentialPAV(prm=swap_prm)
    elif pav_method == "mc_rollout":
        pav = MCRolloutPAV(prm=swap_prm, mu=swap_mu, K=rl_cfg["pav"].get("K", 16))
    else:
        raise ValueError(f"unknown pav.method: {pav_method}")

    # ---- 5. Swap-aware reward fn ----
    reward_cfg = rl_cfg["reward"]
    swap_reward_fn = SwapAwareRewardFn(
        orchestrator=orchestrator,
        pav=pav,
        reducer_mode=reward_cfg.get("mode", "Q3"),
        alpha=reward_cfg.get("alpha", 3.0),
        lam=reward_cfg.get("lam", -0.5),
        cvar_alpha=reward_cfg.get("cvar_alpha", 0.2),
    )

    # ---- 6. Policy + dataset ----
    policy_model, tokenizer, peft_config = build_policy(args.policy_config)
    train_dataset = build_train_dataset(args.rl_config, tokenizer)

    # ---- 7. Trainer build (vLLM enable_sleep_mode 시도) ----
    trainer = build_grpo_trainer_swap(
        rl_cfg=rl_cfg,
        policy_model=policy_model,
        tokenizer=tokenizer,
        swap_reward_fn=swap_reward_fn,
        orchestrator=orchestrator,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # ---- 8. PAV stats monitor (옵션) ----
    from src.train.callbacks import PAVMonitorCallback
    trainer.add_callback(
        PAVMonitorCallback(
            reward_fn=swap_reward_fn,
            dump_every=rl_cfg.get("logging", {}).get("dump_samples_every", 1000),
        )
    )

    # ---- 9. Train (resume 자동 탐색) ----
    ckpt_dir = trainer.args.output_dir
    has_ckpt = (
        os.path.isdir(ckpt_dir)
        and any(d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))
    )
    trainer.train(resume_from_checkpoint=has_ckpt if has_ckpt else None)


if __name__ == "__main__":
    main()
