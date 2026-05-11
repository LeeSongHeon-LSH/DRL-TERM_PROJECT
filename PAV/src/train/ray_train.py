"""Ray Train wrap — 학습 코드를 TorchTrainer로 감싸 multi-GPU/node 확장 가능하게.

단일 GPU 환경에서는 num_workers=1로 사용 (overhead 거의 0, future-ready).
multi-GPU/node로 확장 시 ScalingConfig만 변경.

학습 worker 함수는 각 분산 worker에서 동일하게 실행되며,
Ray Train이 Torch DDP 환경(rank/world_size)을 자동 설정.

reward_fn은 함수 안에서 매 worker가 자체 생성 — 같은 Ray client(PRM/μ)에 접근.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def train_worker(config: dict[str, Any]):
    """각 Ray Train worker에서 실행. config는 trainer 빌더 옵션 모음."""
    import sys
    from pathlib import Path

    # worker 프로세스에서도 PAV src 경로 인식
    repo_root = Path(config["repo_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.prm import load_prm
    from src.rollout.mu_sampler import build_mu_from_policy_yaml
    from src.train.callbacks import PAVMonitorCallback
    from src.train.grpo_trainer import build_grpo_trainer, load_rl_config
    from src.train.policy_data import build_policy, build_train_dataset
    from src.train.reward_fn import PAVRewardFn, build_pav_from_config

    rl_cfg = load_rl_config(config["rl_config"])
    prm = load_prm(config["prm_config"])
    mu = (
        build_mu_from_policy_yaml(config["policy_config"])
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
    log.info(f"PAV: {pav.name} | mode: {reward_fn.mode} | α={reward_fn.alpha} λ={reward_fn.lam}")

    policy_model, tokenizer, peft_config = build_policy(config["policy_config"])
    train_dataset = build_train_dataset(config["rl_config"], tokenizer)

    trainer = build_grpo_trainer(
        rl_cfg=rl_cfg,
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.add_callback(
        PAVMonitorCallback(
            reward_fn=reward_fn,
            dump_every=rl_cfg.get("logging", {}).get("dump_samples_every", 1000),
        )
    )
    trainer.train()


def build_ray_trainer(
    rl_config: str,
    prm_config: str,
    policy_config: str,
    repo_root: str,
    *,
    num_workers: int = 1,
    use_gpu: bool = True,
    resources_per_worker: dict | None = None,
):
    """TorchTrainer 인스턴스 반환. caller가 `.fit()` 호출.

    Args:
        num_workers: 분산 학습 worker 수. 단일 GPU 환경 = 1, multi-GPU = N
        resources_per_worker: 예 {"GPU": 1}. None이면 use_gpu=True에서 자동.
    """
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker or ({"GPU": 1} if use_gpu else None),
    )

    return TorchTrainer(
        train_loop_per_worker=train_worker,
        train_loop_config={
            "rl_config": rl_config,
            "prm_config": prm_config,
            "policy_config": policy_config,
            "repo_root": repo_root,
        },
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="pav_grpo_run",
            storage_path=None,         # 기본 ~/ray_results
        ),
    )
