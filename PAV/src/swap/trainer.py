"""build_grpo_trainer_swap — 단일 PC swap pipeline trainer 빌더.

기존 src.train.grpo_trainer:build_grpo_trainer와 별도. 기존 코드 안 건드림.

핵심 차이:
  - vLLM enable_sleep_mode=True (sleep/wake_up 지원)
  - reward_fn에 SwapAwareRewardFn (PRM/μ swap orchestrator 통합)
  - trainer step 사이에 swap_to("pi") 자동 호출 (callback)
"""
from __future__ import annotations

import logging
from typing import Callable

import yaml

from ..rollout.parser import split_steps
from ..train.grpo_trainer import (
    _adapt_reward_for_trl,
    _build_custom_optimizer,
    GRPOSettings,
)
from .orchestrator import SwapOrchestrator
from .reward_fn import SwapAwareRewardFn

log = logging.getLogger(__name__)


def build_grpo_trainer_swap(
    rl_cfg: dict,
    policy_model,
    tokenizer,
    swap_reward_fn: SwapAwareRewardFn,
    orchestrator: SwapOrchestrator,
    train_dataset,
    *,
    peft_config=None,
    answer_extractor: Callable[[str], str] | None = None,
):
    """TRL GRPOTrainer + swap pipeline.

    swap_reward_fn은 SwapAwareRewardFn — 내부에서 orchestrator로 PRM/μ swap.
    orchestrator는 trainer 학습 step 시작 시 자동으로 swap_to("pi") 호출 (callback).
    """
    from trl import GRPOConfig, GRPOTrainer

    g = rl_cfg["grpo"]
    settings = GRPOSettings(
        group_size=g.get("group_size", 8),
        kl_beta=g.get("kl_beta", 0.04),
        clip_eps=g.get("clip_eps", 0.2),
        learning_rate=g.get("learning_rate", 5e-6),
        total_steps=g.get("total_steps", 5000),
        warmup_steps=g.get("warmup_steps", 100),
        gradient_accumulation=g.get("gradient_accumulation", 4),
        max_completion_length=rl_cfg.get("vllm", {}).get(
            "max_new_tokens", g.get("max_completion_length", 512)
        ),
        optim=g.get("optim", "adamw_torch"),
    )

    log_cfg = rl_cfg.get("logging", {})
    output_dir = "./outputs/" + log_cfg.get("wandb_run_name", "pav_run")

    _CUSTOM_OPTIMS = {"came"}
    hf_optim = "adamw_torch" if settings.optim in _CUSTOM_OPTIMS else settings.optim

    grpo_cfg_kwargs = dict(
        output_dir=output_dir,
        learning_rate=settings.learning_rate,
        num_generations=settings.group_size,
        max_steps=settings.total_steps,
        warmup_steps=settings.warmup_steps,
        gradient_accumulation_steps=settings.gradient_accumulation,
        beta=settings.kl_beta,
        epsilon=settings.clip_eps,
        max_completion_length=settings.max_completion_length,
        optim=hf_optim,
        report_to=(
            ["wandb", "tensorboard"] if log_cfg.get("wandb_project")
            else ["tensorboard"]
        ),
        run_name=log_cfg.get("wandb_run_name"),
        logging_steps=log_cfg.get("log_every", 10),
        save_strategy="steps",
        save_steps=log_cfg.get("eval_every", 500),
    )

    # vLLM colocate + sleep mode (swap orchestrator 핵심)
    vllm_cfg = rl_cfg.get("vllm", {})
    if vllm_cfg.get("colocate", True):
        grpo_cfg_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.15),
        )
        # TRL 0.13~0.15에 enable_sleep_mode 옵션 노출 안 됨 → trainer 생성 후 manual patch
        # (아래 patch_vllm_sleep_mode)

    if settings.optim.startswith("galore_"):
        grpo_cfg_kwargs["optim_target_modules"] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    import inspect
    _sig = inspect.signature(GRPOConfig.__init__)
    _accepted = set(_sig.parameters.keys())
    _dropped = [k for k in grpo_cfg_kwargs if k not in _accepted]
    if _dropped:
        log.warning(f"GRPOConfig가 지원하지 않는 인자 drop: {_dropped}")
    grpo_cfg = GRPOConfig(**{k: v for k, v in grpo_cfg_kwargs.items() if k in _accepted})

    # ---- swap-aware reward wrapper ----
    trl_reward = _adapt_swap_reward_for_trl(
        swap_reward_fn, orchestrator, answer_extractor=answer_extractor
    )
    trl_reward.__name__ = f"pav_{swap_reward_fn.mode}_swap_reward"

    trainer_kwargs = dict(
        model=policy_model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=train_dataset,
        reward_funcs=[trl_reward],
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    custom_opt = _build_custom_optimizer(settings.optim, policy_model, settings.learning_rate)
    if custom_opt is not None:
        trainer_kwargs["optimizers"] = (custom_opt, None)

    trainer = GRPOTrainer(**trainer_kwargs)

    # trainer.llm = TRL이 만든 vLLM 인스턴스. orchestrator에 연결.
    if hasattr(trainer, "llm"):
        orchestrator.pi_vllm = trainer.llm
        # vLLM이 enable_sleep_mode=True로 init됐는지 patch 시도
        _patch_vllm_sleep_mode(trainer.llm)
    else:
        log.warning("GRPOTrainer.llm 속성 없음 — vLLM swap 비활성. PRM/μ만 swap.")

    # jsonl metrics logger (기존과 동일)
    from ..train.callbacks import JsonlMetricsCallback
    trainer.add_callback(JsonlMetricsCallback(output_dir))

    return trainer


def _adapt_swap_reward_for_trl(
    swap_reward_fn: SwapAwareRewardFn,
    orchestrator: SwapOrchestrator,
    *,
    answer_extractor: Callable[[str], str] | None = None,
) -> Callable:
    """SwapAwareRewardFn → TRL reward_func (trajectory scalar list).

    호출 시 swap_to("mu"/"prm")으로 GPU 모델 교체.
    호출 끝나면 swap_to("pi") — trainer가 forward/backward를 위해 π 필요.
    """
    from ..train.grpo_trainer import _default_extract, _build_verifier, _to_text

    extract = answer_extractor or _default_extract
    verifier = _build_verifier()

    def _trl_reward(prompts, completions, **kwargs) -> list[float]:
        gold_list = kwargs.get("answer") or [None] * len(prompts)

        rewards: list[float] = []
        # 직렬 (단일 GPU swap이라 병렬 의미 없음)
        for prompt, completion, gold in zip(prompts, completions, gold_list):
            problem = _to_text(prompt)
            comp_text = _to_text(completion)
            traj = split_steps(comp_text)
            if not traj:
                rewards.append(0.0)
                continue
            final_correct = verifier(extract(comp_text), gold) if gold is not None else False
            step_rewards = swap_reward_fn(problem, traj, final_correct=final_correct)
            rewards.append(sum(step_rewards))

        # reward 계산 끝 → π로 swap (trainer가 forward/backward를 위해 필요)
        orchestrator.swap_to("pi")
        log.debug(orchestrator.report())
        return rewards

    return _trl_reward


def _patch_vllm_sleep_mode(llm):
    """TRL이 만든 vLLM 인스턴스가 sleep_mode 지원하는지 확인 + 활성화 시도.

    vLLM 0.6+는 LLM(enable_sleep_mode=True) 옵션. TRL에서 노출 안 되면 monkeypatch 필요.
    여기선 안전하게 hasattr 체크만.
    """
    if hasattr(llm, "sleep") and hasattr(llm, "wake_up"):
        log.info("vLLM sleep mode 사용 가능 (llm.sleep/wake_up 존재).")
        # 실제 sleep 호출 시 enable_sleep_mode=False면 RuntimeError 발생.
        # 그땐 orchestrator가 try/except로 처리.
    else:
        log.warning("vLLM 인스턴스에 sleep/wake_up 메서드 없음 — sleep mode 비활성. PRM/μ만 swap됨.")
