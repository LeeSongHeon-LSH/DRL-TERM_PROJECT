"""GRPO trainer 빌더 — TRL ≥0.13 + LoRA + vLLM colocate.

TRL GRPOTrainer가 호출하는 reward_func 시그니처:
    reward_func(prompts: list[str|list[dict]],
                completions: list[str|list[dict]],
                **dataset_columns) -> list[float | None]

- prompts/completions는 chat 형식 (list[dict]) 또는 raw text. 우리는 raw text 사용.
- dataset의 추가 컬럼 (예: "answer")이 키워드 인자로 전달됨.
- 반환은 trajectory 당 scalar 보상 1개 — group baseline은 GRPO가 처리.

PAVRewardFn은 step-wise → 합산해서 trajectory scalar로 변환.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from ..rollout.parser import split_steps
from .reward_fn import PAVRewardFn

log = logging.getLogger(__name__)


@dataclass
class GRPOSettings:
    group_size: int = 8
    kl_beta: float = 0.04
    clip_eps: float = 0.2
    learning_rate: float = 5e-6
    total_steps: int = 5000
    warmup_steps: int = 100
    gradient_accumulation: int = 4
    max_completion_length: int = 512
    optim: str = "adamw_torch"   # "adamw_torch" | "paged_adamw_8bit" | "adamw_bnb_8bit" 등


def load_rl_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grpo_trainer(
    rl_cfg: dict,
    policy_model,
    tokenizer,
    reward_fn: PAVRewardFn,
    train_dataset,
    *,
    peft_config=None,
    answer_extractor: Callable[[str], str] | None = None,
):
    """TRL GRPOTrainer 인스턴스화.

    Args:
        rl_cfg: rl_q3.yaml dict (load_rl_config 결과)
        policy_model: AutoModelForCausalLM (PEFT 적용 전 — peft_config로 trainer가 wrap)
        tokenizer: AutoTokenizer (padding_side="left" 권장)
        reward_fn: PAVRewardFn (step-wise → trajectory scalar로 어댑팅됨)
        train_dataset: HF Dataset, "prompt" + "answer" 컬럼 필수
        peft_config: peft.LoraConfig (None이면 full fine-tune)
        answer_extractor: completion에서 정답 후보 추출 함수 (None이면 raw completion)
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

    # HF OptimizerNames enum 밖의 값은 placeholder로 대체 → 실제 optimizer는 optimizers 튜플로 주입.
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
        report_to=["wandb"] if log_cfg.get("wandb_project") else [],
        run_name=log_cfg.get("wandb_run_name"),
        logging_steps=log_cfg.get("log_every", 10),
        save_strategy="steps",
        save_steps=log_cfg.get("eval_every", 500),
    )

    # vLLM colocate (TRL ≥0.13 표준)
    vllm_cfg = rl_cfg.get("vllm", {})
    if vllm_cfg.get("colocate", True):
        grpo_cfg_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.55),
        )

    # GaLore 계열 옵티마이저는 optim_target_modules 필수.
    # HF Trainer의 GaLore 분기는 정규식 substring 매칭 — "all-linear" 키워드는 PEFT 전용이라 미지원.
    # Qwen2.5 (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj)에 명시적 매칭.
    if settings.optim.startswith("galore_"):
        grpo_cfg_kwargs["optim_target_modules"] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # TRL 버전마다 GRPOConfig 인자가 다름 (예: epsilon → epsilon_low/epsilon_high).
    # 알 수 없는 인자는 자동 drop.
    import inspect
    _sig = inspect.signature(GRPOConfig.__init__)
    _accepted = set(_sig.parameters.keys())
    _dropped = [k for k in grpo_cfg_kwargs if k not in _accepted]
    if _dropped:
        log.warning(f"GRPOConfig가 지원하지 않는 인자 drop: {_dropped}")
    grpo_cfg = GRPOConfig(**{k: v for k, v in grpo_cfg_kwargs.items() if k in _accepted})

    trl_reward = _adapt_reward_for_trl(reward_fn, answer_extractor=answer_extractor)
    trl_reward.__name__ = f"pav_{reward_fn.mode}_reward"

    trainer_kwargs = dict(
        model=policy_model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=train_dataset,
        reward_funcs=[trl_reward],
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    # 비-표준 optimizer (HF Trainer string 매핑 밖) 직접 주입.
    # peft_config가 None인 Full FT 전제 — LoRA 사용 시 wrap 이후 파라미터로 재생성 필요.
    custom_opt = _build_custom_optimizer(settings.optim, policy_model, settings.learning_rate)
    if custom_opt is not None:
        trainer_kwargs["optimizers"] = (custom_opt, None)   # scheduler는 Trainer가 default 생성

    return GRPOTrainer(**trainer_kwargs)


def _build_custom_optimizer(optim_name: str, model, lr: float):
    """HF Trainer가 string으로 매핑 못 하는 optimizer를 직접 생성. 지원 안 하면 None 반환."""
    if optim_name == "came":
        try:
            from came_pytorch import CAME
        except ImportError as e:
            raise ImportError(
                "came-pytorch가 필요합니다. `uv sync --extra gpu`로 설치 후 image rebuild."
            ) from e
        log.info(f"CAME optimizer 생성 (lr={lr})")
        return CAME(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
        )
    return None


# ---------------------------------------------------------------------- adapter
def _adapt_reward_for_trl(
    reward_fn: PAVRewardFn,
    *,
    answer_extractor: Callable[[str], str] | None = None,
) -> Callable:
    """PAVRewardFn (step-wise) → TRL reward_func (trajectory scalar).

    group_size trajectory를 ThreadPoolExecutor로 병렬 처리 (max_workers=4).
    추론 PC가 fresh 상태(누적 fragmentation 없음)면 4 동시 처리 가능 — 직렬 대비 ~3-4× 가속 기대.
    """
    from concurrent.futures import ThreadPoolExecutor

    extract = answer_extractor or _default_extract
    verifier = _build_verifier()

    def _one(prompt, completion, gold) -> float:
        problem = _to_text(prompt)
        comp_text = _to_text(completion)
        traj = split_steps(comp_text)
        if not traj:
            return 0.0
        final_correct = verifier(extract(comp_text), gold) if gold is not None else False
        step_rewards = reward_fn(problem, traj, final_correct=final_correct)
        return sum(step_rewards)

    def _trl_reward(prompts, completions, **kwargs) -> list[float]:
        gold_list = kwargs.get("answer") or [None] * len(prompts)
        inputs = list(zip(prompts, completions, gold_list))

        n_workers = max(1, min(len(inputs), 4))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            rewards = list(ex.map(lambda x: _one(*x), inputs))
        return rewards

    return _trl_reward


def _to_text(x) -> str:
    """chat 형식(list[dict])이 들어오면 마지막 user/assistant 메시지를 합쳐 평문화."""
    if isinstance(x, list) and x and isinstance(x[0], dict):
        return "\n".join(m.get("content", "") for m in x)
    return str(x)


def _default_extract(text: str) -> str:
    """\\boxed{...} 또는 마지막 줄을 정답 후보로 추출."""
    import re

    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    last = text.strip().splitlines()[-1] if text.strip() else ""
    return last.strip()


def _build_verifier():
    """math_verify가 있으면 정밀, 없으면 substring fallback."""
    try:
        from math_verify import parse, verify  # type: ignore

        def _v(pred: str, gold) -> bool:
            try:
                return bool(verify(parse(str(gold)), parse(str(pred))))
            except Exception:
                return str(gold).strip() in str(pred)

        return _v
    except ImportError:
        log.warning("math_verify 없음 — substring 매칭으로 fallback. `uv sync --extra gpu`로 설치 권장.")

        def _v(pred: str, gold) -> bool:
            return str(gold).strip() in str(pred)

        return _v
