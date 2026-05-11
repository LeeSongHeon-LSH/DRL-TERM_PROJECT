"""모든 모듈이 import-error 없이 로드되는지 — 새로 추가한 부분 회귀 검사."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_import_pav():
    from src.pav import DifferentialPAV, MCRolloutPAV, PAVMethod, reduce_advantage
    from src.pav.base import is_distributional
    assert PAVMethod is not None
    assert callable(reduce_advantage)
    assert callable(is_distributional)
    assert DifferentialPAV.__name__ == "DifferentialPAV"
    assert MCRolloutPAV.__name__ == "MCRolloutPAV"


def test_import_prm():
    from src.prm import PRM, load_prm
    assert callable(load_prm)
    assert PRM is not None


def test_import_skywork_helpers():
    from src.prm.skywork import (
        PRM_MODEL,
        derive_step_rewards,
        derive_step_rewards_vllm,
        prepare_batch_input_for_model,
        prepare_input,
    )
    assert PRM_MODEL is not None
    assert callable(prepare_input)
    assert callable(prepare_batch_input_for_model)
    assert callable(derive_step_rewards)
    assert callable(derive_step_rewards_vllm)


def test_import_train():
    from src.train import (
        PAVMonitorCallback,
        PAVRewardFn,
        build_eval_dataset,
        build_pav_from_config,
        build_policy,
        build_train_dataset,
    )
    assert PAVRewardFn is not None
    assert callable(build_pav_from_config)
    assert callable(build_policy)
    assert callable(build_train_dataset)
    assert callable(build_eval_dataset)
    assert PAVMonitorCallback is not None


def test_import_rollout():
    from src.rollout import MuSampler, VLLMRollout, normalize_step, split_steps
    from src.rollout.mu_sampler import MuConfig, build_mu_from_policy_yaml
    from src.rollout.vllm_rollout import Trajectory, VLLMRolloutConfig
    assert MuSampler is not None
    assert VLLMRollout is not None
    assert callable(split_steps)
    assert callable(normalize_step)
    assert MuConfig is not None
    assert VLLMRolloutConfig is not None
    assert Trajectory is not None
    assert callable(build_mu_from_policy_yaml)


def test_import_eval():
    from src.eval import bon_pav, run_sanity_checks
    from src.eval.sanity import SanityItem, SanityResult
    assert callable(bon_pav)
    assert callable(run_sanity_checks)
    assert SanityItem is not None
    assert SanityResult is not None


def test_import_grpo_trainer():
    """trl이 미설치라도 모듈 자체는 import 가능해야 함 (lazy import 패턴)."""
    from src.train.grpo_trainer import GRPOSettings, build_grpo_trainer, load_rl_config
    assert callable(load_rl_config)
    assert callable(build_grpo_trainer)
    assert GRPOSettings is not None


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    n = 0
    for f in funcs:
        try:
            f()
            print(f"PASS  {f.__name__}")
            n += 1
        except Exception:
            print(f"FAIL  {f.__name__}")
            traceback.print_exc()
    print(f"\n{n}/{len(funcs)} passed")
