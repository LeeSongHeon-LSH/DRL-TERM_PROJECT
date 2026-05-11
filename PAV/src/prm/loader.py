"""PRM 로더 — Skywork-o1-Open-PRM-Qwen-2.5-1.5B (천공 PRM, 경량 default).

PRM_MODEL (TRL ValueHead 패턴) 기반. 동일 family 7B 변형도 동작.
실제 가중치 다운로드는 scripts/download_models.py 또는:
    huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

mode 분기:
    "local" — 같은 GPU에 PRM 가중치 로드 (PRM 인스턴스)
    "ray"   — Ray cluster에 등록된 named actor(들)에 RPC (RayPRMClient / RayPRMClientPool)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .score import PRM


@dataclass
class PRMConfig:
    name: str
    model_id: str
    quantization: str = "none"
    dtype: str = "float16"
    device: str = "cuda"
    max_model_len: int = 4096
    step_token: str = "\n"
    batch_size: int = 16
    # ---- 분산 (Ray RPC)
    mode: str = "local"                       # "local" | "ray"
    ray_address: str = "auto"                 # ray.init(address=...) — "auto"는 head 자동 감지
    namespace: str = "pav-rl"
    actor_name: str = "prm-actor"             # named actor (단일) 또는 prefix (multi: prm-actor-0, -1, ...)
    num_replicas: int = 1                     # 1: 단일 actor (RayPRMClient), 2+: Pool round-robin
    rpc_timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PRMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_prm(config: str | Path | PRMConfig | dict, **overrides: Any):
    """PRM 또는 RayPRMClient/Pool 인스턴스 생성.

    Returns:
        PRM (mode='local') | RayPRMClient (mode='ray' + num_replicas=1)
                          | RayPRMClientPool (mode='ray' + num_replicas≥2)
        세 케이스 모두 score / score_batch / score_per_step 인터페이스 동일.
    """
    if isinstance(config, (str, Path)):
        cfg = PRMConfig.from_yaml(config)
    elif isinstance(config, dict):
        known = {k: v for k, v in config.items() if k in PRMConfig.__dataclass_fields__}
        cfg = PRMConfig(**known)
    elif isinstance(config, PRMConfig):
        cfg = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    for k, v in overrides.items():
        setattr(cfg, k, v)

    if cfg.mode == "ray":
        return _load_ray(cfg)
    if cfg.mode == "local":
        return PRM(cfg)
    raise ValueError(
        f"Unknown PRM mode: {cfg.mode!r} (지원: 'local' | 'ray')"
    )


# ---------------------------------------------------------------- ray helpers
def _load_ray(cfg: PRMConfig):
    import ray

    from .ray_client import RayPRMClient, RayPRMClientPool, RayPRMConfig

    # head에 연결 — 이미 init되어 있으면 skip
    if not ray.is_initialized():
        ray.init(address=cfg.ray_address, namespace=cfg.namespace, ignore_reinit_error=True)

    rc = RayPRMConfig(
        name=cfg.name,
        model_id=cfg.model_id,
        actor_name=cfg.actor_name,
        namespace=cfg.namespace,
        num_replicas=cfg.num_replicas,
        rpc_timeout=cfg.rpc_timeout,
        step_token=cfg.step_token,
        batch_size=cfg.batch_size,
    )
    if cfg.num_replicas <= 1:
        return RayPRMClient(rc)
    return RayPRMClientPool(rc)
