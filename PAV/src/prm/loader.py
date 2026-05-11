"""PRM 로더 — Skywork-o1-Open-PRM-Qwen-2.5-1.5B (천공 PRM, 경량 default).

PRM_MODEL (TRL ValueHead 패턴) 기반. 동일 family 7B 변형도 동작.
실제 가중치 다운로드는 scripts/download_models.py 또는:
    huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

mode="remote"이면 분산된 PRM 서버(scripts/serve_prm.py)에 HTTP로 요청.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .remote_client import RemotePRM, RemotePRMConfig
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
    # ---- remote (RabbitMQ RPC)
    mode: str = "local"                       # "local" | "remote"
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    request_queue: str = "prm.requests"
    rpc_timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PRMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_prm(config: str | Path | PRMConfig | dict, **overrides: Any):
    """PRM 또는 RemotePRM 인스턴스 생성.

    Returns:
        PRM (mode='local')  또는  RemotePRM (mode='remote') —
        둘 다 score / score_batch / score_per_step 인터페이스.
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

    if cfg.mode == "remote":
        return RemotePRM(
            RemotePRMConfig(
                name=cfg.name,
                model_id=cfg.model_id,
                amqp_url=cfg.amqp_url,
                request_queue=cfg.request_queue,
                rpc_timeout=cfg.rpc_timeout,
                step_token=cfg.step_token,
                batch_size=cfg.batch_size,
            )
        )
    return PRM(cfg)
