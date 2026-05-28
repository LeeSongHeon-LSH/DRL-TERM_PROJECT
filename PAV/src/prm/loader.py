"""PRM 로더 — Skywork-o1-Open-PRM-Qwen-2.5-1.5B (천공 PRM, 경량 default).

PRM_MODEL (TRL ValueHead 패턴) 기반. 동일 family 7B 변형도 동작.
실제 가중치 다운로드는 scripts/download_models.py 또는:
    huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

mode:
    "local"   → 같은 GPU에 PRM 가중치 로드 (PRM 클래스 반환)
    "remote"  → HTTP로 PRM 서버 호출 (RemotePRM 클래스 반환)
                서버: scripts/serve_prm_http.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .score import PRM


@dataclass
class _RemoteSection:
    endpoint: str = "http://localhost:8002"
    timeout: float = 120.0


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
    # ---- remote (HTTP)
    mode: str = "local"                       # "local" | "remote"
    remote: dict = field(default_factory=dict)
    # ---- FRP LB pool (remote mode only)
    num_replicas: int = 1
    frps_dashboard_url: str = "http://frps:7500"

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

    환경변수 override:
        PRM_ENDPOINT  → remote.endpoint를 override (mode='remote'일 때만)
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
        from .remote_client import RemotePRM, RemotePRMConfig

        remote = dict(cfg.remote or {})
        # 환경변수 override (배포 시 yaml 수정 없이 endpoint 변경)
        env_endpoint = os.environ.get("PRM_ENDPOINT")
        if env_endpoint:
            remote["endpoint"] = env_endpoint
        # FRP LB pool 설정 (yaml 또는 환경변수에서)
        num_replicas = int(os.environ.get("PRM_REPLICAS", getattr(cfg, "num_replicas", 1)))
        frps_dashboard_url = os.environ.get("FRPS_DASHBOARD_URL", getattr(cfg, "frps_dashboard_url", "http://frps:7500"))
        return RemotePRM(
            RemotePRMConfig(
                name=cfg.name,
                model_id=cfg.model_id,
                endpoint=remote.get("endpoint", "http://localhost:8002"),
                timeout=float(remote.get("timeout", 120.0)),
                quantization=cfg.quantization,
                batch_size=cfg.batch_size,
                step_token=cfg.step_token,
                num_replicas=num_replicas,
                frps_dashboard_url=frps_dashboard_url,
            )
        )
    return PRM(cfg)
