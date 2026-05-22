"""RemotePRM — HTTP 클라이언트 (FastAPI 서버와 통신).

기존 PRM과 동일한 (score, score_batch, score_per_step) 인터페이스 유지 —
DifferentialPAV / MCRolloutPAV / PAVRewardFn은 RemotePRM을 그대로 받음.

서버: scripts/serve_prm_http.py (FastAPI + uvicorn)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

log = logging.getLogger(__name__)


@dataclass
class RemotePRMConfig:
    name: str = "remote-prm"
    model_id: str = "remote"
    endpoint: str = "http://localhost:8002"
    timeout: float = 300.0                # K=16 score_batch가 추론 서버에서 한 번에 처리 → 여유 300초
    quantization: str = "remote"          # 호환용 placeholder
    batch_size: int = 32                  # 16 → 32 (추론 서버 여유 있음, 한 번에 더 많이)
    step_token: str = "\n"


class RemotePRM:
    """PRM과 동일 인터페이스 — score / score_batch / score_per_step (HTTP transport)."""

    def __init__(self, cfg: RemotePRMConfig):
        self.cfg = cfg
        self._client = None
        self._endpoint = cfg.endpoint.rstrip("/")

    # ------------------------------------------------------------------ http
    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.cfg.timeout)
        return self._client

    def _post(self, path: str, payload: dict) -> dict:
        client = self._get_client()
        url = f"{self._endpoint}{path}"
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"PRM HTTP {resp.status_code} on {path}: {resp.text[:200]}"
            )
        return resp.json()

    # ------------------------------------------------------------------ api
    @torch.no_grad()
    def score(self, problem: str, solution_prefix: str) -> torch.Tensor:
        if not solution_prefix.strip():
            return torch.tensor(0.5)
        data = self._post(
            "/v1/score",
            {"problem": problem, "solution_prefix": solution_prefix},
        )
        return torch.tensor(float(data["score"]))

    @torch.no_grad()
    def score_batch(
        self,
        problem: str,
        solution_prefixes: Sequence[str],
    ) -> torch.Tensor:
        prefixes = list(solution_prefixes)
        if not prefixes:
            return torch.empty(0)
        out: list[float] = []
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(prefixes), bs):
            chunk = prefixes[i : i + bs]
            data = self._post(
                "/v1/score_batch",
                {"problem": problem, "solution_prefixes": chunk},
            )
            out.extend(float(x) for x in data["scores"])
        return torch.tensor(out, dtype=torch.float32)

    @torch.no_grad()
    def score_per_step(self, problem: str, solution: str) -> list[float]:
        data = self._post(
            "/v1/score_per_step",
            {"problem": problem, "solution": solution},
        )
        return [float(x) for x in data["per_step"]]

    # ------------------------------------------------------------------ admin
    def health(self) -> dict:
        client = self._get_client()
        resp = client.get(f"{self._endpoint}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
