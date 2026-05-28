"""RemotePRM — HTTP 클라이언트 (FastAPI 서버와 통신).

기존 PRM과 동일한 (score, score_batch, score_per_step) 인터페이스 유지 —
DifferentialPAV / MCRolloutPAV / PAVRewardFn은 RemotePRM을 그대로 받음.

서버: scripts/serve_prm_http.py (FastAPI + uvicorn)

FRP LB pool 분산:
    - frps dashboard API 로 살아있는 prm_cluster 프록시 수 확인
    - score_batch / score_per_step 을 live_replicas 개로 분할 → ThreadPoolExecutor 동시 요청
    - 각 chunk 는 FRP LB 가 round-robin 으로 다른 PRM 서버에 분배
"""
from __future__ import annotations

import logging
import threading
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
    # ---- FRP LB pool (remote_mu.py 와 동일 패턴)
    num_replicas: int = 1                 # FRP LB pool 에 등록된 PRM 서버 수 (동시 요청 분할 수)
    frps_dashboard_url: str = "http://frps:7500"   # FRP server dashboard (live replica 확인용)


class RemotePRM:
    """PRM과 동일 인터페이스 — score / score_batch / score_per_step (HTTP transport).

    FRP LB pool 의 살아있는 PRM 서버 수를 실시간 확인 → 동적으로 병렬 분할.
    """

    def __init__(self, cfg: RemotePRMConfig):
        self.cfg = cfg
        self._local = threading.local()   # thread별 독립 client (병렬 요청 안전)
        self._endpoint = cfg.endpoint.rstrip("/")
        self._live_replicas: int | None = None   # 캐시 (None이면 아직 확인 안 함)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ live replica discovery
    def _get_live_replicas(self) -> int:
        """FRP dashboard API 로 살아있는 prm_cluster 프록시 수 확인. 실패 시 cfg.num_replicas fallback."""
        # 캐시가 있으면 재사용 (매번 HTTP 부담 줄임)
        if self._live_replicas is not None:
            return self._live_replicas

        import httpx as _httpx
        try:
            # frps dashboard /api/status 에서 proxy 목록 확인
            client = self._get_client()
            resp = client.get(f"{self.cfg.frps_dashboard_url}/api/status")
            if resp.status_code != 200:
                log.warning(f"FRP dashboard unreachable ({resp.status_code}) — fallback to cfg.num_replicas={self.cfg.num_replicas}")
                with self._lock:
                    self._live_replicas = self.cfg.num_replicas
                return self.cfg.num_replicas
            data = resp.json()
            # proxyInfos 에서 prm_cluster group 에 속하고 online 인 것 카운트
            proxies = data.get("proxyInfos", [])
            count = 0
            for p in proxies:
                if p.get("conf", {}).get("loadBalancer", {}).get("group") == "prm_cluster":
                    if p.get("status") == "online":
                        count += 1
            if count == 0:
                log.warning("FRP prm_cluster has 0 online proxies — fallback to cfg.num_replicas=1")
                count = 1
            log.info(f"FRP prm_cluster live replicas detected: {count}")
            with self._lock:
                self._live_replicas = count
            return count
        except Exception as e:
            log.warning(f"FRP discovery failed ({e}) — fallback to cfg.num_replicas={self.cfg.num_replicas}")
            with self._lock:
                self._live_replicas = self.cfg.num_replicas
            return self.cfg.num_replicas

    def refresh_live_replicas(self) -> int:
        """외부에서 호출 → live replica 수 강제 재확인 (학습 중간에 추론 PC 추가/제거 시)."""
        with self._lock:
            self._live_replicas = None
        return self._get_live_replicas()

    # ------------------------------------------------------------------ http
    def _get_client(self):
        if getattr(self._local, "client", None) is None:
            import httpx
            self._local.client = httpx.Client(timeout=self.cfg.timeout)
        return self._local.client

    def _reset_client(self):
        """keepalive pool 의 깨진 socket 강제 폐기 — 다음 _get_client() 가 fresh 생성."""
        client = getattr(self._local, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            self._local.client = None

    def _post(self, path: str, payload: dict) -> dict:
        """PRM HTTP POST. 무한 retry on disconnect (사용자 직접 중단할 때까지 학습 안 죽임).

        실패 시 _reset_client() — 깨진 keepalive socket 재사용 회피 (이전 stuck 의 근본 원인).
        """
        import time
        import httpx as _httpx

        url = f"{self._endpoint}{path}"
        delay = 2.0
        attempt = 0
        while True:
            attempt += 1
            client = self._get_client()   # retry 마다 fresh client (이전 client 가 closed 면 새로 생성)
            try:
                resp = client.post(url, json=payload)
                if 500 <= resp.status_code < 600:
                    # 5xx (nginx upstream 일시 장애 등) — retry
                    log.warning(f"PRM HTTP {resp.status_code} on {path} (attempt {attempt}, ∞) — retry in {delay}s")
                    self._reset_client()
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"PRM HTTP {resp.status_code} on {path}: {resp.text[:200]}"
                    )
                return resp.json()
            except (_httpx.RemoteProtocolError, _httpx.ReadError, _httpx.ConnectError, _httpx.ReadTimeout) as e:
                log.warning(f"PRM disconnect on {path} (attempt {attempt}, ∞): {type(e).__name__} — retry in {delay}s")
                self._reset_client()
                time.sleep(delay)
                delay = min(delay * 2, 30.0)   # exponential backoff cap 30s

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
        """FRP LB pool 에 병렬 분산 → throughput = live_replicas × 단일 서버 throughput."""
        prefixes = list(solution_prefixes)
        if not prefixes:
            return torch.empty(0)

        replicas = self._get_live_replicas()
        if len(prefixes) <= 1 or replicas <= 1:
            # 단일 요청이거나 replica 1개면 직접 처리
            return self._score_batch_single(problem, prefixes)

        # prefixes 를 replicas 개로 최대한 균등 분할
        base = len(prefixes) // replicas
        rem = len(prefixes) % replicas
        chunks = []
        idx = 0
        for i in range(replicas):
            size = base + (1 if i < rem else 0)
            if size > 0:
                chunks.append(prefixes[idx:idx + size])
                idx += size

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(self._score_batch_single, problem, c) for c in chunks]
            results = [f.result() for f in futures]

        # flatten
        out: list[float] = []
        for r in results:
            out.extend(r.tolist())
        return torch.tensor(out, dtype=torch.float32)

    def _score_batch_single(self, problem: str, prefixes: list[str]) -> torch.Tensor:
        """단일 HTTP 요청으로 score_batch 처리 (내부 mini-batch)."""
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
        """FRP LB pool 에 병렬 분산 → 여러 solution 을 동시에 채점."""
        # 단일 solution 이면 직접 처리
        data = self._post(
            "/v1/score_per_step",
            {"problem": problem, "solution": solution},
        )
        return [float(x) for x in data["per_step"]]

    @torch.no_grad()
    def score_per_step_batch(
        self,
        problem: str,
        solutions: Sequence[str],
    ) -> list[list[float]]:
        """여러 solution 을 FRP LB pool 에 병렬 분산 → 각 PRM 서버가 독립 처리.

        Step-wise PRM 의 group_size trajectory 를 한 번에 처리할 때 사용.
        """
        solutions = list(solutions)
        if not solutions:
            return []

        replicas = self._get_live_replicas()
        if len(solutions) <= 1 or replicas <= 1:
            return [self.score_per_step(problem, s) for s in solutions]

        # solutions 를 replicas 개로 분할
        base = len(solutions) // replicas
        rem = len(solutions) % replicas
        chunks = []
        idx = 0
        for i in range(replicas):
            size = base + (1 if i < rem else 0)
            if size > 0:
                chunks.append(solutions[idx:idx + size])
                idx += size

        from concurrent.futures import ThreadPoolExecutor

        def _batch_one(chunk):
            return [self.score_per_step(problem, s) for s in chunk]

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_batch_one, c) for c in chunks]
            results = [f.result() for f in futures]

        # flatten
        out: list[list[float]] = []
        for r in results:
            out.extend(r)
        return out

    # ------------------------------------------------------------------ admin
    def health(self) -> dict:
        client = self._get_client()
        resp = client.get(f"{self._endpoint}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        client = getattr(self._local, "client", None)
        if client is not None:
            try:
                client.close()
            finally:
                self._local.client = None
