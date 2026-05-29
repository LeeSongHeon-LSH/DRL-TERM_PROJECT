"""RemoteMuSampler — vLLM의 OpenAI 호환 API 클라이언트.

추론 PC에서 vLLM stock 서버를 띄우면 됨 (별도 서버 코드 X):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-Math-7B-Instruct \
        --gpu-memory-utilization 0.65 \
        --max-model-len 4096 --port 8001

기존 MuSampler와 동일한 (sample_step, sample_step_batch) 인터페이스.
MCRolloutPAV는 그대로 받음.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RemoteMuConfig:
    endpoint: str = "http://localhost:8001"
    model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    timeout: float = 600.0   # K=16 batch generation은 추론 서버에서 ~120-180초 소요 가능 → 여유 600초
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    step_stop: tuple[str, ...] = ("\n\n",)
    num_replicas: int = 1    # FRP LB pool 에 등록된 μ 서버 수 (동시 요청 분할 수)
    frps_dashboard_url: str = "http://frps:7500"
    frps_dashboard_user: str = field(default_factory=lambda: __import__("os").environ.get("FRPS_DASHBOARD_USER", "admin"))
    frps_dashboard_password: str = field(default_factory=lambda: __import__("os").environ.get("FRPS_DASHBOARD_PASSWORD", "changeme"))

class _CfgShim:
    """vLLM 어떤 코드도 cfg.model_id를 참조할 수 있게 — 호환용."""

    def __init__(self, model_id: str):
        self.model_id = model_id


class RemoteMuSampler:
    """MuSampler와 동일 인터페이스 — sample_step / sample_step_batch (HTTP transport).
    
    FRP LB pool 의 살아있는 μ 서버 수를 실시간 확인 → 동적으로 병렬 분할.
    """

    def __init__(self, cfg: RemoteMuConfig):
        self.cfg = cfg
        self._local = threading.local()   # thread별 독립 client (병렬 요청 안전)
        self._endpoint = cfg.endpoint.rstrip("/")
        self._live_replicas: int | None = None   # 캐시 (None이면 아직 확인 안 함)
        self._lock = threading.Lock()
        # MuSampler처럼 .cfg.model_id 접근 호환
        # (이미 RemoteMuConfig에 model_id 있어 추가 작업 불필요)

    # ------------------------------------------------------------------ live replica discovery
    def _get_live_replicas(self) -> int:
        """FRP dashboard API 로 살아있는 mu_cluster 프록시 수 확인. 실패 시 cfg.num_replicas fallback."""
        # 캐시가 있으면 재사용 (매번 HTTP 부담 줄임)
        if self._live_replicas is not None:
            return self._live_replicas

        import httpx as _httpx
        import base64
        try:
            # frps dashboard /api/proxy/tcp 에서 proxy 목록 확인 (v0.61)
            auth_str = base64.b64encode(f"{self.cfg.frps_dashboard_user}:{self.cfg.frps_dashboard_password}".encode()).decode()
            headers = {"Authorization": f"Basic {auth_str}"}
            client = self._get_client()
            resp = client.get(f"{self.cfg.frps_dashboard_url}/api/proxy/tcp", headers=headers)
            if resp.status_code != 200:
                log.warning(f"FRP dashboard unreachable ({resp.status_code}) — fallback to cfg.num_replicas={self.cfg.num_replicas}")
                with self._lock:
                    self._live_replicas = self.cfg.num_replicas
                return self.cfg.num_replicas
            data = resp.json()
            # proxies 에서 mu_cluster group 에 속하고 online 인 것 카운트
            proxies = data.get("proxies", [])
            count = 0
            for p in proxies:
                if p.get("conf", {}).get("loadBalancer", {}).get("group") == "mu_cluster":
                    if p.get("status") == "online":
                        count += 1
            if count == 0:
                log.warning("FRP mu_cluster has 0 online proxies — fallback to cfg.num_replicas=1")
                count = 1
            log.info(f"FRP mu_cluster live replicas detected: {count}")
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

    @staticmethod
    def _build_prompt(problem: str, prefix: str) -> str:
        # MuSampler와 동일 prompt 포맷 (Qwen2.5 chat template)
        return (
            "<|im_start|>system\nYou solve math step by step. "
            "Number each step on its own line.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"
        )

    def _call_completions(self, prompt: str, n: int) -> list[str]:
        """vLLM OpenAI /v1/completions 호출 — n개 alternative 생성. retry on disconnect."""
        import time
        import httpx as _httpx

        payload = {
            "model": self.cfg.model_id,
            "prompt": prompt,
            "n": n,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_new_tokens,
            "stop": list(self.cfg.step_stop),
        }
        # 추론 PC vLLM이 KV cache fragmentation 누적으로 가끔 disconnect → 무한 재시도로 회복
        # 사용자 직접 중단할 때까지 학습 안 죽임.
        # 실패 시 _reset_client() — 깨진 keepalive socket 재사용 회피 (이전 stuck 의 근본 원인).
        delay = 2.0
        attempt = 0
        while True:
            attempt += 1
            client = self._get_client()   # retry 마다 fresh client (이전 client 가 closed 면 새로 생성)
            try:
                resp = client.post(f"{self._endpoint}/v1/completions", json=payload)
                if 500 <= resp.status_code < 600:
                    # 5xx (502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout) — 일시 장애, retry
                    log.warning(f"μ HTTP {resp.status_code} (attempt {attempt}, ∞) — retry in {delay}s")
                    self._reset_client()
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"μ HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                return [c["text"].strip() for c in data["choices"]]
            except (_httpx.RemoteProtocolError, _httpx.ReadError, _httpx.ConnectError, _httpx.ReadTimeout) as e:
                log.warning(f"μ disconnect (attempt {attempt}, ∞): {type(e).__name__} — retry in {delay}s")
                self._reset_client()
                time.sleep(delay)
                delay = min(delay * 2, 30.0)   # exponential backoff cap 30s

    # ------------------------------------------------------------------ api
    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        """n개 rollout을 live_replicas개로 분할해 동시에 요청 → FRP LB가 μ 서버들에 round-robin 분배."""
        prompt = self._build_prompt(problem, prefix)

        replicas = self._get_live_replicas()
        if n <= 1 or replicas <= 1:
            return self._call_completions(prompt, n=n)

        # n을 replicas개로 최대한 균등 분할
        base = n // replicas
        rem = n % replicas
        chunks = [base + (1 if i < rem else 0) for i in range(replicas)]
        # 0개 chunk 제거 (n < replicas 일 때)
        chunks = [c for c in chunks if c > 0]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(self._call_completions, prompt, c) for c in chunks]
            results = [f.result() for f in futures]

        # flatten
        out: list[str] = []
        for r in results:
            out.extend(r)
        return out

    # ------------------------------------------------------------------ admin
    def health(self) -> dict:
        """vLLM은 /health 가 200/empty 반환."""
        client = self._get_client()
        resp = client.get(f"{self._endpoint}/health")
        return {"ok": resp.status_code == 200, "model_id": self.cfg.model_id}

    def close(self):
        client = getattr(self._local, "client", None)
        if client is not None:
            try:
                client.close()
            finally:
                self._local.client = None
