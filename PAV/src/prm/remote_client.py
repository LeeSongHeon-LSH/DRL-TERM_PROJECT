"""RemotePRM — RabbitMQ RPC 클라이언트.

본체에서 publish하면 PRM 워커들이 큐에서 consume → 응답을 본체의 임시 reply_queue로 회신.
RabbitMQ가 워커 풀 라운드로빈을 자동 처리.

기존 PRM과 동일한 (score, score_batch, score_per_step) 인터페이스 유지 —
DifferentialPAV / MCRolloutPAV / PAVRewardFn은 RemotePRM을 그대로 받음.

설계:
  - BlockingConnection (pika 표준 동기 RPC 패턴, correlation_id + reply_to)
  - 단일 connection / channel / exclusive 임시 reply queue를 인스턴스가 보유
  - PAVRewardFn이 단일 스레드에서 순차 호출하는 가정 (TRL GRPO single-process per GPU OK)
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Sequence

import torch

log = logging.getLogger(__name__)


@dataclass
class RemotePRMConfig:
    name: str = "remote-prm"
    model_id: str = "remote"
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    request_queue: str = "prm.requests"
    rpc_timeout: float = 120.0
    step_token: str = "\n"             # 워커 측 PRMConfig와 일치 (참고용)
    batch_size: int = 16               # 한 RPC 요청에 묶을 prefix 수


class RemotePRM:
    """PRM과 동일 인터페이스 — score / score_batch / score_per_step."""

    def __init__(self, cfg: RemotePRMConfig):
        self.cfg = cfg
        self._conn = None
        self._channel = None
        self._reply_queue: str | None = None
        self._responses: dict[str, bytes] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ pika
    def _ensure_connected(self):
        if self._channel is not None:
            return
        import pika  # base deps

        self._conn = pika.BlockingConnection(pika.URLParameters(self.cfg.amqp_url))
        self._channel = self._conn.channel()
        # 요청 큐 declare (워커가 같은 이름으로 만들기 전에 publish해도 안전)
        self._channel.queue_declare(queue=self.cfg.request_queue, durable=False)
        # 임시 reply queue (exclusive, auto-delete)
        result = self._channel.queue_declare(queue="", exclusive=True, auto_delete=True)
        self._reply_queue = result.method.queue
        self._channel.basic_consume(
            queue=self._reply_queue,
            on_message_callback=self._on_response,
            auto_ack=True,
        )

    def _on_response(self, ch, method, properties, body):
        self._responses[properties.correlation_id] = body

    # ------------------------------------------------------------------ rpc
    def _call(self, op: str, payload: dict) -> dict:
        """publish + reply_queue 폴링. 단일 스레드 호출 가정."""
        import pika

        with self._lock:
            self._ensure_connected()
            cid = str(uuid.uuid4())
            req = {"op": op, **payload}
            self._channel.basic_publish(
                exchange="",
                routing_key=self.cfg.request_queue,
                properties=pika.BasicProperties(
                    reply_to=self._reply_queue,
                    correlation_id=cid,
                    content_type="application/json",
                ),
                body=json.dumps(req).encode("utf-8"),
            )

            # reply 대기
            elapsed = 0.0
            poll = 0.05
            while cid not in self._responses:
                self._conn.process_data_events(time_limit=poll)
                elapsed += poll
                if elapsed >= self.cfg.rpc_timeout:
                    raise TimeoutError(
                        f"PRM RPC timeout ({self.cfg.rpc_timeout}s) for op={op}"
                    )

            data = json.loads(self._responses.pop(cid))

        if "error" in data:
            raise RuntimeError(f"PRM worker error: {data['error']}")
        return data

    # ------------------------------------------------------------------ api
    @torch.no_grad()
    def score(self, problem: str, solution_prefix: str) -> torch.Tensor:
        if not solution_prefix.strip():
            return torch.tensor(0.5)
        out = self._call("score", {"problem": problem, "solution_prefix": solution_prefix})
        return torch.tensor(float(out["score"]))

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
            data = self._call(
                "score_batch",
                {"problem": problem, "solution_prefixes": chunk},
            )
            out.extend(float(x) for x in data["scores"])
        return torch.tensor(out, dtype=torch.float32)

    @torch.no_grad()
    def score_per_step(self, problem: str, solution: str) -> list[float]:
        data = self._call(
            "score_per_step", {"problem": problem, "solution": solution}
        )
        return [float(x) for x in data["per_step"]]

    # ------------------------------------------------------------------ admin
    def health(self) -> dict:
        return self._call("health", {})

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
                self._channel = None
                self._reply_queue = None
