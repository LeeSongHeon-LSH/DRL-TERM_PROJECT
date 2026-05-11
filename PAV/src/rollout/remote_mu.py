"""RemoteMuSampler — RabbitMQ RPC 클라이언트.

기존 MuSampler와 동일한 (sample_step, sample_step_batch) 인터페이스. MCRolloutPAV는 그대로 받음.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class RemoteMuConfig:
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    request_queue: str = "mu.requests"
    rpc_timeout: float = 180.0
    # 서버 측 SamplingParams 힌트 (현재 무시)
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256


class RemoteMuSampler:
    def __init__(self, cfg: RemoteMuConfig):
        self.cfg = cfg
        self._conn = None
        self._channel = None
        self._reply_queue: str | None = None
        self._responses: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def _ensure_connected(self):
        if self._channel is not None:
            return
        import pika

        self._conn = pika.BlockingConnection(pika.URLParameters(self.cfg.amqp_url))
        self._channel = self._conn.channel()
        self._channel.queue_declare(queue=self.cfg.request_queue, durable=False)
        result = self._channel.queue_declare(queue="", exclusive=True, auto_delete=True)
        self._reply_queue = result.method.queue
        self._channel.basic_consume(
            queue=self._reply_queue,
            on_message_callback=self._on_response,
            auto_ack=True,
        )

    def _on_response(self, ch, method, properties, body):
        self._responses[properties.correlation_id] = body

    def _call(self, op: str, payload: dict) -> dict:
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

            elapsed = 0.0
            poll = 0.05
            while cid not in self._responses:
                self._conn.process_data_events(time_limit=poll)
                elapsed += poll
                if elapsed >= self.cfg.rpc_timeout:
                    raise TimeoutError(
                        f"μ RPC timeout ({self.cfg.rpc_timeout}s) for op={op}"
                    )

            data = json.loads(self._responses.pop(cid))

        if "error" in data:
            raise RuntimeError(f"μ worker error: {data['error']}")
        return data

    # ------------------------------------------------------------------ api
    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        data = self._call(
            "sample",
            {
                "problem": problem,
                "prefix": prefix,
                "n": n,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "max_new_tokens": self.cfg.max_new_tokens,
            },
        )
        return list(data["steps"])

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
