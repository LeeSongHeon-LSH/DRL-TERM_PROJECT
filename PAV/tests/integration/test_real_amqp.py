"""실제 RabbitMQ broker 와의 end-to-end smoke.

- 사전: docker compose --profile broker up -d  (localhost:5672)
- 워커는 이 테스트가 직접 consume — 가짜 PRM/μ를 인-메모리로 처리.
- 학습 컨테이너 빌드 전에 큐 ↔ 클라이언트 통신 자체를 검증.

skip 조건: pika 미설치 또는 broker 미가용.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest
import torch

pytest.importorskip("pika")
import pika


AMQP_URL = "amqp://guest:guest@localhost:5672/"


def _broker_alive() -> bool:
    try:
        conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
        conn.close()
        return True
    except Exception:
        return False


broker_alive = pytest.mark.skipif(
    not _broker_alive(), reason="RabbitMQ broker not running on localhost:5672"
)


# ----------------------------------------------------------------- fake worker
def _start_fake_worker(queue: str, handler) -> threading.Thread:
    """별도 스레드에서 큐를 consume — 메시지마다 handler(body) → reply 로 응답."""

    def _run():
        conn = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
        ch = conn.channel()
        ch.queue_declare(queue=queue, durable=False)
        ch.basic_qos(prefetch_count=1)

        def _on_msg(c, method, props, body):
            reply = handler(body)
            if props.reply_to:
                c.basic_publish(
                    exchange="",
                    routing_key=props.reply_to,
                    properties=pika.BasicProperties(correlation_id=props.correlation_id),
                    body=reply,
                )
            c.basic_ack(delivery_tag=method.delivery_tag)

        ch.basic_consume(queue=queue, on_message_callback=_on_msg, auto_ack=False)
        try:
            ch.start_consuming()
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.5)  # consumer 등록 대기
    return t


# ----------------------------------------------------------------- tests
@broker_alive
def test_real_remote_prm_score_through_rabbitmq():
    from src.prm.remote_client import RemotePRM, RemotePRMConfig

    queue = f"test.prm.{socket.gethostname()}.{int(time.time()*1000)}"

    def fake_handler(body):
        req = json.loads(body)
        assert req["op"] == "score"
        return json.dumps({"score": 0.7321}).encode()

    _start_fake_worker(queue, fake_handler)

    cli = RemotePRM(RemotePRMConfig(amqp_url=AMQP_URL, request_queue=queue, rpc_timeout=10))
    try:
        s = cli.score("x²=9 풀어라", "Step 1: x = ±3.\n")
        assert isinstance(s, torch.Tensor)
        assert abs(s.item() - 0.7321) < 1e-6
    finally:
        cli.close()


@broker_alive
def test_real_remote_mu_sample_through_rabbitmq():
    from src.rollout.remote_mu import RemoteMuConfig, RemoteMuSampler

    queue = f"test.mu.{socket.gethostname()}.{int(time.time()*1000)}"

    def fake_handler(body):
        req = json.loads(body)
        assert req["op"] == "sample"
        return json.dumps({"steps": [f"alt_{i}\n" for i in range(req["n"])]}).encode()

    _start_fake_worker(queue, fake_handler)

    cli = RemoteMuSampler(RemoteMuConfig(amqp_url=AMQP_URL, request_queue=queue, rpc_timeout=10))
    try:
        steps = cli.sample_step_batch("p", "prefix\n", n=4)
        assert len(steps) == 4
        assert steps[0].startswith("alt_")
    finally:
        cli.close()
