"""μ RabbitMQ 워커 — 큐에서 sample 요청 consume → vLLM rollout → reply queue로 publish.

배포:
    uv run python scripts/serve_mu.py --config configs/policy.yaml \
        --amqp-url amqp://guest:guest@<broker-host>:5672/ \
        --queue mu.requests
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .mu_handlers import handle_request

log = logging.getLogger("mu-worker")


@dataclass
class WorkerSettings:
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    request_queue: str = "mu.requests"
    prefetch: int = 1


def serve(policy_yaml: str | Path, settings: WorkerSettings):
    """μ 로드 + RabbitMQ consume loop."""
    import pika

    from .mu_sampler import build_mu_from_policy_yaml

    mu = build_mu_from_policy_yaml(policy_yaml)
    log.info(f"Loading μ weights ({mu.cfg.model_id})…")
    mu._ensure_loaded()
    log.info(f"Connecting to RabbitMQ at {settings.amqp_url}")

    conn = pika.BlockingConnection(pika.URLParameters(settings.amqp_url))
    channel = conn.channel()
    channel.queue_declare(queue=settings.request_queue, durable=False)
    channel.basic_qos(prefetch_count=max(1, settings.prefetch))

    def _on_message(ch, method, properties, body):
        reply_body = handle_request(mu, body)
        if properties.reply_to:
            ch.basic_publish(
                exchange="",
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(
                    correlation_id=properties.correlation_id,
                ),
                body=reply_body,
            )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(
        queue=settings.request_queue,
        on_message_callback=_on_message,
        auto_ack=False,
    )
    log.info(f"μ worker ready. queue='{settings.request_queue}' (prefetch={settings.prefetch})")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        log.info("Stopping μ worker…")
        channel.stop_consuming()
    finally:
        conn.close()
