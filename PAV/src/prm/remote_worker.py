"""PRM RabbitMQ 워커 — 큐에서 요청 consume → score → reply queue로 publish.

배포 패턴:
    [본체] --publish--> prm.requests queue --consume--> [워커 1, 2, ...] --publish--> reply queue --> [본체]

여러 워커를 띄우면 RabbitMQ가 자동으로 라운드로빈 분산 (basic_qos prefetch=1로 GPU 1개당 1개 in-flight).

배포:
    uv run python scripts/serve_prm.py --config configs/prm.yaml \
        --amqp-url amqp://guest:guest@<broker-host>:5672/ \
        --queue prm.requests
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .handlers import handle_request

log = logging.getLogger("prm-worker")


@dataclass
class WorkerSettings:
    amqp_url: str = "amqp://guest:guest@localhost:5672/"
    request_queue: str = "prm.requests"
    prefetch: int = 1


def serve(config: str | Path, settings: WorkerSettings):
    """PRM 로드 + RabbitMQ consume loop."""
    import pika

    from .loader import load_prm

    prm = load_prm(config)
    log.info(f"Loading PRM weights ({prm.cfg.model_id})…")
    prm._ensure_loaded()
    log.info(f"Connecting to RabbitMQ at {settings.amqp_url}")

    conn = pika.BlockingConnection(pika.URLParameters(settings.amqp_url))
    channel = conn.channel()
    channel.queue_declare(queue=settings.request_queue, durable=False)
    channel.basic_qos(prefetch_count=max(1, settings.prefetch))

    def _on_message(ch, method, properties, body):
        reply_body = handle_request(prm, body)
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
    log.info(f"Worker ready. Listening on queue '{settings.request_queue}' (prefetch={settings.prefetch})")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        log.info("Stopping worker…")
        channel.stop_consuming()
    finally:
        conn.close()
