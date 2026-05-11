"""PRM RabbitMQ 워커 entry — 다른 PC(5070급)에서 띄움.

사전: docker-compose.yml로 RabbitMQ broker 1대 기동 (또는 외부 RabbitMQ).

실행:
    uv run python scripts/serve_prm.py \
        --config configs/prm.yaml \
        --amqp-url amqp://guest:guest@<broker-host>:5672/ \
        --queue prm.requests

여러 PC에서 동일 명령으로 띄우면 RabbitMQ가 자동 라운드로빈 분산.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prm.remote_worker import WorkerSettings, serve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/prm.yaml")
    ap.add_argument("--amqp-url", default="amqp://guest:guest@localhost:5672/")
    ap.add_argument("--queue", default="prm.requests")
    ap.add_argument("--prefetch", type=int, default=1,
                    help="동시 in-flight 요청 수. GPU 1개당 1개 권장 (vLLM batching이 처리)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    settings = WorkerSettings(
        amqp_url=args.amqp_url,
        request_queue=args.queue,
        prefetch=args.prefetch,
    )
    serve(args.config, settings)


if __name__ == "__main__":
    main()
