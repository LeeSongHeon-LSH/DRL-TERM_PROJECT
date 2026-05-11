"""μ RabbitMQ 워커 entry — 16GB+ GPU PC에서 띄움.

실행:
    uv run python scripts/serve_mu.py \
        --config configs/policy.yaml \
        --amqp-url amqp://guest:guest@<broker-host>:5672/ \
        --queue mu.requests
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rollout.mu_worker import WorkerSettings, serve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/policy.yaml")
    ap.add_argument("--amqp-url", default="amqp://guest:guest@localhost:5672/")
    ap.add_argument("--queue", default="mu.requests")
    ap.add_argument("--prefetch", type=int, default=1)
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
