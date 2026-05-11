"""μ Ray Actor entry — 16GB+ GPU PC(예: PC B, RTX 3090)에서 띄움.

Ray Head에 worker로 join 후 named actor `mu-actor` 등록.

실행:
    ray start --address=<head>:6379
    uv run python scripts/serve_mu_ray.py --config configs/policy.yaml --address <head>:6379
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/policy.yaml")
    ap.add_argument("--address", default="auto")
    ap.add_argument("--namespace", default="pav-rl")
    ap.add_argument("--actor-name", default="mu-actor")
    ap.add_argument("--keep-alive", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    log = logging.getLogger("mu-ray-entry")

    import ray
    from src.rollout.ray_actor import get_actor_cls

    log.info(f"Connecting to Ray cluster at {args.address!r} (namespace={args.namespace})")
    ray.init(address=args.address, namespace=args.namespace, ignore_reinit_error=True)

    ActorCls = get_actor_cls()
    # 비정상 종료로 남은 detached actor 정리
    try:
        existing = ray.get_actor(args.actor_name, namespace=args.namespace)
        log.warning(f"Killing existing actor: {args.actor_name}")
        ray.kill(existing)
    except ValueError:
        pass
    log.info(f"Registering named actor: {args.actor_name}")
    h = ActorCls.options(
        name=args.actor_name,
        namespace=args.namespace,
        lifetime="detached",
        max_concurrency=4,
    ).remote(args.config)

    # vLLM init이 길 수 있어 polling 방식으로 (10분마다 진행 확인)
    log.info("Waiting for μ model load (vLLM init + KV cache, 첫 회 10~25분)...")
    import time as _time
    deadline = _time.time() + 1800   # 30분
    while _time.time() < deadline:
        try:
            info = ray.get(h.health.remote(), timeout=120)
            log.info(f"μ actor ready: {info}")
            break
        except Exception as e:
            log.info(f"  still loading... ({type(e).__name__})")
            _time.sleep(10)
    else:
        log.error("μ actor health check timed out — check worker logs")
        raise SystemExit(1)

    if args.keep_alive:
        log.info("Keeping process alive. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            log.info("Stopping. Detached actor remains until killed.")


if __name__ == "__main__":
    main()
