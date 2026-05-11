"""PRM Ray Actor entry — 워커 PC(5070급)에서 띄움.

Ray Head에 워커로 join 후 named actor(들)를 등록. 본체의 RayPRMClient가
`ray.get_actor("prm-actor", namespace="pav-rl")`로 핸들 획득.

실행 (워커 PC):
    1) Ray head로 worker join: ray start --address=<head>:6379
    2) 본 스크립트로 actor 등록:
       uv run python scripts/serve_prm_ray.py \
           --config configs/prm.yaml \
           --num-replicas 1 \
           --address <head>:6379

여러 PC에서 동일 명령 + 다른 --replica-id로 띄우면 named actor가 라운드로빈 분산됨.
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
    ap.add_argument("--config", default="configs/prm.yaml")
    ap.add_argument("--address", default="auto",
                    help="Ray head address (예: ray://head:10001 또는 head:6379, 'auto'=자동 감지)")
    ap.add_argument("--namespace", default="pav-rl")
    ap.add_argument("--actor-name", default="prm-actor",
                    help="named actor (replicas=1) 또는 prefix (replicas≥2 → prm-actor-0, -1, …)")
    ap.add_argument("--num-replicas", type=int, default=1,
                    help="이 노드에서 등록할 replica 수. GPU 1장당 1개 권장")
    ap.add_argument("--keep-alive", action="store_true", default=True,
                    help="actor 등록 후 프로세스 유지 (named actor의 lifetime 보장)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    log = logging.getLogger("prm-ray-entry")

    import ray
    from src.prm.ray_actor import get_actor_cls

    log.info(f"Connecting to Ray cluster at {args.address!r} (namespace={args.namespace})")
    ray.init(address=args.address, namespace=args.namespace, ignore_reinit_error=True)

    ActorCls = get_actor_cls()
    handles = []
    for i in range(args.num_replicas):
        name = args.actor_name if args.num_replicas == 1 else f"{args.actor_name}-{i}"
        # 비정상 종료로 남은 detached actor가 있으면 정리
        try:
            existing = ray.get_actor(name, namespace=args.namespace)
            log.warning(f"Killing existing actor: {name}")
            ray.kill(existing)
        except ValueError:
            pass
        log.info(f"Registering named actor: {name}")
        h = ActorCls.options(
            name=name,
            namespace=args.namespace,
            lifetime="detached",          # head 살아있는 한 actor 유지
            max_concurrency=4,            # 동시 RPC 처리 (prefetch=1 의미와 유사)
        ).remote(args.config)
        handles.append((name, h))

    # 모델 로드 완료 대기 (health 호출로 확인)
    for name, h in handles:
        log.info(f"Waiting for {name} model load...")
        info = ray.get(h.health.remote(), timeout=600)
        log.info(f"  {name} ready: {info}")

    log.info(f"All {args.num_replicas} PRM actor(s) registered.")
    if args.keep_alive:
        log.info("Keeping process alive — actor lifetime maintained. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            log.info("Stopping. Detached actors remain in cluster until killed.")


if __name__ == "__main__":
    main()
