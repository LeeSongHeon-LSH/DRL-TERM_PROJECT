"""PRM 워커 처리 로직 — transport 무관.

RabbitMQ 워커(remote_worker.py)와 단위 테스트 둘 다 이 함수들을 직접 호출.
직렬화 포맷: JSON. 요청 body는 {"op": "...", ...} 형태.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def handle_request(prm: Any, body: bytes | str) -> bytes:
    """단일 요청 처리. 응답을 JSON bytes로 반환 — 워커가 reply queue로 publish."""
    raw = body if isinstance(body, str) else body.decode("utf-8")
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        return _err(f"invalid JSON: {e}")

    op = req.get("op")
    try:
        if op == "score":
            s = prm.score(req["problem"], req["solution_prefix"])
            return _ok({"score": _scalar(s)})

        if op == "score_batch":
            ss = prm.score_batch(req["problem"], req["solution_prefixes"])
            return _ok({"scores": [float(x) for x in ss.tolist()]})

        if op == "score_per_step":
            ps = prm.score_per_step(req["problem"], req["solution"])
            return _ok({"per_step": [float(x) for x in ps]})

        if op == "health":
            return _ok({
                "ok": True,
                "name": prm.cfg.name,
                "model_id": prm.cfg.model_id,
            })

        return _err(f"unknown op: {op!r}")
    except KeyError as e:
        return _err(f"missing field: {e}")
    except Exception as e:
        log.exception("handler error")
        return _err(f"{type(e).__name__}: {e}")


def _ok(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _err(msg: str) -> bytes:
    return json.dumps({"error": msg}).encode("utf-8")


def _scalar(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)
