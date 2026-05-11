"""μ 워커 처리 로직 — transport 무관."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def handle_request(mu: Any, body: bytes | str) -> bytes:
    raw = body if isinstance(body, str) else body.decode("utf-8")
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        return _err(f"invalid JSON: {e}")

    op = req.get("op")
    try:
        if op == "sample":
            steps = mu.sample_step_batch(req["problem"], req["prefix"], n=req.get("n", 1))
            return _ok({"steps": list(steps)})

        if op == "health":
            return _ok({"ok": True, "model_id": mu.cfg.model_id})

        return _err(f"unknown op: {op!r}")
    except KeyError as e:
        return _err(f"missing field: {e}")
    except Exception as e:
        log.exception("μ handler error")
        return _err(f"{type(e).__name__}: {e}")


def _ok(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _err(msg: str) -> bytes:
    return json.dumps({"error": msg}).encode("utf-8")
