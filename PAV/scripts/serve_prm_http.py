"""PRM HTTP 서버 — FastAPI로 src.prm.PRM을 wrap.

추론 PC(3090 Ti)에서 띄워두면 training PC의 RemotePRM이 HTTP로 호출.

실행 (uvicorn 직접 또는 docker-compose):
    uv run python scripts/serve_prm_http.py --config configs/prm.yaml --port 8002

또는 uvicorn:
    PRM_CONFIG=configs/prm.yaml uv run uvicorn \
        scripts.serve_prm_http:app --host 0.0.0.0 --port 8002

엔드포인트:
    GET  /health
    POST /v1/score          {"problem": str, "solution_prefix": str}
                              -> {"score": float}
    POST /v1/score_batch    {"problem": str, "solution_prefixes": list[str]}
                              -> {"scores": list[float]}
    POST /v1/score_per_step {"problem": str, "solution": str}
                              -> {"per_step": list[float]}
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prm import load_prm

log = logging.getLogger("prm-http")

app = FastAPI(title="PAV PRM Server", version="1.0")
_prm = None  # lazy 적재 (start-up 이벤트 또는 첫 호출)


def _ensure_prm():
    """env PRM_CONFIG에서 yaml 경로 읽어 PRM 적재 (한 번만)."""
    global _prm
    if _prm is None:
        cfg_path = os.environ.get("PRM_CONFIG", "configs/prm.yaml")
        log.info(f"Loading PRM from {cfg_path}")
        _prm = load_prm(cfg_path)
        _prm._ensure_loaded()
        log.info(f"PRM ready: {_prm.cfg.name} ({_prm.cfg.model_id}) [{_prm.cfg.quantization}]")
    return _prm


@app.on_event("startup")
def _startup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
    )
    _ensure_prm()


# ----------------------------------------------------------------- schemas
class ScoreRequest(BaseModel):
    problem: str
    solution_prefix: str


class ScoreResponse(BaseModel):
    score: float


class ScoreBatchRequest(BaseModel):
    problem: str
    solution_prefixes: list[str] = Field(default_factory=list)


class ScoreBatchResponse(BaseModel):
    scores: list[float]


class ScorePerStepRequest(BaseModel):
    problem: str
    solution: str


class ScorePerStepResponse(BaseModel):
    per_step: list[float]


class HealthResponse(BaseModel):
    ok: bool
    name: str
    model_id: str
    quantization: str


# ----------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    prm = _ensure_prm()
    return HealthResponse(
        ok=True,
        name=prm.cfg.name,
        model_id=prm.cfg.model_id,
        quantization=prm.cfg.quantization,
    )


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    prm = _ensure_prm()
    try:
        s = prm.score(req.problem, req.solution_prefix)
        return ScoreResponse(score=float(s.item() if hasattr(s, "item") else s))
    except Exception as e:
        log.exception("score error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/score_batch", response_model=ScoreBatchResponse)
def score_batch(req: ScoreBatchRequest):
    prm = _ensure_prm()
    try:
        out = prm.score_batch(req.problem, req.solution_prefixes)
        return ScoreBatchResponse(scores=[float(x) for x in out.tolist()])
    except Exception as e:
        log.exception("score_batch error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/score_per_step", response_model=ScorePerStepResponse)
def score_per_step(req: ScorePerStepRequest):
    prm = _ensure_prm()
    try:
        out = prm.score_per_step(req.problem, req.solution)
        return ScorePerStepResponse(per_step=[float(x) for x in out])
    except Exception as e:
        log.exception("score_per_step error")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------- CLI entry
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/prm.yaml")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()

    os.environ["PRM_CONFIG"] = args.config

    import uvicorn

    uvicorn.run(
        "scripts.serve_prm_http:app",
        host=args.host,
        port=args.port,
        log_level="info",
        workers=1,  # GPU model이라 단일 worker (멀티는 의미 없음)
    )


if __name__ == "__main__":
    main()
