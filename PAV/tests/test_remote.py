"""RabbitMQ RPC 분산 — 핸들러 + 클라이언트 인터페이스 회귀.

실제 RabbitMQ broker는 띄우지 않음. 두 레이어로 검증:
  1) handlers.handle_request 직접 호출 — 워커 처리 로직 단위 테스트 (직렬화 / op 분기)
  2) RemotePRM/RemoteMuSampler의 _call을 monkeypatch — pika 없이 client 인터페이스 + PAVMethod 통합 검증

CI에서 외부 broker 의존성 없이 통과 가능.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pav.differential import DifferentialPAV
from src.pav.mc_rollout import MCRolloutPAV
from src.prm import handlers as prm_handlers
from src.prm.remote_client import RemotePRM, RemotePRMConfig
from src.rollout import mu_handlers as mu_handlers_mod
from src.rollout.remote_mu import RemoteMuConfig, RemoteMuSampler


# ----------------------------------------------------------------- fakes
class FakePRM:
    class _Cfg:
        name = "fake-prm"
        model_id = "fake/prm"
    cfg = _Cfg()

    @staticmethod
    def _det(s: str) -> float:
        import zlib
        return (zlib.crc32(s.encode()) % 10_000) / 10_000.0

    def score(self, problem: str, prefix: str) -> torch.Tensor:
        return torch.tensor(self._det(problem + "||" + prefix))

    def score_batch(self, problem: str, prefixes):
        return torch.stack([self.score(problem, p) for p in prefixes])

    def score_per_step(self, problem: str, solution: str):
        steps = [s for s in solution.split("\n") if s.strip()]
        return [self._det(problem + "||" + s) for s in steps]


class FakeMu:
    class _Cfg:
        model_id = "fake/mu"
    cfg = _Cfg()

    def sample_step_batch(self, problem: str, prefix: str, n: int):
        return [f"alt_{i}: cont {prefix[-5:]}\n" for i in range(n)]


# ----------------------------------------------------------------- Layer 1: handlers
def test_prm_handler_score():
    body = json.dumps({"op": "score", "problem": "p", "solution_prefix": "step\n"}).encode()
    out = json.loads(prm_handlers.handle_request(FakePRM(), body))
    assert "score" in out and isinstance(out["score"], float)


def test_prm_handler_score_batch():
    body = json.dumps({
        "op": "score_batch", "problem": "p",
        "solution_prefixes": ["a\n", "b\n", "c\n"],
    }).encode()
    out = json.loads(prm_handlers.handle_request(FakePRM(), body))
    assert "scores" in out and len(out["scores"]) == 3


def test_prm_handler_score_per_step():
    body = json.dumps({"op": "score_per_step", "problem": "p",
                       "solution": "step1\nstep2\nstep3\n"}).encode()
    out = json.loads(prm_handlers.handle_request(FakePRM(), body))
    assert len(out["per_step"]) == 3


def test_prm_handler_health():
    body = json.dumps({"op": "health"}).encode()
    out = json.loads(prm_handlers.handle_request(FakePRM(), body))
    assert out["ok"] is True and out["model_id"] == "fake/prm"


def test_prm_handler_unknown_op():
    body = json.dumps({"op": "nope"}).encode()
    out = json.loads(prm_handlers.handle_request(FakePRM(), body))
    assert "error" in out


def test_prm_handler_invalid_json():
    out = json.loads(prm_handlers.handle_request(FakePRM(), b"not json"))
    assert "error" in out


def test_mu_handler_sample():
    body = json.dumps({"op": "sample", "problem": "p", "prefix": "x", "n": 4}).encode()
    out = json.loads(mu_handlers_mod.handle_request(FakeMu(), body))
    assert len(out["steps"]) == 4


def test_mu_handler_health():
    body = json.dumps({"op": "health"}).encode()
    out = json.loads(mu_handlers_mod.handle_request(FakeMu(), body))
    assert out["ok"] is True


# ----------------------------------------------------------------- Layer 2: client + handler 통합
@pytest.fixture
def prm_client(monkeypatch):
    """RemotePRM이지만 _call을 worker handler에 직접 라우팅 — pika/broker 불필요."""
    cli = RemotePRM(RemotePRMConfig(amqp_url="amqp://test", request_queue="x"))
    fake_prm = FakePRM()

    def _call(op, payload):
        body = json.dumps({"op": op, **payload}).encode()
        resp_body = prm_handlers.handle_request(fake_prm, body)
        data = json.loads(resp_body)
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    monkeypatch.setattr(cli, "_call", _call)
    return cli


@pytest.fixture
def mu_client(monkeypatch):
    cli = RemoteMuSampler(RemoteMuConfig(amqp_url="amqp://test", request_queue="x"))
    fake_mu = FakeMu()

    def _call(op, payload):
        body = json.dumps({"op": op, **payload}).encode()
        resp_body = mu_handlers_mod.handle_request(fake_mu, body)
        data = json.loads(resp_body)
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    monkeypatch.setattr(cli, "_call", _call)
    return cli


def test_remote_prm_score_shape(prm_client):
    s = prm_client.score("p", "step1\n")
    assert isinstance(s, torch.Tensor) and s.shape == ()


def test_remote_prm_score_batch_shape(prm_client):
    s = prm_client.score_batch("p", ["a\n", "b\n", "c\n"])
    assert isinstance(s, torch.Tensor) and s.shape == (3,)


def test_remote_prm_score_per_step(prm_client):
    out = prm_client.score_per_step("p", "step1\nstep2\nstep3\n")
    assert isinstance(out, list) and len(out) == 3


def test_remote_prm_works_with_differential_pav(prm_client):
    pav = DifferentialPAV(prm_client)
    out = pav("problem", "prefix\n", "step\n")
    assert "advantage_scalar" in out and out["advantage_scalar"].shape == ()


def test_remote_mu_sample_step_batch(mu_client):
    steps = mu_client.sample_step_batch("p", "prefix", n=4)
    assert len(steps) == 4 and all(isinstance(s, str) for s in steps)


def test_remote_mu_works_with_mc_rollout_pav(prm_client, mu_client):
    pav = MCRolloutPAV(prm_client, mu_client, K=4)
    out = pav("p", "prefix\n", "step\n")
    assert out["advantage_samples"].shape == (4,) and out["advantage_scalar"].shape == ()
