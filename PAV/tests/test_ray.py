"""Ray 기반 분산 PRM/μ 클라이언트가 기존 PRM/MuSampler 인터페이스와 호환되는지.

실제 Ray cluster는 띄우지 않음. 두 레이어로 검증:
  1) PRMHandler 직접 호출 — transport-무관 처리 로직 (op 분기, 타입 변환)
  2) RayPRMClient의 _actor를 fake로 + ray.get monkeypatch — pika 없이 client 인터페이스 + PAVMethod 통합 검증

실제 Ray cluster 통합은 Stage R0~R7에서 검증.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pav.differential import DifferentialPAV
from src.pav.mc_rollout import MCRolloutPAV
from src.prm.ray_actor import PRMHandler
from src.prm.ray_client import RayPRMClient, RayPRMConfig
from src.rollout.ray_actor import MuHandler
from src.rollout.ray_client import RayMuClient, RayMuConfig


# ----------------------------------------------------------------- fake PRM
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


# ----------------------------------------------------------------- Layer 1: PRMHandler
def test_prm_handler_score():
    h = PRMHandler(FakePRM())
    s = h.score("p", "step\n")
    assert isinstance(s, float) and 0.0 <= s <= 1.0


def test_prm_handler_score_batch():
    h = PRMHandler(FakePRM())
    out = h.score_batch("p", ["a\n", "b\n", "c\n"])
    assert isinstance(out, list) and len(out) == 3 and all(isinstance(x, float) for x in out)


def test_prm_handler_score_per_step():
    h = PRMHandler(FakePRM())
    out = h.score_per_step("p", "step1\nstep2\nstep3\n")
    assert isinstance(out, list) and len(out) == 3


def test_prm_handler_health():
    h = PRMHandler(FakePRM())
    out = h.health()
    assert out["ok"] is True and out["model_id"] == "fake/prm"


# ----------------------------------------------------------------- Layer 2: RayPRMClient + fake actor
class _FakeRef:
    """ObjectRef stub — ray.get으로 풀어낼 수 있도록 value를 보관."""
    def __init__(self, value):
        self.value = value


class _FakeBoundMethod:
    """actor.method.remote(...) 호출을 흉내."""
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return _FakeRef(self.fn(*args, **kwargs))


class FakeActor:
    """Ray actor handle 흉내 — PRMHandler 메서드를 ray-style로 노출."""
    def __init__(self, handler: PRMHandler):
        self.score = _FakeBoundMethod(handler.score)
        self.score_batch = _FakeBoundMethod(handler.score_batch)
        self.score_per_step = _FakeBoundMethod(handler.score_per_step)
        self.health = _FakeBoundMethod(handler.health)


def _fake_ray_get(refs, timeout=None):
    if isinstance(refs, (list, tuple)):
        return [r.value for r in refs]
    return refs.value


@pytest.fixture
def prm_client(monkeypatch):
    import ray
    monkeypatch.setattr(ray, "get", _fake_ray_get)
    actor = FakeActor(PRMHandler(FakePRM()))
    return RayPRMClient(RayPRMConfig(), actor_handle=actor)


def test_ray_prm_client_score_shape(prm_client):
    s = prm_client.score("problem", "prefix\n")
    assert isinstance(s, torch.Tensor) and s.shape == ()


def test_ray_prm_client_score_empty_prefix(prm_client):
    # 빈 prefix는 RPC 안 보내고 0.5 fallback
    s = prm_client.score("problem", "")
    assert s.item() == 0.5


def test_ray_prm_client_score_batch_shape(prm_client):
    s = prm_client.score_batch("p", ["a\n", "b\n", "c\n", "d\n"])
    assert isinstance(s, torch.Tensor) and s.shape == (4,)


def test_ray_prm_client_score_per_step(prm_client):
    out = prm_client.score_per_step("p", "step1\nstep2\n")
    assert isinstance(out, list) and len(out) == 2


def test_ray_prm_client_health(prm_client):
    h = prm_client.health()
    assert h["ok"] is True


def test_ray_prm_client_works_with_differential_pav(prm_client):
    pav = DifferentialPAV(prm_client)
    out = pav("problem", "prefix\n", "step\n")
    assert "advantage_scalar" in out
    assert out["advantage_scalar"].shape == ()


def test_ray_prm_client_works_with_mc_rollout_pav(prm_client):
    # μ는 별도 테스트라 fake μ로 inline
    class FakeMu:
        def sample_step_batch(self, problem, prefix, n):
            return [f"alt_{i}\n" for i in range(n)]

    pav = MCRolloutPAV(prm_client, FakeMu(), K=4)
    out = pav("problem", "prefix\n", "step\n")
    assert out["advantage_samples"].shape == (4,) and out["advantage_scalar"].shape == ()


# ----------------------------------------------------------------- μ fakes
class FakeMu:
    class _Cfg:
        model_id = "fake/mu"
    cfg = _Cfg()

    def sample_step_batch(self, problem: str, prefix: str, n: int) -> list[str]:
        return [f"alt_{i}: continuation of {prefix[-5:]}\n" for i in range(n)]

    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, 1)[0]


# ----------------------------------------------------------------- Layer 1: MuHandler
def test_mu_handler_sample_step_batch():
    h = MuHandler(FakeMu())
    steps = h.sample_step_batch("p", "prefix", 4)
    assert isinstance(steps, list) and len(steps) == 4 and all(isinstance(s, str) for s in steps)


def test_mu_handler_sample_step():
    h = MuHandler(FakeMu())
    s = h.sample_step("p", "prefix")
    assert isinstance(s, str) and len(s) > 0


def test_mu_handler_health():
    h = MuHandler(FakeMu())
    out = h.health()
    assert out["ok"] is True and out["model_id"] == "fake/mu"


# ----------------------------------------------------------------- Layer 2: RayMuClient + fake actor
class FakeMuActor:
    """Ray μ actor handle 흉내."""
    def __init__(self, handler: MuHandler):
        self.sample_step = _FakeBoundMethod(handler.sample_step)
        self.sample_step_batch = _FakeBoundMethod(handler.sample_step_batch)
        self.health = _FakeBoundMethod(handler.health)


@pytest.fixture
def mu_client(monkeypatch):
    import ray
    monkeypatch.setattr(ray, "get", _fake_ray_get)
    actor = FakeMuActor(MuHandler(FakeMu()))
    return RayMuClient(RayMuConfig(), actor_handle=actor)


def test_ray_mu_client_sample_step_batch(mu_client):
    steps = mu_client.sample_step_batch("p", "prefix", n=4)
    assert isinstance(steps, list) and len(steps) == 4 and all(isinstance(s, str) for s in steps)


def test_ray_mu_client_sample_step(mu_client):
    s = mu_client.sample_step("p", "prefix")
    assert isinstance(s, str) and len(s) > 0


def test_ray_mu_client_health(mu_client):
    h = mu_client.health()
    assert h["ok"] is True


def test_ray_prm_plus_mu_clients_with_mc_rollout(prm_client, mu_client):
    """Phase 1 end-to-end: 두 Ray client가 MCRolloutPAV에 함께 들어감."""
    pav = MCRolloutPAV(prm_client, mu_client, K=8)
    out = pav("Solve x^2=9.", "prefix\n", "Step 1: x=±3.\n")
    assert out["advantage_samples"].shape == (8,)
    assert out["advantage_scalar"].shape == ()
    # p_q는 0-d tensor (단일 step PRM 점수)
    assert out["p_q"].dim() == 0
    # p_v_samples는 [K] tensor (K개 alternative의 PRM 점수)
    assert out["p_v_samples"].shape == (8,)
