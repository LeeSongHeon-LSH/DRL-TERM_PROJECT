"""Phase 0 ↔ Phase 1 swap 무중단 검증.

- Protocol 적합성 (isinstance(pav, PAVMethod))
- 동일 reducer (B1/Q1/Q3/Q4) 호환
- 동일 PAVRewardFn에 인스턴스만 갈아끼우면 동작
- 실제 PRM / μ 없이 가짜 객체로 빠르게 검증
"""
from __future__ import annotations

from pathlib import Path

import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pav import DifferentialPAV, MCRolloutPAV, PAVMethod, reduce_advantage
from src.train.reward_fn import PAVRewardFn


# ----------------------------------------------------------------- fakes
class FakePRM:
    """결정적 점수: zlib.crc32 기반 — Python 내장 hash()와 달리 PYTHONHASHSEED 영향 없음."""

    @staticmethod
    def _det_hash(problem: str, solution: str) -> float:
        import zlib
        h = zlib.crc32((problem + "||" + solution).encode("utf-8"))
        return (h % 10_000) / 10_000.0

    def score(self, problem: str, solution: str) -> torch.Tensor:
        return torch.tensor(self._det_hash(problem, solution))

    def score_batch(self, problem: str, solutions):
        return torch.stack([self.score(problem, s) for s in solutions])


class FakeMu:
    """K개 결정적 alternative step (해시로 다양성)."""

    def sample_step(self, problem: str, prefix: str) -> str:
        return self.sample_step_batch(problem, prefix, n=1)[0]

    def sample_step_batch(self, problem: str, prefix: str, n: int):
        return [f"alt_{i}: continuation of {prefix[-10:]}\n\n" for i in range(n)]


# ----------------------------------------------------------------- tests
def test_protocol_compliance():
    prm, mu = FakePRM(), FakeMu()
    pav0 = DifferentialPAV(prm)
    pav1 = MCRolloutPAV(prm, mu, K=8)
    assert isinstance(pav0, PAVMethod)
    assert isinstance(pav1, PAVMethod)
    assert pav0.name == "differential"
    assert pav1.name == "mc_rollout"


def test_output_shape_phase0():
    pav = DifferentialPAV(FakePRM())
    out = pav("problem", "prefix\n\n", "step\n\n")
    assert "advantage_scalar" in out
    assert out["advantage_samples"] is None
    assert out["advantage_scalar"].shape == ()


def test_output_shape_phase1():
    pav = MCRolloutPAV(FakePRM(), FakeMu(), K=8)
    out = pav("problem", "prefix\n\n", "step\n\n")
    assert out["advantage_samples"] is not None
    assert out["advantage_samples"].shape == (8,)
    assert out["advantage_scalar"].shape == ()


def test_reducer_handles_both():
    """동일 reducer가 스칼라/분포 둘 다 처리."""
    pav0 = DifferentialPAV(FakePRM())
    pav1 = MCRolloutPAV(FakePRM(), FakeMu(), K=16)
    out0 = pav0("p", "", "step\n\n")
    out1 = pav1("p", "", "step\n\n")
    for mode in ("B1", "Q1", "Q3", "Q4"):
        v0 = reduce_advantage(out0, mode=mode)
        v1 = reduce_advantage(out1, mode=mode)
        assert isinstance(v0, float)
        assert isinstance(v1, float)


def test_reward_fn_swap():
    """PAVRewardFn은 PAVMethod만 받음 — 인스턴스 교체로 swap 완료."""
    prm, mu = FakePRM(), FakeMu()
    traj = ["Step 1: a\n\n", "Step 2: b\n\n", "Step 3: c\n\n"]

    rf0 = PAVRewardFn(DifferentialPAV(prm), alpha=3.0, mode="Q1")
    rf1 = PAVRewardFn(MCRolloutPAV(prm, mu, K=8), alpha=3.0, mode="Q3", lam=-0.5)

    r_correct = rf0("problem", traj, final_correct=True)
    r_wrong   = rf0("problem", traj, final_correct=False)
    assert len(r_correct) == len(traj)
    # final_correct=True면 마지막 step에 R_ex=+1 추가 → 그 step만 정확히 1.0 더 큼
    assert abs((r_correct[-1] - r_wrong[-1]) - 1.0) < 1e-6
    assert all(abs(a - b) < 1e-6 for a, b in zip(r_correct[:-1], r_wrong[:-1]))
    # MCRolloutPAV swap도 동작
    r1 = rf1("problem", traj, final_correct=True)
    assert len(r1) == len(traj)
    assert all(isinstance(x, float) for x in r_correct + r1)


def test_q3_lambda_negative_increases_reward_with_spread():
    """Q3 (λ=−0.5)는 std에 보너스 — 동일 mean에서 분산 큰 분포의 reward가 더 큼."""
    out_low = {
        "advantage_scalar": torch.tensor(0.0),
        "advantage_samples": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    }
    out_hi = {
        "advantage_scalar": torch.tensor(0.0),
        "advantage_samples": torch.tensor([-1.0, -1.0, 1.0, 1.0]),
    }
    r_low = reduce_advantage(out_low, mode="Q3", lam=-0.5)
    r_hi = reduce_advantage(out_hi, mode="Q3", lam=-0.5)
    assert r_hi > r_low


if __name__ == "__main__":
    # pytest 없이도 빠르게 돌릴 수 있게
    import traceback
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    n_pass = 0
    for f in funcs:
        try:
            f()
            print(f"PASS  {f.__name__}")
            n_pass += 1
        except Exception:
            print(f"FAIL  {f.__name__}")
            traceback.print_exc()
    print(f"\n{n_pass}/{len(funcs)} passed")
