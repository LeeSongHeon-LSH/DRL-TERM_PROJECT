"""TrainerCallback — PAV-specific 로깅 + 함정 모니터링.

reward_fn이 매 forward에서 채워주는 self.last_stats 큐를 읽어 wandb로 집계해 보냄.
계획서 §8 / §11의 다음 항목을 커버:
  - pav/A_mean, A_std, A_q05, A_q95
  - pav/p_q_mean, pav/p_v_mean
  - pav/correlation_Q1_Q3 (분포가 의미있는지)
  - 1k step마다 sample 5개 dump (trivial step 함정)
"""
from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reward_fn import PAVRewardFn

log = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except Exception:  # transformers가 미설치일 경우의 fallback (테스트용)
    class TrainerCallback:  # type: ignore
        pass


class PAVMonitorCallback(TrainerCallback):
    """PAV reward 통계를 누적하고 wandb로 dump.

    PAVRewardFn에 stats_buffer/sample_buffer를 attach하여,
    매 reward 계산마다 자동으로 푸시되는 dict를 읽음.
    """

    def __init__(self, reward_fn: "PAVRewardFn", dump_every: int = 1000, buf_size: int = 4096):
        self.reward_fn = reward_fn
        self.dump_every = max(1, dump_every)
        # reward_fn이 push할 큐 — Inplace로 attach
        reward_fn.stats_buffer = deque(maxlen=buf_size)
        reward_fn.sample_buffer = deque(maxlen=buf_size)

    # ------------------------------------------------------------- callbacks
    def on_log(self, args, state, control, logs=None, **kwargs):
        """trainer.log() 시점마다 PAV 통계를 추가 보내기."""
        stats = list(self.reward_fn.stats_buffer)
        if not stats:
            return
        agg = self._aggregate(stats)
        try:
            import wandb  # noqa: F401
            if wandb.run is not None:
                wandb.log({f"pav/{k}": v for k, v in agg.items()}, step=state.global_step)
        except ImportError:
            pass
        # logs에도 같이 넣어 progress bar에 표시
        if logs is not None:
            for k, v in agg.items():
                logs[f"pav/{k}"] = v
        self.reward_fn.stats_buffer.clear()

    def on_step_end(self, args, state, control, **kwargs):
        """주기적으로 sample을 덤프 — trivial-step 붕괴 검사용."""
        if state.global_step == 0 or state.global_step % self.dump_every != 0:
            return
        samples = list(self.reward_fn.sample_buffer)[-5:]
        if not samples:
            return
        log.info(f"[PAVMonitor step={state.global_step}] {len(samples)} sample dump:")
        for i, (problem, traj, rewards) in enumerate(samples):
            log.info(f"  --- sample {i} (R_sum={sum(rewards):.3f}) ---")
            log.info(f"  Q: {problem[:120]!r}")
            for h, (s, r) in enumerate(zip(traj, rewards)):
                log.info(f"    step {h} (r={r:+.3f}): {s.strip()[:140]!r}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {"pav/sample_dump": wandb.Table(
                        columns=["step", "problem", "trajectory", "rewards"],
                        data=[[state.global_step, p, "\n---\n".join(t), str(r)]
                              for p, t, r in samples],
                    )},
                    step=state.global_step,
                )
        except ImportError:
            pass

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _aggregate(stats: list[dict]) -> dict[str, float]:
        import statistics

        keys = ("A_mean", "A_std", "A_q05", "A_q95", "p_q", "p_v")
        out: dict[str, float] = {}
        for k in keys:
            vals = [s[k] for s in stats if k in s and s[k] is not None]
            if vals:
                out[k] = float(sum(vals) / len(vals))

        # Q1 vs Q3 correlation (분포가 의미있는지) — 같은 수의 표본일 때만
        q1 = [s.get("Q1") for s in stats if s.get("Q1") is not None]
        q3 = [s.get("Q3") for s in stats if s.get("Q3") is not None]
        if len(q1) == len(q3) and len(q1) > 5:
            try:
                out["corr_Q1_Q3"] = float(_pearson(q1, q3))
            except statistics.StatisticsError:
                pass
        out["n_steps"] = float(len(stats))
        return out


def _pearson(xs, ys) -> float:
    import statistics

    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)
