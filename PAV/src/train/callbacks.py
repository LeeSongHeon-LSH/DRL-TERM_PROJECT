"""TrainerCallback — PAV-specific 로깅 + 함정 모니터링.

reward_fn이 매 forward에서 채워주는 self.last_stats 큐를 읽어 wandb로 집계해 보냄.
계획서 §8 / §11의 다음 항목을 커버:
  - pav/A_mean, A_std, A_q05, A_q95
  - pav/p_q_mean, pav/p_v_mean
  - pav/correlation_Q1_Q3 (분포가 의미있는지)
  - 1k step마다 sample 5개 dump (trivial step 함정)

추가:
  - JsonlMetricsCallback — 모든 trainer logs를 outputs/<run_name>/metrics.jsonl로 적재.
    scripts/plot_metrics.py로 그래프 생성, scripts/dashboard.py로 streamlit 인터랙티브 보기.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reward_fn import PAVRewardFn

log = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except Exception:  # transformers가 미설치일 경우의 fallback (테스트용)
    class TrainerCallback:  # type: ignore
        pass


class JsonlMetricsCallback(TrainerCallback):
    """매 logging_steps의 metrics을 jsonl 파일로 적재 — 분석/그래프용.

    각 line: {"step": N, "loss": ..., "reward": ..., ...}

    resume 시:
      - on_train_begin에서 state.global_step > 0이면 resume으로 간주 → 기존 jsonl 보존
      - fresh start면 (global_step == 0) jsonl backup 후 새로 시작
    """

    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 파일 제거는 on_train_begin에서 resume 여부 보고 결정

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step > 0:
            # resume — 기존 jsonl에서 step > global_step 인 데이터 truncate (중복 방지),
            # step <= global_step 까지만 남김. 이후 새 학습이 step+1 부터 append.
            if self.path.exists():
                kept: list[str] = []
                with self.path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            if d.get("step", 0) <= state.global_step:
                                kept.append(line)
                        except json.JSONDecodeError:
                            continue
                self.path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                log.info(f"JsonlMetricsCallback: resume from step {state.global_step}, "
                         f"truncated jsonl to {len(kept)} lines (step ≤ {state.global_step})")
            return
        # fresh start — 기존 파일 있으면 .bak으로 백업, 새 jsonl 시작
        if self.path.exists():
            backup = self.path.with_suffix(".jsonl.bak")
            try:
                self.path.rename(backup)
                log.info(f"JsonlMetricsCallback: fresh start, backed up old → {backup}")
            except OSError:
                self.path.unlink()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        record = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


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
