"""Streamlit 실시간 대시보드 — 학습 중 metrics 시각화.

사용:
    uv pip install streamlit
    uv run streamlit run scripts/dashboard.py
    # 또는:
    uv run streamlit run scripts/dashboard.py -- --jsonl outputs/stage8_smoke/metrics.jsonl

브라우저: http://localhost:8501

기능:
    - reward / reward_std / kl / lr / grad_norm / completion_length 실시간 그래프
    - 5초마다 jsonl 재로드 (학습 진행 시 자동 갱신)
    - 최근 N step만 보기 옵션
    - metric 별 통계 (min/max/평균)

학습 PC + Docker 환경: 컨테이너 안에서 띄우지 말고 host에서 직접:
    streamlit run scripts/dashboard.py
    (outputs/는 host filesystem에 mount되어 있음)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st


# --- argparse via sys.argv (streamlit cli args after `--`)
def _parse_args():
    import sys
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        argv = sys.argv[idx + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None)   # None이면 outputs/*/metrics.jsonl 중 최근 수정된 거
    return ap.parse_args(argv)


_args = _parse_args()


def _auto_detect_jsonl() -> str:
    """outputs/*/metrics.jsonl 중 가장 최근 수정된 거 선택."""
    from pathlib import Path
    candidates = list(Path("outputs").glob("*/metrics.jsonl"))
    if not candidates:
        return "outputs/stage8_smoke/metrics.jsonl"  # fallback (있을 수도 없을 수도)
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


# args.jsonl이 None이면 자동 탐색
_DEFAULT_JSONL = _args.jsonl or _auto_detect_jsonl()


@st.cache_data(ttl=5)   # 5초 캐시 → auto-refresh
def load_metrics(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


@st.cache_data(ttl=5)
def load_samples(metrics_path: str) -> pd.DataFrame:
    """metrics.jsonl 과 같은 디렉토리의 samples.jsonl 읽기."""
    samples_path = Path(metrics_path).parent / "samples.jsonl"
    if not samples_path.exists():
        return pd.DataFrame()
    records = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


# ============================================================================
st.set_page_config(page_title="PAV-RL Dashboard", layout="wide", page_icon="📈")
st.title("PAV-RL 학습 대시보드")

# --- sidebar
with st.sidebar:
    st.header("설정")
    jsonl_path = st.text_input("metrics jsonl 경로", value=_DEFAULT_JSONL)
    auto_refresh = st.checkbox("5초마다 자동 갱신", value=True)
    last_n = st.slider("최근 N step만 보기 (0 = 전체)", 0, 1000, 0, step=50)
    st.markdown("---")
    st.caption("jsonl은 `src/train/callbacks.py:JsonlMetricsCallback`이 자동 적재")

df = load_metrics(jsonl_path)
samples_df = load_samples(jsonl_path)

if df.empty:
    st.warning(f"`{jsonl_path}` 에서 metrics을 읽을 수 없습니다. 학습이 한 번 이상 돌아야 생성됨.")
    st.stop()

if last_n > 0 and "step" in df.columns:
    df = df[df["step"] > df["step"].max() - last_n]

# --- summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 log 수", len(df))
col2.metric("현재 step", int(df["step"].max()) if "step" in df.columns else "-")
if "reward" in df.columns:
    col3.metric("최근 reward", f"{df['reward'].iloc[-1]:.3f}")
if "learning_rate" in df.columns:
    col4.metric("최근 LR", f"{df['learning_rate'].iloc[-1]:.2e}")

st.markdown("---")

# --- charts grid 2x3
def _chart(df, key, label, color=None):
    if key not in df.columns:
        st.info(f"`{key}` 데이터 없음")
        return
    sub = df[["step", key]].dropna()
    if sub.empty:
        st.info(f"`{key}` 데이터 없음")
        return
    st.line_chart(sub.set_index("step"), height=240)

cols = st.columns(3)
with cols[0]:
    st.subheader("Reward (PAV)")
    _chart(df, "reward", "reward")
with cols[1]:
    st.subheader("Reward std")
    _chart(df, "reward_std", "std")
with cols[2]:
    st.subheader("KL(π || π_ref)")
    _chart(df, "kl", "kl")

cols2 = st.columns(3)
with cols2[0]:
    st.subheader("Learning rate")
    _chart(df, "learning_rate", "lr")
with cols2[1]:
    st.subheader("Gradient L2 norm")
    _chart(df, "grad_norm", "norm")
with cols2[2]:
    st.subheader("Completion length")
    _chart(df, "completion_length", "tokens")

st.markdown("---")

# --- samples viewer (새로 추가)
if not samples_df.empty:
    st.header("📝 GRPO 샘플 상세 (문제 + Step별 보상)")
    
    # 최근 샘플 중 하나 선택
    latest_step = int(samples_df["step"].max())
    latest_samples = samples_df[samples_df["step"] == latest_step]
    
    sample_idx = st.selectbox(
        "샘플 선택",
        range(len(latest_samples)),
        format_func=lambda i: f"샘플 {i+1} (step {latest_step}, total_reward={latest_samples.iloc[i]['total_reward']:.3f})"
    )
    
    row = latest_samples.iloc[sample_idx]
    
    with st.container():
        st.subheader("📌 문제")
        st.markdown(f"```\n{row['problem']}\n```")
        
        st.subheader("🔢 Step별 풀이 및 보상")
        
        # Step별 보상 테이블
        step_data = []
        for h, (step_text, reward) in enumerate(zip(row["trajectory"], row["rewards"])):
            step_data.append({
                "Step": h + 1,
                "보상": f"{reward:+.3f}",
                "풀이": step_text.strip()[:200] + ("..." if len(step_text.strip()) > 200 else "")
            })
        
        st.dataframe(step_data, use_container_width=True, hide_index=True)
        
        # Step별 보상 바 차트
        import plotly.express as px
        fig = px.bar(
            x=[f"Step {i+1}" for i in range(len(row['rewards']))],
            y=row['rewards'],
            labels={"x": "Step", "y": "Reward"},
            title=f"Step별 보상 분포 (Total: {row['total_reward']:.3f})"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.caption(f"step {row['step']} | 총 보상: {row['total_reward']:.3f}")
else:
    st.info("samples.jsonl 이 아직 생성되지 않았습니다. 학습이 진행되면 1000 step마다 자동 생성됩니다.")

st.markdown("---")

# --- raw table
with st.expander("Raw metrics table"):
    st.dataframe(df.tail(50), use_container_width=True)

if auto_refresh:
    time.sleep(5)
    st.rerun()
