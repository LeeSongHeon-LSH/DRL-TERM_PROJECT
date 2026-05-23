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
    ap.add_argument("--jsonl", default="outputs/stage8_smoke/metrics.jsonl")
    return ap.parse_args(argv)


_args = _parse_args()


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


# ============================================================================
st.set_page_config(page_title="PAV-RL Dashboard", layout="wide", page_icon="📈")
st.title("PAV-RL 학습 대시보드")

# --- sidebar
with st.sidebar:
    st.header("설정")
    jsonl_path = st.text_input("metrics jsonl 경로", value=_args.jsonl)
    auto_refresh = st.checkbox("5초마다 자동 갱신", value=True)
    last_n = st.slider("최근 N step만 보기 (0 = 전체)", 0, 1000, 0, step=50)
    st.markdown("---")
    st.caption("jsonl은 `src/train/callbacks.py:JsonlMetricsCallback`이 자동 적재")

df = load_metrics(jsonl_path)

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

# --- raw table
with st.expander("Raw metrics table"):
    st.dataframe(df.tail(50), use_container_width=True)

if auto_refresh:
    time.sleep(5)
    st.rerun()
