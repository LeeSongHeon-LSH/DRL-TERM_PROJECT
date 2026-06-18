#!/usr/bin/env bash
# PRM-Scalar 베이스라인 (C2) 실행 스크립트
#   C3와 동일한 mc_rollout + K=16, reducer만 Q1(mean)으로 바꾼 ablation.
#   기존 run_train.sh 수정 없이 별도 파일로 관리.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python scripts/03_grpo_train.py \
  --rl-config configs/rl_c2_scalar.yaml \
  "$@"
