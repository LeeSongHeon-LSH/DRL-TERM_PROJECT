#!/usr/bin/env bash
# PAV-RL one-shot launcher — A100 단일 호스트 가정 (PRM/μ local)
#
# 사용:
#   bash run_train.sh                     # 기본: smoke (Phase 0, 50 step)
#   bash run_train.sh --mode phase0       # Phase 0 본학습 (differential, ~5k step)
#   bash run_train.sh --mode phase1       # Phase 1 본학습 (mc_rollout K=16)
#   bash run_train.sh --mode phase1 --k 8 # Phase 1, K 축소로 속도 ↑
#   bash run_train.sh --skip-download     # 가중치가 이미 캐시에 있을 때
#   bash run_train.sh --skip-smoke        # PRM smoke 건너뛰기
#   bash run_train.sh --steps 1000        # 학습 step 수 override
#   bash run_train.sh --gpu-mem 0.55      # A100 80GB면 0.55 권장
#
# 환경변수 (선택):
#   HF_TOKEN      — HF rate-limit 회피
#   WANDB_API_KEY — 지정 시 wandb 로깅 활성
#   HF_HOME       — 모델 캐시 위치 (default ~/.cache/huggingface)

set -euo pipefail

# ---- 경로 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 기본값 ----
MODE="smoke"            # smoke | phase0 | phase1
K=16                    # MCRollout K (phase1만)
STEPS=""                # 빈 값이면 mode에 따라 결정
GPU_MEM=""              # 빈 값이면 yaml 값 사용
SKIP_DOWNLOAD=0
SKIP_SMOKE=0
SKIP_SANITY=0
N_SANITY=100

# ---- 인자 파싱 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)          MODE="$2"; shift 2 ;;
    --k|--K)         K="$2"; shift 2 ;;
    --steps)         STEPS="$2"; shift 2 ;;
    --gpu-mem)       GPU_MEM="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-smoke)    SKIP_SMOKE=1; shift ;;
    --skip-sanity)   SKIP_SANITY=1; shift ;;
    --n-sanity)      N_SANITY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *)
      echo "[err] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# mode별 default step
if [[ -z "$STEPS" ]]; then
  case "$MODE" in
    smoke)  STEPS=50 ;;
    phase0) STEPS=5000 ;;
    phase1) STEPS=5000 ;;
    *) echo "[err] --mode 는 smoke | phase0 | phase1"; exit 2 ;;
  esac
fi

# mode별 pav.method
case "$MODE" in
  smoke|phase0) PAV_METHOD="differential" ;;
  phase1)       PAV_METHOD="mc_rollout" ;;
esac

# ---- 0. 환경 점검 ----
echo "==[ 0. 환경 점검 ]=="
command -v uv >/dev/null || {
  echo "[err] uv가 없습니다. 설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

if ! command -v nvidia-smi >/dev/null; then
  echo "[warn] nvidia-smi 없음 — GPU 사용 불가 환경일 수 있음"
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# ---- 1. uv 환경 sync (base + gpu) ----
echo
echo "==[ 1. uv sync (base + gpu extras) ]=="
uv sync --extra gpu

# ---- 2. 모델 다운로드 ----
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo
  echo "==[ 2. HF 가중치 다운로드 (PRM 1.5B + Qwen2.5-Math-7B-Instruct) ]=="
  uv run python scripts/download_models.py
else
  echo
  echo "==[ 2. 다운로드 skip ]=="
fi

# ---- 3. PRM smoke ----
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  echo
  echo "==[ 3. PRM smoke (toy 4문장으로 forward 검증) ]=="
  uv run python scripts/00_smoke_prm.py --config configs/prm.yaml
else
  echo
  echo "==[ 3. PRM smoke skip ]=="
fi

# ---- 4. Sanity 라벨 + Phase 0 게이트 (phase0/phase1 본학습에만) ----
if [[ "$SKIP_SANITY" -eq 0 && "$MODE" != "smoke" ]]; then
  echo
  echo "==[ 4. Sanity 라벨 생성 + Phase 0 게이트 (G0) ]=="
  mkdir -p data
  if [[ ! -s data/sanity_items.jsonl ]]; then
    uv run python scripts/10_label_steps.py \
      --dataset gsm8k --n-problems "$N_SANITY" \
      --out data/sanity_items.jsonl
  else
    echo "  data/sanity_items.jsonl 존재 → 재사용"
  fi
  uv run python scripts/01_phase0_diff.py --items-jsonl data/sanity_items.jsonl
else
  echo
  echo "==[ 4. Sanity skip ]=="
fi

# ---- 5. RL config override 생성 ----
echo
echo "==[ 5. RL config 준비 (mode=$MODE, steps=$STEPS, K=$K) ]=="
RL_CONFIG="configs/rl_${MODE}_run.yaml"
uv run python - "$RL_CONFIG" "$PAV_METHOD" "$K" "$STEPS" "$GPU_MEM" "$MODE" <<'PY'
import sys, yaml, pathlib
out_path, method, K, steps, gpu_mem, mode = sys.argv[1:7]
base = yaml.safe_load(open("configs/rl_q3.yaml"))
base["pav"]["method"] = method
base["pav"]["K"] = int(K)
base["grpo"]["total_steps"] = int(steps)
if gpu_mem:
    base["vllm"]["gpu_memory_utilization"] = float(gpu_mem)
# run name 갱신
base.setdefault("logging", {})["wandb_run_name"] = f"{mode}_K{K}_s{steps}"
pathlib.Path(out_path).write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
print(f"  → {out_path} 작성")
PY

# ---- 6. wandb 로그인 (key 있을 때만) ----
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo
  echo "==[ 6. wandb login ]=="
  uv run wandb login --relogin "$WANDB_API_KEY" || echo "[warn] wandb login 실패 — stdout만 사용"
fi

# ---- 7. GRPO 학습 ----
echo
echo "==[ 7. GRPO 학습 시작 — mode=$MODE / steps=$STEPS ]=="
echo "  config: $RL_CONFIG"
echo "  로그/체크포인트: ./outputs/"
echo

uv run python scripts/03_grpo_train.py \
  --rl-config "$RL_CONFIG" \
  --prm-config configs/prm.yaml \
  --policy-config configs/policy.yaml

echo
echo "==[ 완료 ]=="
echo "다음 단계 — MathNet 평가:"
echo "  uv run python scripts/20_eval_mathnet.py --lora ./outputs/<run>/checkpoint-${STEPS} --N 64"
