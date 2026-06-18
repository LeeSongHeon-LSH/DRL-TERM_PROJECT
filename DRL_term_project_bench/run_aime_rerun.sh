#!/bin/bash
# AIME 2023/2024/2025 Pass@K RE-RUN (reproducibility check, seed=42, K=256).
# Writes to NEW *-rerun directories so the original logs are preserved.
#
# Usage:
#   bash run_aime_rerun.sh          # all three models sequentially
#   bash run_aime_rerun.sh 1|2|3    # only Qwen baseline | PAV-dist | PAV-scalar-c2

set -e

DATASETS="AIME2023,AIME2024,AIME2025"
K=256
MICRO_N=32
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=2048
SEED=42
BACKEND="vllm"
DTYPE="bfloat16"

RUN="${1:-all}"

echo "=========================================="
echo "AIME Pass@K RE-RUN (seed=$SEED, K=$K)  ->  *-rerun dirs"
echo "Datasets: $DATASETS"
echo "=========================================="

run_model() {
    local model_path="$1"; local out_dir="$2"; local model_name="$3"
    echo ""
    echo "=== Model: $model_name  ->  $out_dir ==="
    python -m bench_passatk.run_bench \
        --model_path "$model_path" \
        --backend "$BACKEND" \
        --datasets "$DATASETS" \
        --k "$K" \
        --micro_n "$MICRO_N" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_new_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --dtype "$DTYPE" \
        --resume \
        --out_dir "$out_dir"
}

if [[ "$RUN" == "all" || "$RUN" == "1" ]]; then
    run_model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
        "runs/qwen-math-1.5b-baseline-rerun" \
        "Qwen2.5-Math-1.5B-Instruct (baseline)"
fi

if [[ "$RUN" == "all" || "$RUN" == "2" ]]; then
    run_model "PAV-distribution-test/checkpoint-500" \
        "runs/pav-checkpoint-500-rerun" \
        "PAV-distribution-test/checkpoint-500"
fi

if [[ "$RUN" == "all" || "$RUN" == "3" ]]; then
    run_model "PAV-scalar-c2-test/checkpoint-500" \
        "runs/pav-scalar-c2-checkpoint-500-rerun" \
        "PAV-scalar-c2-test/checkpoint-500"
fi

echo ""
echo "=== RE-RUN complete ==="
