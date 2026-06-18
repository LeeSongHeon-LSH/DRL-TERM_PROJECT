#!/bin/bash
# AIME 2023/2024/2025 Pass@K Benchmark for 3 models
# 
# Usage:
#   bash run_aime_bench.sh          # Run all models sequentially
#   bash run_aime_bench.sh 1        # Run only model 1 (Qwen baseline)
#   bash run_aime_bench.sh 2        # Run only model 2 (PAV-distribution)
#   bash run_aime_bench.sh 3        # Run only model 3 (PAV-scalar-c2)

set -e

# Common hyperparameters
DATASETS="AIME2023,AIME2024,AIME2025"
K=256
MICRO_N=32
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=2048
SEED=42
BACKEND="vllm"
DTYPE="bfloat16"

# Select which model(s) to run (default: all)
RUN="${1:-all}"

echo "=========================================="
echo "AIME Pass@K Benchmark (3 years × 3 models)"
echo "=========================================="
echo "Datasets: $DATASETS"
echo "K=$K, micro_n=$MICRO_N, temp=$TEMPERATURE, top_p=$TOP_P"
echo "Backend: $BACKEND, dtype: $DTYPE"
echo ""

run_model() {
    local model_path="$1"
    local out_dir="$2"
    local model_name="$3"

    echo ""
    echo "=========================================="
    echo "Model: $model_name"
    echo "Output: $out_dir"
    echo "=========================================="
    echo ""

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

# Model 1: Qwen baseline
if [[ "$RUN" == "all" || "$RUN" == "1" ]]; then
    run_model \
        "Qwen/Qwen2.5-Math-1.5B-Instruct" \
        "runs/qwen-math-1.5b-baseline" \
        "Qwen2.5-Math-1.5B-Instruct (baseline)"
fi

# Model 2: PAV-distribution checkpoint-500
if [[ "$RUN" == "all" || "$RUN" == "2" ]]; then
    run_model \
        "PAV-distribution-test/checkpoint-500" \
        "runs/pav-checkpoint-500" \
        "PAV-distribution-test/checkpoint-500"
fi

# Model 3: PAV-scalar-c2 checkpoint-500
if [[ "$RUN" == "all" || "$RUN" == "3" ]]; then
    run_model \
        "PAV-scalar-c2-test/checkpoint-500" \
        "runs/pav-scalar-c2-checkpoint-500" \
        "PAV-scalar-c2-test/checkpoint-500"
fi

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="