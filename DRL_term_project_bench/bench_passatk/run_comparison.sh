#!/bin/bash
# Run benchmark on both base model and trained model, then compare results.
#
# Usage:
#   bash run_comparison.sh
#
# Environment variables:
#   BASE_MODEL_PATH: Path to base model (default: Qwen/Qwen2.5-Math-1.5B-Instruct)
#   TRAINED_MODEL_PATH: Path to trained model (default: ./PAV-distribution-test-1/checkpoint-200)
#   DATASETS: Datasets to evaluate (default: gsm8k,MATH,AIME2024,OlympiadBench)
#   K: Number of samples (default: 256)
#   MICRO_N: Micro-batch size (default: 32)

set -e

# Configuration
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-./PAV-distribution-test-1/checkpoint-200}"
DATASETS="${DATASETS:-gsm8k,MATH,AIME2024,OlympiadBench}"
K="${K:-256}"
MICRO_N="${MICRO_N:-32}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
SEED="${SEED:-42}"
BACKEND="${BACKEND:-vllm}"

# Output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="runs/base_model_${TIMESTAMP}"
TRAINED_DIR="runs/trained_model_${TIMESTAMP}"
COMPARISON_DIR="runs/comparison_${TIMESTAMP}"

echo "=========================================="
echo "Pass@K Benchmark Comparison"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL_PATH"
echo "  Trained Model: $TRAINED_MODEL_PATH"
echo "  Datasets: $DATASETS"
echo "  K (samples): $K"
echo "  Micro-batch: $MICRO_N"
echo "  Temperature: $TEMPERATURE"
echo "  Backend: $BACKEND"
echo ""
echo "Output:"
echo "  Base Model Results: $BASE_DIR"
echo "  Trained Model Results: $TRAINED_DIR"
echo "  Comparison Report: $COMPARISON_DIR/comparison_report.md"
echo ""

# Function to run benchmark
run_benchmark() {
    local model_path="$1"
    local out_dir="$2"
    local model_name="$3"
    
    echo ""
    echo "=========================================="
    echo "Running benchmark for: $model_name"
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
        --out_dir "$out_dir"
}

# Run base model benchmark
run_benchmark "$BASE_MODEL_PATH" "$BASE_DIR" "Base Model (Qwen2.5-Math-1.5B-Instruct)"

# Run trained model benchmark
run_benchmark "$TRAINED_MODEL_PATH" "$TRAINED_DIR" "Trained Model (PAV-distribution)"

# Compare results
echo ""
echo "=========================================="
echo "Generating comparison report"
echo "=========================================="
echo ""

mkdir -p "$COMPARISON_DIR"

python -m bench_passatk.compare \
    --base_dir "$BASE_DIR" \
    --trained_dir "$TRAINED_DIR" \
    --output "$COMPARISON_DIR/comparison_report.md"

# Print summary
echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Base Model: $BASE_DIR/report.md"
echo "  Trained Model: $TRAINED_DIR/report.md"
echo "  Comparison: $COMPARISON_DIR/comparison_report.md"
echo ""

# Display comparison summary
if command -v column &> /dev/null; then
    echo "Quick Summary:"
    echo ""
    head -20 "$COMPARISON_DIR/comparison_report.md"
fi