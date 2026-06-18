#!/bin/bash
# Smoke test for bench_passatk
# Runs a minimal test with 5 problems to verify the setup works

set -e

echo "=========================================="
echo "Pass@K Benchmark - Smoke Test"
echo "=========================================="

# Configuration
MODEL_PATH="${MODEL_PATH:-./PAV-distribution-test-1/checkpoint-200}"
OUT_DIR="runs/smoke_test_$(date +%Y%m%d_%H%M%S)"
N_PROBLEMS=5
K=16
MICRO_N=4

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or create a symlink."
    echo "Example: MODEL_PATH=/path/to/model bash examples/smoke_test.sh"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUT_DIR"
echo "  Problems per dataset: $N_PROBLEMS"
echo "  K (samples): $K"
echo "  Micro-batch: $MICRO_N"
echo ""

# Create a temporary test dataset
echo "Creating test dataset..."
python3 -c "
import json
from pathlib import Path

# Create a small test dataset
test_problems = [
    {
        'id': 'test/0',
        'problem': 'What is 2 + 2?',
        'gold': '4',
        'level': None,
    },
    {
        'id': 'test/1',
        'problem': 'What is 3 * 4?',
        'gold': '12',
        'level': None,
    },
    {
        'id': 'test/2',
        'problem': 'If x = 5, what is x + 10?',
        'gold': '15',
        'level': None,
    },
    {
        'id': 'test/3',
        'problem': 'What is the square root of 16?',
        'gold': '4',
        'level': None,
    },
    {
        'id': 'test/4',
        'problem': 'What is 100 / 25?',
        'gold': '4',
        'level': None,
    },
]

# Save to temporary file
out_dir = Path('$OUT_DIR')
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / 'test.jsonl', 'w') as f:
    for p in test_problems:
        f.write(json.dumps(p) + '\n')

print(f'Created {len(test_problems)} test problems')
"

# Run the benchmark
echo ""
echo "Running benchmark..."
echo ""

python3 -m bench_passatk.run_bench \
    --model_path "$MODEL_PATH" \
    --backend vllm \
    --datasets test \
    --k $K \
    --micro_n $MICRO_N \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 512 \
    --seed 42 \
    --out_dir "$OUT_DIR"

# Check results
echo ""
echo "=========================================="
echo "Results:"
echo "=========================================="

if [ -f "$OUT_DIR/report.md" ]; then
    echo ""
    cat "$OUT_DIR/report.md"
    echo ""
fi

echo ""
echo "Smoke test completed!"
echo "Results saved to: $OUT_DIR"
echo ""

# Verify metrics
python3 -c "
import json
from pathlib import Path

results_file = Path('$OUT_DIR') / 'test.jsonl'

if not results_file.exists():
    print('Error: Results file not found')
    exit(1)

with open(results_file) as f:
    results = [json.loads(line) for line in f]

print(f'Processed {len(results)} problems')

for r in results[:3]:
    pid = r['problem_id']
    n = r['per_problem']['n']
    c = r['per_problem']['c']
    p1 = r['per_problem'].get('pass@1', 'N/A')
    print(f'  {pid}: n={n}, c={c}, pass@1={p1:.4f}' if isinstance(p1, float) else f'  {pid}: n={n}, c={c}, pass@1={p1}')

print('')
print('✓ Smoke test passed!')
"