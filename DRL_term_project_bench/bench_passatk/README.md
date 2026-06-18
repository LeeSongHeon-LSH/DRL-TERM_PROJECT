# Pass@K Benchmark for Mathematical Reasoning

A comprehensive benchmark suite for evaluating mathematical reasoning models using Pass@K, Majority@K, and Oracle metrics.

## Overview

This benchmark evaluates models trained with PAV-distribution or similar methods on mathematical reasoning tasks. It implements:

- **Pass@K (unbiased estimator)**: The probability that at least one of K samples is correct, using the unbiased estimator from the Codex paper.
- **Majority@K (self-consistency)**: Accuracy of the majority-voted answer among K samples.
- **Oracle Pass@K**: Upper bound on Pass@K (whether any sample is correct).

## Features

- **Multiple Backends**: Supports vLLM (fast) and HuggingFace transformers (fallback).
- **Multiple Datasets**: MATH, AIME, OlympiadBench.
- **Memory Efficient**: Micro-batch sampling for large K values.
- **Resume Support**: Continue from interrupted runs.
- **Reproducible**: Deterministic seeding for all random sources.
- **Comprehensive Reports**: Markdown reports with Wilson confidence intervals.

## Installation

```bash
cd bench_passatk
pip install -r requirements.txt
```

### GPU Requirements

| K | Model Size | GPU Memory | Recommended GPU |
|---|------------|------------|-----------------|
| 256 | 7B | ~40GB | A100 40GB / A6000 |
| 256 | 14B | ~80GB | A100 80GB |
| 256 | 70B | ~160GB | 2x A100 80GB |

For smaller GPUs, use `--micro_n` to reduce memory usage.

## Usage

### Basic Usage

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --backend vllm \
    --datasets MATH,AIME2024 \
    --k 256 \
    --micro_n 32 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 2048 \
    --seed 42 \
    --out_dir runs/ckpt200_passat256/
```

### Resume Interrupted Run

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --datasets MATH,AIME2024 \
    --k 256 \
    --resume \
    --out_dir runs/ckpt200_passat256/
```

### Using HuggingFace Backend (Fallback)

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --backend hf \
    --datasets MATH \
    --k 64 \
    --out_dir runs/ckpt200_hf/
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to model checkpoint directory |
| `--backend` | `vllm` | Backend: `vllm` or `hf` |
| `--datasets` | `MATH` | Comma-separated datasets: `MATH`, `AIME2023`, `AIME2024`, `OlympiadBench` |
| `--k` | `256` | Total samples per problem |
| `--micro_n` | `32` | Samples per micro-batch |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.95` | Top-p sampling |
| `--max_new_tokens` | `2048` | Maximum generation length |
| `--seed` | `42` | Random seed |
| `--out_dir` | `runs/default/` | Output directory |
| `--resume` | False | Resume from previous run |
| `--tensor_parallel_size` | `1` | Number of GPUs (vLLM) |
| `--gpu_memory_utilization` | `0.9` | GPU memory ratio (vLLM) |
| `--use_wandb` | False | Enable Weights & Biases logging |
| `--wandb_project` | `passatk-benchmark` | wandb project name |
| `--wandb_entity` | None | wandb entity (username or team) |
| `--wandb_run_name` | Auto | wandb run name |

## Output

### Directory Structure

```
runs/ckpt200_passat256/
├── config.yaml          # Configuration metadata
├── MATH.jsonl           # MATH results
├── AIME2024.jsonl       # AIME results
└── report.md            # Summary report
```

### JSONL Format

Each line in the JSONL file contains:

```json
{
  "problem_id": "math/0",
  "problem": "How many square units...",
  "gold": "4.5",
  "samples": [
    {"text": "...", "pred": "4.5", "is_correct": true}
  ],
  "per_problem": {
    "n": 256,
    "c": 62,
    "pass@1": 0.242,
    "pass@256": 1.0,
    "maj@256": 1.0,
    "oracle": true
  }
}
```

### Report Example

```markdown
## MATH

### Pass@K

| k | Pass@k | 95% CI |
|---|--------|--------|
| 1 | 0.2420 | [0.2300, 0.2540] |
| 4 | 0.4521 | [0.4400, 0.4642] |
| 16 | 0.6892 | [0.6780, 0.7004] |
| 64 | 0.8723 | [0.8640, 0.8806] |
| 256 | 0.9654 | [0.9600, 0.9708] |

### Majority@K

| k | Accuracy | 95% CI |
|---|----------|--------|
| 1 | 0.2420 | [0.2300, 0.2540] |
| 4 | 0.3892 | [0.3770, 0.4014] |
| 16 | 0.5123 | [0.5000, 0.5246] |
| 64 | 0.5892 | [0.5770, 0.6014] |
| 256 | 0.6234 | [0.6110, 0.6358] |
```

## Metrics Explained

### Pass@K (Unbiased Estimator)

From Chen et al. (2021), the unbiased estimator for Pass@K is:

$$\text{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ is the total number of samples, $c$ is the number of correct samples, and $k$ is the number of samples considered.

This estimator is unbiased and avoids the overestimation of the naive $\frac{c}{k}$ approach.

### Majority@K

The accuracy of the majority-voted answer among $K$ samples. This implements self-consistency voting.

### Oracle Pass@K

Whether at least one of the $K$ samples is correct. This is an upper bound on Pass@K.

## Answer Extraction

### MATH / AIME / OlympiadBench

Extracts answers from `\boxed{...}` format and compares using:
1. `math_verify` library (if available)
2. SymPy symbolic equivalence
3. String normalization fallback

## Running Tests

```bash
cd bench_passatk
pytest tests/ -v
```

## Weights & Biases Integration

Enable wandb logging for real-time experiment tracking and visualization:

```bash
# Enable wandb logging
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --backend vllm \
    --datasets MATH,AIME2024 \
    --k 256 \
    --use_wandb \
    --wandb_project my-project \
    --wandb_run_name my-experiment
```

### wandb Features

- **Real-time monitoring**: Track progress during long runs
- **Pass@K charts**: Automatic visualization of Pass@K curves
- **Experiment comparison**: Compare multiple runs side-by-side
- **Team sharing**: Share results with team members via links

### wandb Dashboard

After running with `--use_wandb`, you can view:
- Pass@K curves (k vs accuracy)
- Best-of-N accuracy
- Majority@K (self-consistency)
- Oracle upper bound
- Per-problem metrics

## Smoke Test

A minimal test with 5 problems:

```bash
bash examples/smoke_test.sh
```

## Expected Runtime

| Dataset | Problems | K | GPU | Time |
|---------|----------|---|-----|------|
| MATH | 5000 | 256 | A100 40GB | ~15h |
| AIME | 30 | 256 | A100 40GB | ~10min |

## Troubleshooting

### Out of Memory

Reduce `--micro_n` or use tensor parallelism:

```bash
# Reduce micro-batch size
--micro_n 16

# Use multiple GPUs
--tensor_parallel_size 2
```

### Slow Generation

Ensure you're using vLLM backend:

```bash
--backend vllm
```

### Import Errors

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{bench_passatk,
  title={Pass@K Benchmark for Mathematical Reasoning},
  author={Your Name},
  year={2024}
}
```

## License

MIT License