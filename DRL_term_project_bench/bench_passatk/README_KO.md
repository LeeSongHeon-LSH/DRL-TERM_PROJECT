# Pass@K 수학 추론 벤치마크

Pass@K, Majority@K, Oracle 지표를 사용하여 수학 추론 모델을 평가하는 종합 벤치마크 스위트입니다.

## 개요

이 벤치마크는 PAV-distribution 또는 유사한 방법으로 학습된 모델을 수학 추론 작업에서 평가합니다. 다음을 구현합니다:

- **Pass@K (편향 없는 추정치)**: K개 샘플 중 적어도 하나가 정답일 확률. Codex 논문의 편향 없는 추정치를 사용합니다.
- **Majority@K (자기 일치성)**: K개 샘플 중 다수결로 선택된 답안의 정확도.
- **Oracle Pass@K**: Pass@K의 상한 (K개 샘플 중 하나라도 정답인지 여부).

## 기능

- **다중 백엔드**: vLLM (빠름)과 HuggingFace transformers (대체) 지원.
- **다중 데이터셋**: MATH, AIME, OlympiadBench.
- **메모리 효율**: 큰 K 값을 위한 마이크로 배치 샘플링.
- **이어하기 지원**: 중단된 실행을 이어서 계속.
- **재현 가능**: 모든 난수 소스에 대한 결정적 시딩.
- **종합 보고서**: Wilson 신뢰구간이 포함된 마크다운 보고서.

## 설치

```bash
cd bench_passatk
pip install -r requirements.txt
```

### GPU 요구사항

| K | 모델 크기 | GPU 메모리 | 권장 GPU |
|---|-----------|------------|----------|
| 256 | 7B | ~40GB | A100 40GB / A6000 |
| 256 | 14B | ~80GB | A100 80GB |
| 256 | 70B | ~160GB | 2x A100 80GB |

GPU가 작은 경우 `--micro_n`을 사용하여 메모리 사용량을 줄이세요.

## 사용법

### 기본 사용법

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

### 중단된 실행 이어하기

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --datasets MATH,AIME2024 \
    --k 256 \
    --resume \
    --out_dir runs/ckpt200_passat256/
```

### HuggingFace 백엔드 사용 (대체)

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --backend hf \
    --datasets MATH \
    --k 64 \
    --out_dir runs/ckpt200_hf/
```

## 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model_path` | 필수 | 모델 체크포인트 디렉토리 경로 |
| `--backend` | `vllm` | 백엔드: `vllm` 또는 `hf` |
| `--datasets` | `MATH` | 쉼표로 구분된 데이터셋: `MATH`, `AIME2023`, `AIME2024`, `OlympiadBench` |
| `--k` | `256` | 문제당 총 샘플 수 |
| `--micro_n` | `32` | 마이크로 배치당 샘플 수 |
| `--temperature` | `0.7` | 샘플링 온도 |
| `--top_p` | `0.95` | Top-p 샘플링 |
| `--max_new_tokens` | `2048` | 최대 생성 길이 |
| `--seed` | `42` | 난수 시드 |
| `--out_dir` | `runs/default/` | 출력 디렉토리 |
| `--resume` | False | 이전 실행 이어하기 |
| `--tensor_parallel_size` | `1` | GPU 수 (vLLM) |
| `--gpu_memory_utilization` | `0.9` | GPU 메모리 비율 (vLLM) |
| `--use_wandb` | False | Weights & Biases 로깅 활성화 |
| `--wandb_project` | `passatk-benchmark` | wandb 프로젝트 이름 |
| `--wandb_entity` | None | wandb 엔티티 (사용자명 또는 팀) |
| `--wandb_run_name` | 자동 | wandb 실행 이름 |

## 출력

### 디렉토리 구조

```
runs/ckpt200_passat256/
├── config.yaml          # 설정 메타데이터
├── gsm8k.jsonl          # GSM8K 결과 (한 줄에 한 문제)
├── MATH.jsonl           # MATH 결과
├── AIME2024.jsonl       # AIME 결과
└── report.md            # 요약 보고서
```

### JSONL 형식

JSONL 파일의 각 줄은 다음을 포함합니다:

```json
{
  "problem_id": "gsm8k/123",
  "problem": "Janet's ducks lay 16 eggs...",
  "gold": "42",
  "samples": [
    {"text": "...", "pred": "42", "is_correct": true}
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

### 보고서 예시

```markdown
## GSM8K

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

## 지표 설명

### Pass@K (편향 없는 추정치)

Chen et al. (2021)에서, Pass@K의 편향 없는 추정치는:

$$\text{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

여기서 $n$은 총 샘플 수, $c$는 정답 샘플 수, $k$는 고려하는 샘플 수입니다.

이 추정치는 편향이 없으며 단순한 $\frac{c}{k}$ 접근 방식의 과대 추정을 피합니다.

### Majority@K

$K$개 샘플 중 다수결로 선택된 답안의 정확도입니다. 자기 일치성(self-consistency) 투표를 구현합니다.

### Oracle Pass@K

$K$개 샘플 중 적어도 하나가 정답인지 여부입니다. Pass@K의 상한입니다.

## 답안 추출

### GSM8K

`#### 숫자` 형식에서 마지막 숫자를 추출합니다.

### MATH / AIME / OlympiadBench

`\boxed{...}` 형식에서 답안을 추출하고 다음을 사용하여 비교합니다:
1. `math_verify` 라이브러리 (사용 가능한 경우)
2. SymPy 기호적 동등성
3. 문자열 정규화 대체

## 테스트 실행

```bash
cd bench_passatk
pytest tests/ -v
```

## Weights & Biases 통합

실시간 실험 추적과 시각화를 위해 wandb 로깅을 활성화하세요:

```bash
# wandb 로깅 활성화
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --backend vllm \
    --datasets gsm8k,MATH \
    --k 256 \
    --use_wandb \
    --wandb_project my-project \
    --wandb_run_name my-experiment
```

### wandb 기능

- **실시간 모니터링**: 긴 실행 중 진행 상황 추적
- **Pass@K 차트**: Pass@K 곡선 자동 시각화
- **실험 비교**: 여러 실행을 나란히 비교
- **팀 공유**: 링크를 통해 팀원과 결과 공유

### wandb 대시보드

`--use_wandb`로 실행 후 다음을 확인할 수 있습니다:
- Pass@K 곡선 (k vs 정확도)
- Best-of-N 정확도
- Majority@K (자기 일치성)
- Oracle 상한
- 문제별 지표

## 스모크 테스트

5개 문제로 구성된 최소 테스트:

```bash
bash examples/smoke_test.sh
```

## 예상 실행 시간

| 데이터셋 | 문제 수 | K | GPU | 시간 |
|----------|---------|---|-----|------|
| GSM8K | 1319 | 256 | A100 40GB | ~4시간 |
| MATH | 5000 | 256 | A100 40GB | ~15시간 |
| AIME | 30 | 256 | A100 40GB | ~10분 |