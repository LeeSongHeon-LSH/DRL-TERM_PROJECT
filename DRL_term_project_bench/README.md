# DRL Term Project — PAV (Pass@K-Aligned Verification) Benchmark Suite

강화학습 기반 수학 추론 모델 평가를 위한 Pass@K 벤치마크 및 분석 도구 모음입니다.

본 프로젝트는 **PAV-distribution**, **PAV-scalar-c2**, **Qwen2.5-Math-1.5B (Baseline)** 세 모델을 AIME 2023/2024/2025 및 MATH 데이터셋에서 Pass@K, Majority@K, Oracle 지표로 평가하고, 통계적 비교 및 분포 분석을 수행하는 완전한 파이프라인을 제공합니다.

---

## 프로젝트 구조

```
DRL_term_project_bench/
├── README.md                          # 이 파일 (프로젝트 전체 소개)
├── .gitignore
├── run_aime_bench.sh                  # AIME 2023/2024/2025 Pass@K 벤치마크 실행 스크립트 (3모델)
├── run_aime_rerun.sh                  # 재현성 검증용 리런 스크립트 (seed=42, K=256)
└── bench_passatk/                     # Pass@K 벤치마크 핵심 패키지
    ├── README.md                      # 벤치마크 영문 매뉴얼
    ├── README_KO.md                   # 벤치마크 한국어 매뉴얼
    ├── requirements.txt                # Python 의존성
    ├── __init__.py
    ├── __main__.py
    │
    ├── run_bench.py                    # 메인 벤치마크 실행 엔진
    ├── compare.py                      # Base vs Trained 모델 비교 리포트
    ├── extract_solutions.py            # Few-shot vs Zero-shot 솔루션 텍스트 비교 추출
    ├── analyze_distribution.py         # 분포 서명 분석 (답안 다양성 등)
    ├── analyze_rerun.py               # 리런 재현성 분석
    ├── combine_aime.py                 # AIME 2023/2024/2025 통합 (89문제)
    ├── four_model_report.py            # 4모델 비교 리포트 생성
    ├── paired_compare.py               # Paired 통계 검정 (McNemar / 부호 검정)
    ├── plot_3model.py                  # 3모델 Pass@K 차트 생성
    ├── plot_passk.py                   # Pass@K 비교 플롯 생성
    ├── resample_pass1.py               # Pass@1 Bootstrap 리샘플링 분석
    ├── run_comparison.sh               # 비교 실행 스크립트
    │
    ├── datasets/                       # 데이터셋 로더
    │   ├── aime.py                     #   AIME 2023/2024/2025 로더
    │   ├── math.py                     #   MATH 데이터셋 로더
    │   ├── olympiad.py                  #   OlympiadBench 로더
    │   └── test.py                     #   테스트용 샘플 로더
    │
    ├── eval/                           # 평가 모듈
    │   ├── grader.py                   #   답안 추출 및 채점 (\boxed{}, sympy)
    │   └── metrics.py                  #   Pass@K, Majority@K, Oracle 지표 계산
    │
    ├── samplers/                       # 샘플링 백엔드
    │   ├── vllm_sampler.py             #   vLLM 고속 배치 생성
    │   └── hf_sampler.py               #   HuggingFace transformers 백엔드 (fallback)
    │
    ├── utils/                          # 유틸리티
    │   ├── io.py                       #   결과 저장/로드 I/O
    │   └── seeding.py                  #   재현성을 위한 시드 관리
    │
    ├── tests/
    │   └── test_metrics.py             #   지표 계산 단위 테스트
    │
    └── examples/
        └── smoke_test.sh               #   빠른 동작 확인용 스크립트
```

---

## 주요 구성 요소 설명

### 1. 벤치마크 실행 엔진 (`bench_passatk/run_bench.py`)

모델의 수학 추론 성능을 Pass@K, Majority@K, Oracle 지표로 평가하는 메인 스크립트입니다.

- **Pass@K (편향 없는 추정치)**: K개 샘플 중 적어도 하나가 정답일 확률 (Codex 논문 기반)
- **Majority@K (자기 일치성)**: K개 샘플의 다수결 답안 정확도
- **Oracle Pass@K**: K개 샘플 중 하나라도 정답이 있는지 (상한)

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

### 2. 데이터셋 로더 (`bench_passatk/datasets/`)

| 파일 | 역할 |
|------|------|
| `aime.py` | AIME 2023/2024/2025 문제 로드 (연도별 분리 지원) |
| `math.py` | HuggingFace MATH 데이터셋 로드 |
| `olympiad.py` | OlympiadBench 데이터셋 로드 |
| `test.py` | 빠른 동작 확인용 샘플 데이터 |

### 3. 평가 모듈 (`bench_passatk/eval/`)

| 파일 | 역할 |
|------|------|
| `grader.py` | `\boxed{}` 답안 추출, `math_verify` / SymPy 기호 동등성 / 문자열 정규화 3단계 채점 |
| `metrics.py` | Pass@K 편향 없는 추정치, Pass@1, Majority@K, Best-of-N 계산 |

### 4. 샘플링 백엔드 (`bench_passatk/samplers/`)

| 파일 | 역할 |
|------|------|
| `vllm_sampler.py` | vLLM 기반 고속 배치 생성 (권장, 마이크로 배치로 메모리 효율적) |
| `hf_sampler.py` | HuggingFace transformers 백엔드 (vLLM 미지원 시 fallback) |

### 5. 분석 스크립트 (`bench_passatk/` 상단)

| 파일 | 역할 |
|------|------|
| `compare.py` | Base 모델과 Trained 모델의 벤치마크 결과를 비교하는 마크다운 리포트 생성 |
| `combine_aime.py` | AIME 2023/2024/2025 연도별 결과를 89문제 통합 세트로 병합 후 재집계 |
| `four_model_report.py` | 4개 모델(PAV-dist+few-shot, PAV-dist, PAV-scalar-c2, Baseline) 통합 비교 리포트 |
| `paired_compare.py` | 동일 문제에 대한 paired 검정(McNemar / 부호 검정)으로 모델 간 유의미한 차이 검증 |
| `analyze_distribution.py` | 분포 서명 분석 — 답안 다양성, 정답 분포 형태 비교 (PAV-distribution의 분포적 보상 효과 검증) |
| `analyze_rerun.py` | 동일 시드 리런 결과와 원본의 재현성 분석 (문제 수준 정답 수 일치도, oracle flip) |
| `extract_solutions.py` | Few-shot 모델이 풀었지만 Zero-shot 모델이 실패한 문제의 솔루션 텍스트 추출 비교 |
| `resample_pass1.py` | K=256 샘플에서 1개씩 부트스트랩 리샘플링하여 Pass@1 분포 및 신뢰구간 추정 |
| `plot_passk.py` | 모델별 Pass@K 비교 플롯 생성 (AIME 연도별 서브플롯) |
| `plot_3model.py` | 4모델 차트에서 PAV-distribution(zero-shot) 제외한 3모델 버전 차트 생성 |

### 6. 유틸리티 (`bench_passatk/utils/`)

| 파일 | 역할 |
|------|------|
| `io.py` | JSONL 결과 저장/로드, 설정 메타데이터 관리 |
| `seeding.py` | Python `random`, NumPy, PyTorch 전체 난수 소스 시드 고정 (재현성 보장) |

### 7. 실행 스크립트 (루트)

| 파일 | 역할 |
|------|------|
| `run_aime_bench.sh` | 3개 모델에 대해 AIME 2023/2024/2025 Pass@K 벤치마크를 순차 실행 |
| `run_aime_rerun.sh` | 재현성 검증용 리런 (seed=42, K=256, 원본 로그 보존) |

---

## 설치

```bash
cd DRL_term_project_bench
pip install -r bench_passatk/requirements.txt
```

### GPU 요구사항

| K | 모델 크기 | GPU 메모리 | 권장 GPU |
|---|-----------|------------|----------|
| 256 | 7B | ~40GB | A100 40GB / A6000 |
| 256 | 14B | ~80GB | A100 80GB |
| 256 | 70B | ~160GB | 2× A100 80GB |

GPU 메모리가 부족한 경우 `--micro_n` 값을 줄여 사용하세요.

---

## 사용법

### 1. 단일 모델 벤치마크 실행

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --backend vllm \
    --datasets MATH,AIME2024 \
    --k 256 \
    --micro_n 32 \
    --out_dir runs/ckpt200_passat256/
```

### 2. 전체 AIME 벤치마크 (3모델)

```bash
bash run_aime_bench.sh
```

### 3. 재현성 리런

```bash
bash run_aime_rerun.sh
```

### 4. 결과 분석

```bash
# AIME 2023-2025 통합 리포트
python -m bench_passatk.combine_aime

# 4모델 비교 리포트
python -m bench_passatk.four_model_report

# Paired 통계 검정
python -m bench_passatk.paired_compare

# 분포 분석
python -m bench_passatk.analyze_distribution

# Pass@K 플롯
python -m bench_passatk.plot_passk
```

### 5. 중단된 실행 이어하기

```bash
python -m bench_passatk.run_bench \
    --model_path ./PAV-distribution-test-1/checkpoint-200 \
    --datasets MATH,AIME2024 \
    --k 256 \
    --resume \
    --out_dir runs/ckpt200_passat256/
```

---

## 출력 형식

### 디렉토리 구조

```
runs/ckpt200_passat256/
├── config.yaml          # 설정 메타데이터
├── MATH.jsonl           # MATH 결과 (문제당 1줄)
├── AIME2024.jsonl       # AIME 2024 결과
└── report.md            # 요약 리포트 (Wilson 신뢰구간 포함)
```

### JSONL 형식 (문제당 1줄)

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

---

## 지표 설명

### Pass@K (편향 없는 추정치)

$$\text{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

- $n$: 총 샘플 수, $c$: 정답 샘플 수, $k$: 고려하는 샘플 수
- 단순한 $\frac{c}{k}$ 방식의 과대 추정을 피하는 편향 없는 추정치 (Chen et al., 2021)

### Majority@K (자기 일치성)

$K$개 샘플 중 다수결로 선택된 답안의 정확도

### Oracle Pass@K

$K$개 샘플 중 적어도 하나가 정답인지 여부 (Pass@K의 상한)

---

## 테스트

```bash
cd bench_passatk
pytest tests/ -v
```

---

## 라이선스

본 프로젝트는 학술 연구 목적으로 작성되었습니다.