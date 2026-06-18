# AIME pass@k Evaluation

> Qwen2.5-1.5B 로 AIME 2023 · 2024 · 2025 세 연도의 총 89문제를 pass@256 방식으로 풀어 평가한 코드입니다.

## 프로젝트 개요

본 프로젝트는 1.5B 규모의 소형 수학 특화 LLM이 미국 수학 경시대회 AIME(American Invitational
Mathematics Examination)의 2023·2024·2025년 문제(총 89문제)를 얼마나 풀 수 있는지를 pass@k
지표로 측정합니다. 문제당 256개의 풀이를 샘플링해 pass@256(256번 시도 중 1번이라도 정답을
맞히면 성공)까지 계산하는 것이 핵심입니다.

- 평가 대상 : AIME 2023 / 2024 / 2025, 총 89문제 (정답은 0–999 사이 정수)
- 핵심 지표 : 문제당 256개 풀이를 샘플링한 pass@256 + pass@1·2·4·…·128 마일스톤
- 추론 엔진 : vLLM continuous batching — `SamplingParams(n=256)` 으로 한 문제의 256개 풀이를 단일 호출로 생성
- 2단계 평가 : ① greedy 디코딩으로 결정론적 pass@1 측정 → ② temperature 샘플링으로 pass@1 … pass@256 측정
- 채점 방식 : 생성 결과에서 `\boxed{N}` 정답을 추출, 불편 추정량 `pass@k = 1 − C(N−c, k) / C(N, k)` 로 집계
- 로깅·시각화 : 모든 pass@k·연도별 분해 지표를 wandb에 기록하고, 결과 JSON 저장 및 논문용 matplotlib 그래프 생성

> 사용 모델은 `Qwen/Qwen2.5-1.5B-Instruct` (코드 기본값) 이며, `--model-name` 으로 교체할 수 있습니다.

### 결과 요약 (3개 연도 Combined · 문제당 256 samples)

| 지표 | 점수 |
|---|---|
| Greedy pass@1 | 3.3% |
| pass@1 (sampling) | 3.1% |
| pass@8 | 9.1% |
| pass@64 | 23.3% |
| pass@256 | 41.1% |

샘플 수 `k` 를 1 → 256으로 늘리면 정답률이 약 3%에서 41% 까지 상승합니다. 충분한 시도 기회를
주면 소형 모델도 상당수의 AIME 문제를 "풀어낼 수 있는 잠재력"을 가지고 있음을 보여줍니다.

## 코드 구성

| 파일 | 역할 |
|---|---|
| `main.py` | 진입점. GPU 점검 → 데이터 로드 → vLLM 모델 로드 → 2단계 평가 루프 → 채점 → wandb 로깅 → JSON 저장 |
| `config.py` | `EvalConfig` 설정 dataclass 와 CLI 인자 파싱 (`--num-samples`, `--years`, `--temperature`, `--model-name` 등) |
| `dataset.py` | AIME 2023/2024/2025 문제 로딩. 연도별 복수 HuggingFace 소스 fallback 후 `AIMEProblem` 으로 정규화 |
| `model.py` | vLLM 엔진 로드 및 추론. `build_chat_prompt`(chat 템플릿), `generate_greedy`, `generate_samples`(n=N 일괄 생성) |
| `evaluate.py` | 답 추출(`\boxed{N}` 우선), pass@k 불편 추정량 계산, wandb 테이블·차트 로깅, 결과 JSON 저장 |
| `plot_results.py` | Combined pass@k 스케일링 곡선과 요약 테이블 PNG 생성 (논문용) |
| `plot_aime_comparison.py` | 연도별 greedy pass@1·pass@k 비교 차트 PNG 생성 |
| `requirements.txt` | 의존성 목록 (torch, vllm, transformers, datasets, wandb 등) |

### 동작 흐름

1. `dataset.load_aime_problems()` 가 연도별 문제를 불러와 하나의 `AIMEProblem` 리스트로 합칩니다.
2. `model.load_model_and_tokenizer()` 가 vLLM 엔진을 띄우고 워밍업 1회를 수행합니다.
3. `eval_all()` 가 모든 문제에 대해 ① greedy 1회 → ② 256개 샘플을 생성하고, `extract_answer()` 로 정답 여부를 채점합니다.
4. `score_combined()` 가 pass@1 … pass@256 과 연도별 분해 지표를 계산합니다.
5. `log_combined_to_wandb()` 와 `save_combined_results()` 가 결과를 wandb 및 `results/*.json` 에 기록합니다.

---

# 설치 및 실행 가이드

프로그램 채점자 및 검증자를 위한 설치 및 실행 가이드입니다.

## 환경 요구사항

| 항목 | 사양 |
|---|---|
| CPU | AMD Ryzen 5 9600 |
| GPU | GeForce RTX 5060 (Blackwell, VRAM 8–12 GB) |
| CUDA | 12.x 이상 |
| Python | 3.10 이상 |

---

## 1. 의존성 설치

```powershell
pip install -r requirements.txt
```

주요 패키지:

| 패키지 | 버전 | 역할 |
|---|---|---|
| `torch` | ≥ 2.6.0 | Blackwell GPU 지원 |
| `transformers` | ≥ 4.47.0 | Qwen2.5 모델 로드 |
| `datasets` | ≥ 3.2.0 | AIME 데이터셋 로드 |
| `wandb` | ≥ 0.19.0 | 실험 대시보드 |
| `accelerate` | ≥ 1.2.0 | 자동 device 배치 |

---

## 2. Wandb 로그인 (최초 1회)

```powershell
wandb login
```

---

## 3. 실행

### 기본 실행 (2023 / 2024 / 2025, pass@256)

```powershell
python main.py
```

### 빠른 동작 확인 (샘플 수 줄이기)

```powershell
python main.py --num-samples 32 --pass-k-values 1 8 32
```

### 특정 연도만 평가

```powershell
python main.py --years 2024 2025
```

### greedy baseline (pass@1, 빠름)

```powershell
python main.py --no-sample
```

### wandb 팀/런 이름 지정

```powershell
python main.py --wandb-entity my-team --wandb-run-name qwen-1.5b
```

---

## 4. 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--num-samples N` | `256` | 문제당 생성 샘플 수 (pass@k 분모) |
| `--sample-batch-size N` | `16` | `model.generate()` 1회당 `num_return_sequences` (VRAM 절약) |
| `--pass-k-values K...` | `1 8 64 256` | 보고할 pass@k 값 목록 (`num-samples` 이하만 유효) |
| `--temperature T` | `0.7` | 샘플링 temperature |
| `--no-sample` | (미설정) | Greedy decoding (`num-samples` 강제 1) |
| `--years Y...` | `2023 2024 2025` | 평가할 연도 |
| `--model-name NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace 모델 경로 |
| `--dtype TYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--max-new-tokens N` | `2048` | 생성 최대 토큰 수 |
| `--wandb-project NAME` | `aime-qwen-eval` | wandb 프로젝트 이름 |
| `--wandb-entity NAME` | (없음) | wandb 팀/유저 이름 |
| `--wandb-run-name NAME` | (모델 이름 사용) | wandb 런 이름의 base |
| `--output-dir DIR` | `results` | JSON 결과 저장 폴더 |
| `--seed N` | `42` | 랜덤 시드 |

---

## 5. wandb 런 구조

1회 실행(연도 3개 모두)하면 wandb에 4개의 런이 생성됩니다.

| wandb 런 이름 | 내용 |
|---|---|
| `<base>-2023` | AIME 2023 pass@k 지표, 문제별 결과 테이블, 난이도 곡선 차트 |
| `<base>-2024` | AIME 2024 동일 구성 |
| `<base>-2025` | AIME 2025 동일 구성 |
| `<base>-comparison` | 세 연도 비교 테이블 + pass@k별 연도 비교 바 차트 |

`<base>`는 `--wandb-run-name` 값이며, 미지정 시 모델 이름 뒷부분(예: `Qwen2.5-1.5B-Instruct`)을 사용합니다.

---

## 6. pass@k 계산 방식

각 문제에 대해 N개 샘플을 생성하고, 아래 불편 추정량을 사용합니다.

```
pass@k = 1 - C(N-c, k) / C(N, k)
```

- `N` : 생성한 샘플 수 (`--num-samples`, 기본값 256)
- `c` : 그 중 정답인 샘플 수
- `k` : 평가할 k 값

pass@256 (k = N = 256) 문제 수준에서는 `c > 0`이면 `1.0`, `c = 0`이면 `0.0`으로 단순화됩니다.  
전체 pass@256 점수는 "256번 시도 중 1번이라도 맞힌 문제"의 비율입니다.

---

## 7. 출력 파일

```
results/
  aime_2023_results.json
  aime_2024_results.json
  aime_2025_results.json
```

각 JSON 구조:

```json
{
  "year": 2023,
  "metrics": {
    "pass@1/overall": 0.12,
    "pass@256/overall": 0.43,
    "pass@1/AIME_I": 0.10,
    ...
  },
  "results": [
    {
      "problem_id": "AIME_2023_I_01",
      "n_correct": 12,
      "n_samples": 256,
      "pass@1": 0.047,
      "pass@256": 1.0,
      "predicted_answers": [42, 17, 42, ...],
      "raw_outputs": ["...", "...", ...]
    },
    ...
  ]
}
```

---

## 8. 소요 시간 참고 (RTX 5060 / 12 GB VRAM 기준)

| 설정 | 연도당 예상 시간 |
|---|---|
| `--num-samples 256 --sample-batch-size 16` (기본값) | 약 2~4시간 |
| `--num-samples 64 --sample-batch-size 16` | 약 30~60분 |
| `--num-samples 32 --sample-batch-size 16` | 약 15~30분 |
| `--no-sample` (greedy) | 약 5분 |

VRAM이 여유롭다면 `--sample-batch-size 32`로 늘려 속도를 높일 수 있습니다.

---

## 9. GPU 설정 메모

- `dtype = bfloat16` — Blackwell 네이티브 정밀도
- `attn_implementation = "sdpa"` — PyTorch 내장 어텐션, Blackwell 빌드 없이 안정 동작
- `device_map = "auto"` — 모델 전체(~3 GB)를 VRAM에 탑재, CPU 개입 없음
- `padding_side = "left"` — `num_return_sequences` 배치 생성 시 올바른 토큰 위치 보장

---

## 10. 파일 구조

```
DRL-TERM_PROJECT/
├── main.py            # 진입점 — 연도별 추론 루프 + wandb 런 관리
├── config.py          # 설정 dataclass + CLI 인자 파싱
├── dataset.py         # AIME 데이터셋 로딩 (복수 소스 fallback)
├── model.py           # Qwen2.5 로드 + generate_samples() (pass@k용)
├── evaluate.py        # 답 추출 + pass@k 채점 + wandb 로깅
├── instruction.md     # 이 문서
└── results/           # 연도별 JSON 결과 (자동 생성)
```
