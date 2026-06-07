# AIME pass@k Evaluation — 설치 및 실행 가이드

AIME 2023 / 2024 / 2025 문제를 **Qwen2.5-1.5B-Instruct** 모델로 **연도별 분리 추론**하고,  
**pass@k** 지표를 계산한 뒤 **연도별 wandb 런 + 비교 대시보드**로 결과를 게시하는 파이프라인입니다.

---

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

1회 실행(연도 3개 모두)하면 wandb에 **4개의 런**이 생성됩니다.

| wandb 런 이름 | 내용 |
|---|---|
| `<base>-2023` | AIME 2023 pass@k 지표, 문제별 결과 테이블, 난이도 곡선 차트 |
| `<base>-2024` | AIME 2024 동일 구성 |
| `<base>-2025` | AIME 2025 동일 구성 |
| `<base>-comparison` | 세 연도 비교 테이블 + pass@k별 연도 비교 바 차트 |

`<base>`는 `--wandb-run-name` 값이며, 미지정 시 모델 이름 뒷부분(예: `Qwen2.5-1.5B-Instruct`)을 사용합니다.

---

## 6. pass@k 계산 방식

각 문제에 대해 N개 샘플을 생성하고, 아래 **불편 추정량**을 사용합니다.

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

## 9. GPU 설정 메모 (코드에 이미 적용됨)

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
