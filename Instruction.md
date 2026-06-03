# AIME Evaluation — 설치 및 실행 가이드

AIME 2023 / 2024 / 2025 데이터셋을 **Qwen2.5-1.5B-Instruct** 모델로 평가하고 **Wandb** 대시보드로 결과를 확인하는 파이프라인입니다.

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

브라우저에서 [wandb.ai](https://wandb.ai) 로그인 후 API 키를 붙여넣습니다.

---

## 3. 실행

### 기본 실행 (2023, 2024, 2025 전체)

```powershell
python main.py
```

### 특정 연도만 평가

```powershell
python main.py --years 2024 2025
```

### 주요 옵션

```powershell
python main.py \
  --years 2023 2024 2025 \
  --batch-size 4 \
  --max-new-tokens 2048 \
  --dtype bfloat16 \
  --wandb-project aime-qwen-eval \
  --wandb-entity <your-wandb-username> \
  --wandb-run-name "qwen1.5b-aime-full" \
  --output-dir results
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--years` | `2023 2024 2025` | 평가할 연도 (스페이스로 구분) |
| `--batch-size` | `4` | 배치 크기 (VRAM 여유 시 8로 증가 가능) |
| `--max-new-tokens` | `2048` | 모델 출력 최대 토큰 수 |
| `--dtype` | `bfloat16` | 모델 정밀도 (`bfloat16` / `float16` / `float32`) |
| `--temperature` | `0.0` | 생성 온도 (0 = greedy, 결정론적 평가) |
| `--do-sample` | off | 이 플래그를 추가하면 샘플링 활성화 |
| `--wandb-project` | `aime-qwen-eval` | Wandb 프로젝트 이름 |
| `--wandb-entity` | 없음 | Wandb 팀/개인 계정명 |
| `--wandb-run-name` | 없음 | 실험 이름 |
| `--output-dir` | `results` | JSON 결과 저장 경로 |
| `--seed` | `42` | 재현성을 위한 랜덤 시드 |

---

## 4. 출력 결과

### 콘솔 요약

```
==================================================
RESULTS SUMMARY
==================================================
  accuracy/2023/AIME_I                    0.0%
  accuracy/2023/AIME_II                   6.7%
  accuracy/2023/overall                   3.3%
  ...
  accuracy/overall                        5.0%
  total_correct                           5
  total_problems                          90
```

### JSON 파일

`results/aime_results.json` — 문제별 상세 결과 (문제 텍스트, 정답, 예측값, 모델 원본 출력)

### Wandb 대시보드

| 항목 | 내용 |
|---|---|
| 스칼라 메트릭 | 연도별 / 경시별 / 전체 정확도 |
| 문제 번호별 차트 | 난이도 곡선 (1번 쉬움 → 15번 어려움) |
| 연도별 차트 | AIME 2023 / 2024 / 2025 비교 |
| 상세 테이블 | 문제, 정답, 예측, 모델 출력 전체 |

---

## 5. GPU 설정 메모

RTX 5060 (Blackwell) 최적화 설정 (코드에 이미 적용됨):

- `dtype = bfloat16` — Blackwell 네이티브 정밀도
- `attn_implementation = "sdpa"` — PyTorch 내장 어텐션, Blackwell 빌드 없이 안정 동작
- `device_map = "auto"` — 모델 전체(~3 GB)를 VRAM에 탑재, CPU 개입 없음
- `padding_side = "left"` — 배치 생성 시 올바른 토큰 위치 보장

VRAM이 충분히 남는다면 `--batch-size 8`로 처리 속도를 높일 수 있습니다.

---

## 6. 파일 구조

```
DRL-TERM_PROJECT/
├── main.py            # 진입점
├── config.py          # 설정 + CLI 인자 파싱
├── dataset.py         # AIME 데이터셋 로딩 (복수 소스 fallback)
├── model.py           # Qwen2.5-1.5B 로드 + 배치 추론
├── evaluate.py        # 답 추출 + 채점 + Wandb 로깅
├── requirements.txt   # 의존성 목록
├── Instruction.md     # 이 문서
└── results/           # 평가 결과 JSON (자동 생성)
```
