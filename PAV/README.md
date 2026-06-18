# PAV-RL — 차분 PAV + 분포형 보상 (Phase 0 + Phase 1)

천공 PRM(Skywork-o1-Open-PRM-Qwen-2.5-1.5B)을 활용해 PRM 추가 학습 없이 GRPO 학습까지 가는 파이프라인.
**현재 default: π / μ = Qwen2.5-Math-1.5B (Full FT)**, PRM = Skywork 1.5B int8.
7B + QLoRA로 확장은 yaml 키 변경만으로 가능 ([docs/QUICKSTART.md §6](docs/QUICKSTART.md#6-7b로-확장-옵션) 참고).

> **🚀 빠른 시작 (2 PC 분산)** — [docs/QUICKSTART.md](docs/QUICKSTART.md) 참고.
> 추론 PC: `docker compose -f docker-compose.inference.yml up -d` /
> 학습 PC: `.env`에 IP 채우고 `docker compose up -d`
>
> 단일 PC도 지원 — yaml에서 `mode: local`로 swap. (이전 RabbitMQ는 제거, 분산은 HTTP transport)

## 디렉토리

```
PAV/
├── pyproject.toml                # uv canonical (torch 2.5.1+cu124 + extras)
├── .python-version               # 3.11
├── configs/                      # prm.yaml, policy.yaml, rl_q3.yaml
├── src/
│   ├── prm/
│   │   ├── loader.py / score.py
│   │   └── skywork/              # vendored Skywork PRM_MODEL + io_utils
│   ├── pav/                      # PAVMethod Protocol + Differential / MCRollout + reducer
│   ├── rollout/                  # parser, μ sampler, vLLM π rollout
│   ├── train/
│   │   ├── reward_fn.py          # PAVRewardFn (+stats/sample buffer)
│   │   ├── policy_data.py        # Qwen2.5-Math + LoRA + MATH/GSM8K
│   │   ├── grpo_trainer.py       # TRL GRPOTrainer 빌더
│   │   └── callbacks.py          # PAVMonitorCallback (W&B + 함정 dump)
│   └── eval/                     # S1~S4 sanity, BoN-PAV
├── scripts/
│   ├── 00_smoke_prm.py
│   ├── 01_phase0_diff.py
│   ├── 02_phase1_mc.py
│   ├── 03_grpo_train.py
│   └── 10_label_steps.py         # MATH500 → sanity 라벨 jsonl
└── tests/                        # swap / parser / imports
```

## 핵심 설계 — `PAVMethod` Protocol

`src/pav/base.py`의 단일 Protocol만 만족하면 RL 코드(`PAVRewardFn`, GRPO trainer)는 수정 불필요.

```python
# Phase 0
pav = DifferentialPAV(prm)

# Phase 1 (메인 ⭐)
pav = MCRolloutPAV(prm, mu, K=16)

# 둘 다 동일 인터페이스로 사용
reward_fn = PAVRewardFn(pav, alpha=3.0, mode="Q3", lam=-0.5)
```

## 환경 구성 (uv)

```bash
cd PAV
uv sync                       # base: torch 2.5.1+cu124 + transformers + accelerate + datasets
uv sync --extra gpu           # +vllm 0.7 + trl 0.15 + peft + wandb + math-verify
uv sync --extra awq           # +autoawq (사용자가 별도 변환한 AWQ 가중치를 쓸 때)
uv sync --group dev           # +pytest + ruff
```

> CUDA 12.6 드라이버 + cu124 wheel = forward-compatible. PyTorch가 cu126 wheel for torch 2.5는 빌드하지 않아 cu124 사용.

## 빠른 검증 (PRM 다운로드 불필요)

```bash
uv run --group dev pytest tests/ -v
```

## 실행 — one-shot 런처 (권장)

```bash
bash run_train.sh                     # 기본: smoke (Phase 0, 50 step)
bash run_train.sh --mode phase0       # Phase 0 본학습 (differential, ~5k step)
bash run_train.sh --mode phase1       # Phase 1 본학습 (mc_rollout K=16)
bash run_train.sh --mode phase1 --k 8 # Phase 1, K 축소로 속도 ↑
bash run_train.sh --gpu-mem 0.55      # A100 80GB면 0.55 권장
```

## 실행 순서 (수동, 단계별)

```bash
# 사전 — 가중치 다운로드 (HF 캐시 또는 --local-dir 지정)
uv run python scripts/download_models.py                  # 기본: ~/.cache/huggingface
# 또는 한 모델만:
uv run python scripts/download_models.py --only prm
uv run python scripts/download_models.py --only policy

# W1 — Phase 0 (Sanity 라벨은 GSM8K test가 default)
uv run python scripts/00_smoke_prm.py --config configs/prm.yaml
uv run python scripts/10_label_steps.py --dataset gsm8k --n-problems 200 \
    --out data/sanity_items.jsonl
uv run python scripts/01_phase0_diff.py --items-jsonl data/sanity_items.jsonl

# W2 — Phase 1
uv run python scripts/02_phase1_mc.py --ks 4 8 16 32

# W3~W4 — GRPO 학습 (GSM8K) + MathNet 평가
uv run python scripts/03_grpo_train.py --rl-config configs/rl_q3.yaml
uv run python scripts/20_eval_mathnet.py \
    --lora ./outputs/PAV-distribution-fewshot-test/checkpoint-500 --N 64
```

## Docker로 띄우기 (단일 PC)

단일 trainer 컨테이너로 [Dockerfile](Dockerfile) + [docker-compose.yml](docker-compose.yml).

```bash
# 호스트에 NVIDIA Container Toolkit 한 번 설치
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

cp .env.example .env       # HF_HOME_HOST / HF_TOKEN / WANDB_API_KEY 채움
docker compose build
docker compose up -d
docker compose logs -f trainer
```

### 부가 정보
- **GPU 패스스루**: `deploy.resources.reservations.devices.driver: nvidia` (compose v2 표준)
- **HF 캐시 영속화**: `${HF_HOME_HOST}` (.env) → 컨테이너 `/cache/huggingface`. 모델은 컨테이너 안에 굽지 않음 → 첫 호출 시 lazy 다운, 호스트에 보존.
- **outputs**: trainer는 `./outputs`에 LoRA 체크포인트 저장 (호스트 bind mount).
- **shm_size**: trainer는 vLLM 내부 multiprocessing 때문에 8GB 권장 (compose에 이미 설정).

## 분산 셋업 (2 PC) — 학습 PC ↔ 추론 PC

π(Full FT)+μ+PRM을 24 GB GPU 한 장에 동시에 못 올릴 때(본 실험의 default).
**μ + PRM을 추론 서버로 분리**, π trainer만 단독 GPU 사용. 본 실험은 보유 **3090(학습)** + 클라우드 **T4 ×N(추론)** 구성.

```
┌────────────────────────────┐         ┌────────────────────────────┐
│ 학습 PC — RTX 3090 (24GB)  │   HTTP  │ 추론 — cloud T4 (16GB) ×N  │
│  docker-compose.yml        │ ◄────► │  docker-compose.inference  │
│   └ trainer container      │  (로드  │   ├ μ vLLM server  :8001   │
│                            │ 밸런서) │   └ PRM FastAPI    :8002   │
└────────────────────────────┘         └────────────────────────────┘
```

### 추론 PC 셋업

```bash
cp .env.inference.example .env
# 필요시 vi .env로 HF_HOME_HOST 등 설정
docker compose -f docker-compose.inference.yml build
docker compose -f docker-compose.inference.yml up -d
docker compose -f docker-compose.inference.yml logs -f
```

엔드포인트 확인:
```bash
curl http://localhost:8002/health           # PRM
curl http://localhost:8001/v1/models        # μ vLLM
```

### 학습 PC 셋업

```bash
cp .env.example .env
# .env에서 PRM_ENDPOINT / MU_ENDPOINT를 추론 PC IP로:
#   PRM_ENDPOINT=http://192.168.1.10:8002
#   MU_ENDPOINT =http://192.168.1.10:8001

docker compose build
docker compose up -d
docker compose logs -f
```

또는 docker 없이 직접:
```bash
PRM_ENDPOINT=http://192.168.1.10:8002 \
MU_ENDPOINT=http://192.168.1.10:8001 \
  bash run_train.sh --mode phase1
```

### 분산 모드 yaml 키

학습 PC의 [configs/prm.yaml](configs/prm.yaml):
```yaml
mode: remote
remote:
  endpoint: http://localhost:8002   # PRM_ENDPOINT 환경변수로 override 가능
  timeout: 120
```

학습 PC의 [configs/policy.yaml](configs/policy.yaml) `mu:` 섹션:
```yaml
mu:
  mode: remote
  remote:
    endpoint: http://localhost:8001   # MU_ENDPOINT 환경변수로 override 가능
    timeout: 180
```

### 분산 VRAM 예측 (1.5B Full FT default)

| PC | 구성 | VRAM | 마진 |
|---|---|---:|---:|
| **학습 PC (3090 24GB) ⭐** | **π 1.5B Full FT + adamw_bnb_8bit + vLLM colocate(0.20)** | **~17 GB** | **7 GB ✅** |
| **추론 cloud T4 (16GB) ⭐** | **μ 1.5B (fp16, gpu_mem=0.6) + PRM 1.5B int8** | **~8 GB** | **~8 GB ✅** |
| 학습 PC (7B 옵션) | π 7B + 4bit QLoRA + LoRA + vLLM colocate(0.20) | ~13 GB | 11 GB ✅ |
| 추론 (7B 옵션, ≥24GB 필요) | μ 7B (gpu_mem=0.65) + PRM 1.5B int8 | ~17 GB | T4 16GB 초과 ⚠ |

### 네트워크 부하

GRPO 1 step당 PC 간 트래픽 ~6 MB (PRM/μ RPC). **100 Mbps도 충분** (step당 ~0.5 s 오버헤드, 학습 1 step 30–60 s 대비 1–2%).
weight broadcast는 0 — π는 학습 PC 안에서 trainer + vLLM이 weight 공유.

## 단일 PC VRAM 추정 (3090 24GB)

| 컴포넌트 | VRAM |
|---|---|
| 학습 π 7B + LoRA + Adam states | ~14GB |
| vLLM colocate (rollout, gpu_mem_util=0.30) | ~7~9GB |
| PRM (Skywork 1.5B, fp16) | ~3GB |
| μ (Qwen2.5-Math-7B, bf16) | ~14GB |

→ Phase 1 K=16 + π/μ/PRM 동시 적재는 24GB 한 장으로 빡빡. 다음 중 선택:
- 정책을 1.5B로 다운그레이드해 모두 한 GPU에 올림
- Phase 0(`pav.method: differential`)만 학습 — μ 불필요
- `vllm.gpu_memory_utilization`을 더 낮추거나 K를 줄여(예: K=4) 메모리 절감

A100 80GB / H100 80GB에서는 모두 같이 적재 가능 (`--gpu-mem 0.55` 권장).

## 데이터셋

| 용도 | dataset | 비고 |
|------|------|------|
| 학습 | `openai/gsm8k` (main, train) | 단일 (이전 MATH는 옵션으로 유지) |
| Sanity (G0) 라벨 | `openai/gsm8k` (test) | `10_label_steps.py --dataset {gsm8k\|math500\|mathnet}`로 변경 |
| 검증/평가 | `ShadenA/MathNet` | English + text-only + final_answer 있는 200문제 |

## Reducer 모드

| mode | 식                          | 용도 |
|------|------------------------------|------|
| B1   | sign(A)                     | binary baseline |
| Q1   | mean(A)                     | 평균 advantage |
| Q3 ⭐ | mean − λ·std (λ=−0.5)       | risk-seeking, exploration |
| Q4   | CVaR_α (lower tail mean)    | tail-aware |

## PAV / Skywork PRM 정합성 메모

- Skywork PRM은 **단일 newline (`\n`)**을 step 경계로 학습. 정책이 `\n\n`을 출력해도 `score.py`가 자동 정규화.
- PRM 점수 추출: `PRM_MODEL.forward(return_probs=True) → (lm_logits, loss, value[B,T])`. value는 sigmoid 적용. 각 step의 마지막 token 위치만 채택.
- vendored `src/prm/skywork/`는 [SkyworkAI/skywork-o1-prm-inference](https://github.com/SkyworkAI/skywork-o1-prm-inference) (Apache 2.0)에서 가져옴.

## TRL GRPO 정합성 메모

- TRL ≥ 0.13의 `GRPOConfig`/`GRPOTrainer` 시그니처 사용 (`vllm_mode="colocate"`, `peft_config`, `processing_class`).
- reward func는 `(prompts, completions, **dataset_columns) → list[float]` — dataset의 `answer` 컬럼이 kwargs로 전달됨.
- step-wise PAV 보상은 합산해서 trajectory scalar로 변환 (group baseline은 GRPO가 처리).

## 검증 게이트

| Gate | 기준 |
|------|------|
| G0 | S1~S4 통과 + BoN-PAV ≥ BoN-PRM (MATH500 200) |
| G1 | BoN-PAV(분포) ≥ BoN-PAV(스칼라), corr(Q1, Q3) < 0.95 |
| G2 | Q3/Q4가 pass@256 +3%p **또는** entropy decay 50% 완화 |
| G3 | A.mean only vs A.mean+std ablation 유의차 |

## 추가 문서

- [docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md) — 구현 결정 / 모듈 책임 / 검증 게이트 매핑
- [docs/TRAINING_FLOW.md](docs/TRAINING_FLOW.md) — 시스템 구조 / 데이터 흐름 / 실행 단계 다이어그램
- [docs/구현 계획 — 차분 PAV + 분포형 보상 (Phase 0) 4dc18b40f5ed47259166ff0f0c0f8086.md](docs/구현%20계획%20—%20차분%20PAV%20+%20분포형%20보상%20(Phase%200)%204dc18b40f5ed47259166ff0f0c0f8086.md) — 원본 계획서
