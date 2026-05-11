# PAV-RL — 차분 PAV + 분포형 보상 (Phase 0 + Phase 1)

천공 PRM(Skywork-o1-Open-PRM-Qwen-2.5-1.5B)을 활용해 PRM 추가 학습 없이 GRPO+LoRA 학습까지 가는 파이프라인.
(7B 버전도 [configs/prm.yaml](configs/prm.yaml)의 `model_id` 한 줄 교체로 사용 가능.)

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
└── tests/                        # swap / parser / imports (19 tests)
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
uv run --group dev pytest tests/ -v           # 19 passed
```

## 실행 순서 (실제 모델 다운로드 후)

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
    --lora ./outputs/q3_lambda-0.5_K16/checkpoint-5000 --N 64
```

## Docker로 띄우기 (가장 간편한 방법)

4개 service(`rabbitmq`, `prm-worker`, `mu-worker`, `trainer`)를 단일 [Dockerfile](Dockerfile)로
빌드 + [docker-compose.yml](docker-compose.yml)로 오케스트레이션. **profiles로 PC별 분리 실행**.

### 사전 (각 GPU PC)
```bash
# 호스트에 NVIDIA Container Toolkit 설치 (한 번만)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

cp .env.example .env       # AMQP_URL / HF_HOME_HOST / HF_TOKEN / WANDB_API_KEY 채움
docker compose build       # 단일 이미지 한 번 빌드 (모든 GPU service 공통)
```

### 분산 시 — 각 PC에서 자기 profile만
```bash
# 브로커 PC
docker compose --profile broker  up -d

# PRM 워커 PC들 (RTX 5070 12GB+, 여러 PC 가능)
docker compose --profile prm     up -d

# μ 워커 PC (16GB+ GPU)
docker compose --profile mu      up -d

# 본체 (3090) — 학습 시작
docker compose --profile trainer up -d
docker compose logs -f trainer
```

### 단일 호스트에서 다 띄우기 (테스트용 — 큰 GPU 필요)
```bash
docker compose --profile all up -d
```

### 부가 정보
- **GPU 패스스루**: `deploy.resources.reservations.devices.driver: nvidia` (compose v2 표준)
- **HF 캐시 영속화**: `${HF_HOME_HOST}` (.env) → 컨테이너 `/cache/huggingface`. 모델은 컨테이너 안에 굽지 않음 → 첫 호출 시 lazy 다운, 호스트에 보존.
- **outputs**: trainer는 `./outputs`에 LoRA 체크포인트 저장 (호스트 bind mount).
- **shm_size**: trainer는 vLLM 내부 multiprocessing 때문에 8GB 권장 (compose에 이미 설정).

## 분산 구조 (RabbitMQ)

PRM 1.5B 본체 적재 + 정책 7B LoRA + vLLM colocate를 한 GPU에 다 올리면 빠듯합니다.
**PRM과 μ를 다른 PC들로 분리**하고 RabbitMQ 큐로 RPC하면 본체에는 학습 정책만 남아 OOM 회피 + 워커 추가/제거가 자유로워집니다.

### 1) 브로커 1대에서 — RabbitMQ
```bash
docker compose up -d                    # docker-compose.yml 사용
# Management UI: http://<broker-host>:15672  (id/pw: guest/guest)
```

### 2) 보상모델 PC들에서 (각 PC, 동일 명령)
```bash
uv sync --extra gpu

# PRM 워커 (RTX 5070 12GB+ — 같은 큐를 listen하는 워커 N개 동시 가능)
uv run python scripts/serve_prm.py \
    --config configs/prm.yaml \
    --amqp-url amqp://guest:guest@<broker-host>:5672/ \
    --queue prm.requests

# μ 워커 (16GB+ — μ는 7B base)
uv run python scripts/serve_mu.py \
    --config configs/policy.yaml \
    --amqp-url amqp://guest:guest@<broker-host>:5672/ \
    --queue mu.requests
```

### 3) 본체 (3090) — config 수정 후 학습
```yaml
# configs/prm.yaml
mode: remote
amqp_url: amqp://guest:guest@<broker-host>:5672/
request_queue: prm.requests
rpc_timeout: 120

# configs/policy.yaml (mu: 섹션)
mu:
  mode: remote
  amqp_url: amqp://guest:guest@<broker-host>:5672/
  request_queue: mu.requests
  rpc_timeout: 180
```
이후 `scripts/03_grpo_train.py`는 그대로 실행 — `load_prm` / `build_mu_from_policy_yaml`이
`mode`에 따라 `RemotePRM` / `RemoteMuSampler`를 자동 반환합니다 (모두 표준 AMQP RPC: reply_to + correlation_id).

### 본체 VRAM 추정 (3090 24GB, PRM/μ remote)
| 컴포넌트 | VRAM |
|---|---|
| 학습 π 7B + LoRA + Adam states | ~14GB |
| vLLM colocate (rollout, gpu_mem_util=0.30) | ~7~9GB |
| PRM / μ | 0 (다른 PC) |
| **합계** | **~22GB** ✅ |

`configs/rl_q3.yaml`의 `vllm.gpu_memory_utilization`을 0.30으로 이미 설정해 둠.

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

자세한 실행 계획은 `구현 계획 — 차분 PAV + 분포형 보상 (Phase 0).md` 참고.
