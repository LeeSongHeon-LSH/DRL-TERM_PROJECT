# 현재 학습 설정 (Training Configuration)

> **마지막 업데이트**: 2026-06-15
> **브랜치**: `Implement`

---

## 1. 모델 구성

| 구성 요소 | 모델 | 양자화 | 위치 | 상태 |
|-----------|------|--------|------|------|
| **정책 π** | `Qwen/Qwen2.5-Math-1.5B-Instruct` | `bfloat16` (Full FT) | 학습 PC (3090) | ✅ 학습 중 |
| **Prover μ** | `Qwen/Qwen2.5-Math-1.5B-Instruct` (frozen) | `fp16` (T4) | cloud T4 ×2 | ✅ 2대 online |
| **PRM** | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | `int8` (bitsandbytes) | cloud T4 ×1 | ✅ 1대 online |

---

## 2. GRPO 학습 설정 (`configs/rl_q3.yaml`)

### 2.1 PAV (Process Advantage Verifier)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **method** | `mc_rollout` | Phase 1 — μ로 K개 alternative 생성 |
| **K** | `16` | MC rollout 수 (counterfactual advantage 추정) |
| **reward.mode** | `Q3` | risk-seeking: `mean − λ·std` |
| **reward.alpha** | `3.0` | PAV 가중치 |
| **reward.lam** | `−0.5` | Q3 risk-seeking 계수 |
| **reward.cvar_alpha** | `0.2` | Q4 CVaR tail (미사용) |

> **Ablation (C2 스칼라)**: [`configs/rl_c2_scalar.yaml`](../configs/rl_c2_scalar.yaml) — PAV 추출(mc_rollout, K=16)은 C3와 100% 동일하게 두고 **reducer만 Q3 → Q1(mean)** 으로 바꾼 baseline. 분포 항(std)을 끄고 평균만 사용. 실행: [`scripts/run_c2_scalar.sh`](../scripts/run_c2_scalar.sh).

### 2.2 GRPO

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **group_size** | `8` | 문제당 rollout 수 |
| **kl_beta** | `0.04` | KL divergence penalty (anti-hacking) |
| **clip_eps** | `0.2` | GRPO clipping |
| **learning_rate** | **`2.0e-6`** | 1.5B Full FT 기준 |
| **total_steps** | `5000` | 총 학습 step (~70시간 예상) |
| **warmup_steps** | `50` | LR warmup (1%) |
| **gradient_accumulation** | `8` | gradient 누적 횟수 |
| **optimizer** | `adamw_bnb_8bit` | 8-bit AdamW (VRAM 절반) |
| **max_completion_length** | **`256`** | 코드상 `vllm.max_new_tokens`로 구동 (rl_q3.yaml=256). `g.max_completion_length`(512)는 vllm 키 없을 때만 fallback |

### 2.3 배치 구성 (코드 적용값)

| 파라미터 | 값 | 계산 |
|----------|-----|------|
| `per_device_train_batch_size` | **8** | = group_size |
| `gradient_accumulation_steps` | **8** | |
| `num_generations` | **8** | = group_size |
| **Effective Batch** | **64** | 8 × 1 × 8 |
| **64 / 8 = 8** | ✅ | 나누어 떨어짐 |

---

## 3. vLLM 설정 (`configs/rl_q3.yaml` → `vllm`)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **colocate** | `true` | 학습 PC 내 vLLM colocate |
| **gpu_memory_utilization** | `0.30` | ~7.2 GB 할당 |
| **enable_prefix_caching** | `true` | prefix caching 활성화 |

---

## 4. 분산 인프라 (FRP)

### 4.1 네트워크 구성

| 구성 요소 | 컨테이너 | 상태 | 포트 |
|-----------|----------|------|------|
| **FRPS (학습 PC)** | `pav-frps` | ✅ 실행 중 | 7000, 7500, 18001, 18002 |
| **FRPC (학습 PC)** | `pav-frpc` | ✅ 실행 중 | outbound TCP |
| **Trainer** | `pav-trainer` | ✅ 학습 중 | — |
| **Dashboard** | `pav-dashboard` | ✅ 실행 중 | 8501 |

### 4.2 추론 PC (FRP LB Pool)

| 서비스 | 프록시명 | 그룹 | 상태 | 트래픽 |
|--------|----------|------|------|--------|
| **μ 서버 #1** | `mu-t4-mu-01` | `mu_cluster` | 🟢 online | In: 25MB / Out: 153MB |
| **μ 서버 #2** | `mu-t4-mu-02` | `mu_cluster` | 🟢 online | In: 28MB / Out: 169MB |
| **PRM 서버** | `prm-t4-prm-01` | `prm_cluster` | 🟢 online | In: 525MB / Out: 22MB |

### 4.3 환경변수

```bash
# 학습 PC (PAV/docker-compose.yml)
FRPS_TOKEN=<32+ chars>
FRPS_ADDR=ksisem0811.duckdns.org
FRPS_DASHBOARD_USER=admin
FRPS_DASHBOARD_PASSWORD=123443211qwerrewqq
PRM_ENDPOINT=http://frps:18002
MU_ENDPOINT=http://frps:18001
MU_REPLICAS=2

# 추론 PC (PAV/docker-compose.inference.yml)
FRPS_ADDR=ksisem0811.duckdns.org
FRPS_TOKEN=<동일 토큰>
NODE_NAME=t4-mu-01  # 또는 t4-mu-02, t4-prm-01
```

---

## 5. VRAM 예측 (학습 PC 3090 24GB)

| 항목 | 메모리 |
|------|--------|
| π 1.5B base (bf16) | ~3.1 GB |
| gradients (bf16) | ~3.1 GB |
| 8-bit Adam states | ~3.1 GB |
| activations (grad-ckpt) | ~2–3 GB |
| vLLM colocate (0.30) | ~7.2 GB |
| **합계** | **~18–19 GB** |
| **마진** | **5–6 GB** ✅ |

---

## 6. 데이터

| 항목 | 설정 |
|------|------|
| **학습 데이터** | GSM8K train |
| **평가 데이터** | MathNet (English, text-only) |
| **평가 subset** | 200문제 |
| **프롬프트 포맷** | **few-shot 1-shot + 강제 step 포맷** ([`policy_data.py`](../src/train/policy_data.py) `_make_chat_wrapper`) |

> 프롬프트는 system 규칙(자연어 step만, 줄당 `"Step k:"` 1개, 코드 금지, 마지막 줄 `"Answer: <number>"`) + 예시 1개(user/assistant) 를 정책 입력 앞에 붙인다. 이 변경으로 completion 길이/분산이 zero-shot 대비 줄어든다 (실험: `PAV-distribution-fewshot-test`).

---

## 7. 출력 및 로깅

| 항목 | 설정 |
|------|------|
| **출력 폴더** | `./outputs/<wandb_run_name>/` (타임스탬프 suffix 없음 — [`grpo_trainer.py`](../src/train/grpo_trainer.py) `output_dir`) |
| **wandb_run_name** | `PAV-distribution-fewshot-nomal-test` (rl_q3.yaml 현재값) |
| **로깅 간격** | **1 step** (`log_every` → `logging_steps`) |
| **저장 간격** | **100 step** (`eval_every` → `save_steps`) |
| **dump_samples** | 10 step (`dump_samples_every`) |

> **수행된 실험 run** (`outputs/`): `PAV-distribution-test`(Q3 zero-shot), `PAV-distribution-fewshot-test`(Q3 few-shot), `PAV-scalar-c2-test`(Q1 scalar). 3-run 학습변화량 비교 → [`outputs/comparison/training_comparison.md`](../outputs/comparison/training_comparison.md).
| **Dashboard** | `http://<학습PC>:8501` |
| **FRP Dashboard** | `http://<학습PC>:7500` (admin / `FRPS_DASHBOARD_PASSWORD`) |

---

## 8. 파일 위치

| 파일 | 경로 |
|------|------|
| 메인 RL 설정 | `configs/rl_q3.yaml` |
| 정책 설정 | `configs/policy.yaml` |
| PRM 설정 | `configs/prm.yaml` |
| 학습 스크립트 | `scripts/03_grpo_train.py` |
| GRPO Trainer | `src/train/grpo_trainer.py` |
| PAV Reward | `src/train/reward_fn.py` |
| Remote μ | `src/rollout/remote_mu.py` |
| Remote PRM | `src/prm/remote_client.py` |
| 학습 PC Compose (trainer + frps + frpc + dashboard) | `docker-compose.yml` |
| 추론 PC Compose | `docker-compose.inference.yml` |
| FRP 설정 (toml) | `frp/frps.toml`, `frp/frpc.toml` |
| C2 ablation 설정 / 실행 | `configs/rl_c2_scalar.yaml`, `scripts/run_c2_scalar.sh` |

---

## 9. 빠른 명령어

```bash
# 학습 PC — 전체 시작
FRPS_TOKEN=<token> FRPS_ADDR=ksisem0811.duckdns.org \
  FRPS_DASHBOARD_USER=admin FRPS_DASHBOARD_PASSWORD=<pw> \
  docker compose up -d --build trainer dashboard frpc

# 추론 PC — μ/PRM 시작
FRPS_ADDR=ksisem0811.duckdns.org FRPS_TOKEN=<token> \
  NODE_NAME=t4-mu-01 docker compose -f docker-compose.inference.yml up -d

# 로그 확인
docker logs -f pav-trainer
docker logs -f pav-dashboard

# FRP 상태 확인
docker exec pav-frps wget -qO- --header="Authorization: Basic $(echo -n 'admin:<pw>' | base64)" \
  http://localhost:7500/api/proxy/tcp
```

---

## 10. 알려진 이슈

| 이슈 | 상태 | 비고 |
|------|------|------|
| Checkpoint resume 시 batch size 충돌 | ⚠️ 주의 | 이전 checkpoint 삭제 후 새로 시작 권장 |
| FRP dashboard 404 warning | ✅ 해결 | `/api/proxy/tcp` + Basic Auth 적용 |
| Plotly import error | ✅ 해결 | `Dockerfile.dashboard`에 `plotly>=5.18` 추가 |
| `expandable_segments` WSL 미지원 | ℹ️ 무시 | WSL 한계, 성능 영향 미미 |

---

*이 문서는 `docs/TRAINING_CONFIG.md`에 저장되어 있으며, 설정 변경 시 업데이트 필요.*
