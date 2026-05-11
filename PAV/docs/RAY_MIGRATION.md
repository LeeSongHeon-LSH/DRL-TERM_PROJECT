# Ray 전환 진행 상황

분산 통신을 **RabbitMQ → Ray cluster로 전환**. 단계 1~9 완료, 남은 작업은 Stage R0~R7 실제 검증 + 보고서 작성.
원본 RabbitMQ 구조는 [STAGE_TEST_REPORT.md](STAGE_TEST_REPORT.md)에서 history로 보존.

## 동기

| 측면 | RabbitMQ | Ray |
|---|---|---|
| RPC overhead (작은 페이로드) | 1~3ms | < 1ms |
| 직렬화 | JSON | msgpack/pickle + plasma store |
| Tensor 직접 전달 | full copy | zero-copy |
| Cluster 관리 | broker 1대 | head/worker 통합 |
| 향후 multi-GPU 학습 (Ray Train) | 별도 구현 | 같은 cluster에서 자연 통합 |

작은 텍스트 페이로드라 RPC overhead 절감은 < 1% 수준이지만, **multi-GPU 학습 + 보상 모델 분산 + Tune(HP 탐색)을 같은 Ray 생태계로 통합**하기 위해 전환.

## 최종 토폴로지

```
┌──────────────────────────────────────────────────────────┐
│ PC A: 학습 본체 — RTX 3090 24GB (Ray Head)              │
│  ├─ GRPOTrainer (TRL) wrap in TorchTrainer (Ray Train)  │
│  ├─ π = Qwen2.5-Math-7B + LoRA r=64           ~14GB     │
│  ├─ π_ref = PEFT disabled adapter             +0        │
│  └─ vLLM colocate rollout (gpu_mem=0.30)      ~8GB      │
│  total ≈ 22GB                                            │
└──────────────────────────────────────────────────────────┘
       │ Ray cluster (GCS 6379, dashboard 8265, client 10001)
       │
┌──────────────────────┐   ┌──────────────────────────────┐
│ PC B: RTX 3090 24GB  │   │ PC C…G: RTX 5070 12GB × 5+   │
│ RayMuActor (1 inst)  │   │ RayPRMActor (5+ replicas)    │
│ Qwen-7B base + vLLM  │   │ Skywork-PRM 1.5B             │
│ gpu_mem=0.75, ~18GB  │   │ transformers + Ray actor     │
└──────────────────────┘   └──────────────────────────────┘
```

## 진행 단계 — 11 steps

| # | 단계 | 상태 | 산출물 |
|:---:|---|:---:|---|
| 1 | `pyproject.toml` — `pika` 제거 + `ray[default]>=2.30` 추가, `[gpu]`에 `ray[train]` | ✅ | `ray==2.40.0` 설치 |
| 2 | `RayPRMActor` + `PRMHandler` + `RayPRMClient` + `RayPRMClientPool` | ✅ | `src/prm/ray_actor.py`, `ray_client.py` (11 tests) |
| 3 | `RayMuActor` + `MuHandler` + `RayMuClient` | ✅ | `src/rollout/ray_actor.py`, `ray_client.py` (7 tests) |
| 4 | `loader.load_prm` / `build_mu_from_policy_yaml` — `local|ray`만, RabbitMQ 분기 제거 | ✅ | `loader.py` 재작성, `mu_sampler.py`에 `force_local` 추가 |
| 5 | `configs/prm.yaml`, `configs/policy.yaml` — `amqp_url` 제거, `ray_address`/`actor_name`/`num_replicas` 추가 | ✅ | configs 갱신 |
| 6 | `scripts/serve_prm_ray.py`, `serve_mu_ray.py` + `src/train/ray_train.py` (TorchTrainer wrap) + `scripts/03_grpo_train.py` 갱신 | ✅ | scripts 갱신 |
| 7 | `docker-compose.yml` — `rabbitmq` 제거 + `ray-head` + `prm-worker`/`mu-worker`/`trainer` 4개 service, `.env.example` 갱신 | ✅ | compose 재작성 |
| 8 | 옛 파일 11개 삭제 (handlers/remote_*/mu_handlers/mu_worker/serve_prm/serve_mu/_smoke_remote_prm/tests/test_remote.py/integration) | ✅ | 11개 파일 + integration 폴더 제거 |
| 9 | `tests/test_imports.py`에 Ray 모듈 import 검증 추가, 전체 회귀 38/38 통과 | ✅ | `test_imports.py` 갱신 |
| 10 | Stage R0~R7 재검증 (실제 Ray cluster + 모델) | ⚠️ 부분 완료 | R0~R3 ✅ / R4~R7 ⏸ (WSL stuck — [STAGE_R_REPORT.md](STAGE_R_REPORT.md)) |
| 11 | `docs/*` 다이어그램·표 Ray로 일괄 갱신 + Stage R 결과 보고서 | ✅ | 1차 갱신 완료 + [STAGE_R_REPORT.md](STAGE_R_REPORT.md) 작성 |

## 현재 코드베이스 — Ray 전용 (RabbitMQ 흔적 없음)

| 영역 | 신규 / 변경 / 삭제 |
|---|---|
| **PRM** | ✨ `src/prm/ray_actor.py`, `ray_client.py`, `loader.py` 재작성 / 🗑 `handlers.py`, `remote_client.py`, `remote_worker.py` 삭제 |
| **μ** | ✨ `src/rollout/ray_actor.py`, `ray_client.py`, `mu_sampler.py` 갱신 / 🗑 `mu_handlers.py`, `remote_mu.py`, `mu_worker.py` 삭제 |
| **학습** | ✨ `src/train/ray_train.py` 신규 (TorchTrainer wrap) / `scripts/03_grpo_train.py` 재작성 |
| **인프라** | ✨ `scripts/serve_prm_ray.py`, `serve_mu_ray.py` / `docker-compose.yml`, `pyproject.toml`, `configs/*.yaml`, `.env*` 갱신 / 🗑 `serve_prm.py`, `serve_mu.py`, `_smoke_remote_prm.py` 삭제 |
| **테스트** | ✨ `tests/test_ray.py` (18 tests: PRM 11 + μ 7) / `test_imports.py`에 Ray 모듈 추가 / 🗑 `test_remote.py`, `integration/test_real_amqp.py` 삭제 |

## 단위 테스트 현황

| 파일 | 결과 |
|---|---|
| `tests/test_imports.py` | 8/8 ✅ (Ray 모듈 import 검증 추가) |
| `tests/test_parser.py` | 6/6 ✅ |
| `tests/test_pav_swap.py` | 6/6 ✅ |
| `tests/test_ray.py` | **18/18 ✅** (Layer 1 핸들러 직접 7개, Layer 2 RayClient + fake actor 11개) |
| **총** | **38/38 통과** |

## Ray Stage R0~R7 매핑

| 현행 (RabbitMQ) | Ray |
|---|---|
| Stage 1 broker | R0 Ray Head 컨테이너 단독 |
| Stage 2 publish | R1 본체 `ray.init(address=…)` |
| Stage 3 PRM worker | R2 RayPRMActor 단독 (replicas=1) |
| (추가) | **R3 RayPRMActor replicas=5 round-robin** |
| Stage 4 PRM E2E | R4 RayPRMClient (load_prm 자동 wrap) |
| Stage 5 μ worker | R5 RayMuActor 단독 |
| Stage 6 μ E2E | R6 RayMuClient |
| Stage 7 PRM+μ 동시 | (R6에 통합) — MCRolloutPAV K=16 |
| Stage 8 학습 | R7 TorchTrainer wrap + 50 step smoke + LoRA 검증 |

## 다음 단계

**단일 호스트 WSL 검증은 R0~R3까지 완료** — PRM 분산 RPC 정상 동작 (정답 0.65 vs 오답 0.19 분리, RabbitMQ Stage 4와 동일). R4~R7은 WSL+Docker+Ray+vLLM 조합 stuck으로 보류, **분산 환경(다른 PC가 Linux 네이티브)** 에서 검증 예정.

자세한 결과 → [STAGE_R_REPORT.md](STAGE_R_REPORT.md).

---

## 사용법

### 분산 환경 (다중 PC)
```bash
# 각 PC에서 .env 설정 (RAY_ADDRESS만 head IP로)
cp .env.example .env

# 본체 PC (Head):
docker compose --profile head up -d

# PRM 워커 PC들 (5070):
docker compose --profile prm up -d

# μ 워커 PC (3090):
docker compose --profile mu up -d

# 본체에서 학습 시작:
docker compose --profile trainer up -d
docker compose logs -f trainer
```

### 단일 호스트 (모든 service 한 PC)
```bash
docker compose --profile all up -d
```

### configs/*.yaml 의 `mode`만 변경하면 local ↔ ray 스위치
```yaml
# configs/prm.yaml
mode: local   # 같은 GPU에 PRM 적재
mode: ray     # Ray cluster의 named actor에 RPC
num_replicas: 5   # 5070 ×5 라운드로빈
```
