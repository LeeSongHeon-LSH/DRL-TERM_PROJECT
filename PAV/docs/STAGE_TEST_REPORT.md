# Docker 단계별 테스트 결과 보고서

실행일: 2026-05-11
환경: RTX 3090 24GB 단일 호스트 (WSL2 + Docker Desktop + nvidia runtime)
대상 계획: [DOCKER_STAGE_TESTS.md](DOCKER_STAGE_TESTS.md)

---

## 요약

8단계 모두 통과. 발견된 차단 이슈 5건 모두 진행 중 fix. **분산 RPC 파이프라인(본체 → RabbitMQ → 워커들 → RabbitMQ → 본체)이 단일 24GB 호스트에서 실제 추론 + LoRA 학습까지 검증**됨.

| Stage | 측면 | 결과 | 핵심 증거 |
|---|---|:---:|---|
| 0 (6 sub) | 환경/빌드/import | ✅ | `pav-rl:latest` 10.9GB, 3개 service 모두 import OK, 컨테이너 안 RTX 3090 인식 |
| 1 broker | 🔵 | ✅ | RabbitMQ Up, 5672 listen |
| 2 publish | 🔵 | ✅ | 본체 venv → broker, 큐에 3건 적재 확인 |
| 3 PRM 워커 | 🔵🟡 | ✅ | "Worker ready", consumer 1대 등록 |
| 4 PRM E2E | 🔵🟡 | ✅ | 정답 step **0.6509** vs 오답 **0.1874** 분리 (PRM 추론 정상) |
| 5 μ 워커 | 🔵🟡 | ✅ | vLLM init 13초, KV cache 2674 blocks |
| 6 μ E2E | 🔵🟡 | ✅ | n=4 다양한 alt step 생성 (unique=4/4, temperature=1.0) |
| 7 PRM+μ 동시 | 🔵🟡 | ✅ | MCRolloutPAV K=8, advantage 분포 [8] (4.7초/호출) |
| **8 학습** | 🔵🟡🔴 | ✅ | **lora_B 196/196 학습됨** (zero-init 깨짐), KL 26배 증가, grad_norm > 0 |

---

## 환경

| 항목 | 값 |
|---|---|
| Host OS | Windows 11 + WSL2 (Linux 6.6.x) |
| GPU | NVIDIA GeForce RTX 3090 24GB |
| Driver | 561.09 (CUDA runtime 12.6) |
| Docker | 29.1.2 + Compose v2.40.3 + nvidia runtime |
| Disk (Linux /dev/sdf) | 1007GB total, 879GB free → HF 캐시 위치 |
| 이미지 | `pav-rl:latest` 10.9GB (단일 이미지, GPU service 3개 공유) |

---

## Stage 별 결과 + 측정값

### Stage 0 — 환경/빌드/import (모든 sub-step ✅)

| Sub | 결과 |
|---|---|
| 0.1 호스트 | GPU/driver/nvidia runtime/879GB free 확인 |
| 0.2 .env | 4개 키 설정 (AMQP_URL / HF_HOME_HOST / HF_TOKEN / WANDB_API_KEY) |
| 0.3 빌드 | `docker compose build` → 이미지 10.9GB |
| 0.4 import smoke | prm-worker / mu-worker / trainer 3개 모두 `imports OK` |
| 0.5 GPU 패스스루 | 컨테이너 안 `nvidia-smi` + `torch.cuda.is_available()=True` + `vram=23.99GB` |
| 0.6 정리 | `compose down`으로 컨테이너만 제거 (이미지/HF 캐시 보존) |

### Stage 1~2 — broker + 본체 publish (🔵 AMQP)

- broker Up: `Server startup complete; 5 plugins started`, `started TCP listener on [::]:5672`
- AMQP 포트 검증: `nc -zv localhost 5672` OK
- 본체에서 `pika` 3건 publish → `rabbitmqctl list_queues prm.requests = 3 messages`

### Stage 3 — PRM 워커 단독 (🟡 첫 추론)

PRM 1.5B(Skywork-o1-Open-PRM-Qwen-2.5-1.5B) 로딩 + AMQP consumer 등록. accelerate `device 0 90%` 메모리 분배. 9분만에 `Worker ready. Listening on queue 'prm.requests' (prefetch=1)`.

### Stage 4 — 본체 ↔ PRM E2E (full RPC 라운드트립)

[scripts/_smoke_remote_prm.py](../scripts/_smoke_remote_prm.py) 결과:

| 입력 step | PRM 점수 | 의미 |
|---|---:|---|
| `Step 1: 7 + 5 = 12.` (정답) | **0.6509** | 정답 인식 |
| `Step 1: Let me add these numbers.` (부분) | 0.7192 | 진행 중 step |
| `Step 1: 7 + 5 = 13.` (오답) | **0.1874** | ← 오답 명확히 분리 |
| `Step 1: Let me think carefully about this.` (filler) | 0.6729 | 형식적 step |

`score_batch(4개)` total 457ms, `score_per_step(5단계 full)` 301ms. **정답/오답 점수 분리(0.65 vs 0.19) 확인**.

### Stage 5~6 — μ 워커 단독 + E2E (🟡 실제 step 생성)

μ = Qwen2.5-Math-7B-Instruct, vLLM bf16. KV cache `cuda blocks: 2674`, `Maximum concurrency 10.45x`, `init engine 12.98 seconds`.

`sample_step_batch(n=4)` 응답 (5.7초, 첫 호출이라 prefix caching 없음):
```
[0] 'To solve the quadratic equation \(x^2 - 5x + 6 = 0\), we can use the method of factoring...'
[1] '... we can factor it into simpler linear factors...'
[2] '... we can use factoring...'
[3] '... we can use the factoring method...'
```
unique=4/4 — temperature=1.0의 다양성 작동.

### Stage 7 — PRM + μ 동시 (Phase 1 한 step)

MCRolloutPAV K=8, 4.7초 (PRM 1회 + μ 1회 + PRM batch 1회 = 3 RPC):

| 항목 | 값 |
|---|---|
| p_q (정책 step PRM 점수) | 0.8896 |
| p_v_samples (μ-rollout × 8) | [0.9946, 0.9951, 0.9976, ..., 0.9976] |
| advantage_samples | [-0.1050, -0.1055, ..., -0.1079] (모두 음수 — μ가 더 정교한 intro 생성) |
| mean advantage | -0.1055 |
| std | 0.0024 (분포 좁음 — 빈 prefix에서 μ가 유사한 시작 생성) |

**의미** — Phase 1 분포 신호가 정상으로 생성. 빈 prefix(s=∅) 특성상 분포가 좁은데, 실제 학습에서는 step h가 깊어지면 다양성 증가 예상.

### Stage 8 — trainer + GRPO 50 step (🔴 학습)

옵션 C 적용 (정책 1.5B 다운그레이드 + Phase 0 + 50 step smoke).

| 항목 | 값 |
|---|---|
| 정책 | Qwen2.5-Math-1.5B-Instruct + LoRA r=64 |
| PAV | DifferentialPAV (μ 안 씀) |
| Reward mode | Q3 (λ=-0.5, α=3.0) |
| Total steps | 50 |
| Train runtime | 492초 (≈ 8분) |
| Train samples/sec | 0.813 |

학습 진행 로그 (5 step 간격):

| step | grad_norm | reward (Q3) | reward_std | KL |
|---:|---:|---:|---:|---:|
| 5  | 0.1485 | 1.5890 | 0.4228 | 7.5e-6 |
| 10 | 0.1203 | 1.9102 | 0.5903 | 9.6e-6 |
| 15 | 0.0879 | 1.5016 | 0.3159 | 2.3e-5 |
| 20 | 0.1020 | 2.2343 | 0.1659 | 2.1e-5 |
| 25 | 0.2700 | 2.1103 | 0.5546 | 3.1e-5 |
| 30 | 0.1187 | 1.6565 | 0.4537 | 3.5e-5 |
| 35 | 0.1987 | 2.0883 | 0.3213 | 3.8e-5 |
| 40 | 0.1670 | 1.9820 | 0.3797 | 5.2e-5 |
| 45 | 0.2211 | 1.8994 | 0.5570 | 7.0e-5 |
| 50 | 0.1409 | 2.1115 | 0.2219 | **1.95e-4** |

### 학습 신호 결정적 증거

`outputs/stage8_smoke/checkpoint-50/adapter_model.safetensors` 분석:

```
LoRA tensors total: 392
lora_A modules: 196,  L2 norm range: [4.5994, 4.6372]    (kaiming uniform init, 큰 값 유지 정상)
lora_B modules: 196,  L2 norm range: [0.002859, 0.020525]
lora_B 중 norm==0 (학습 안 됨): 0/196                     ← 핵심: 모든 LoRA에 gradient 흐름
```

**PEFT가 `lora_B`를 0으로 초기화**하므로 norm > 0은 학습된 직접 증거. 196/196 모두 학습됨 + KL drift 26배 증가(7.5e-6 → 1.95e-4) → **분산 reward 신호가 본체 LoRA로 정상 backpropagation**되었음.

---

## 진행 중 발견 + fix한 이슈 5건

| # | Stage | 이슈 | 원인 | Fix | 변경 파일 |
|:---:|---|---|---|---|---|
| 1 | 3 | `ModuleNotFoundError: hf_transfer` | Dockerfile의 `HF_HUB_ENABLE_HF_TRANSFER=1`인데 패키지 미설치 | compose env로 `="0"` override | [docker-compose.yml](../docker-compose.yml) |
| 2 | 3 | `Unrecognized configuration class Qwen2RMConfig for AutoModelForCausalLM` | 1.5B는 `Qwen2ForRewardModel`(reward head 내장), `auto_map`에 `AutoModel`만 등록 → PRM_MODEL(7B 패턴) 비호환 | 1.5B 전용 wrapper 작성. score.py에서 architecture 보고 분기 | [src/prm/skywork_rm.py](../src/prm/skywork_rm.py), [src/prm/score.py](../src/prm/score.py) |
| 3 | 5 | `ValueError: No available memory for the cache blocks` | μ vLLM `gpu_memory_utilization=0.30` → 7B(14GB) 가중치도 못 들어감, `max_model_len` 기본 32k → KV cache 폭증 | 0.75로 상향, `max_model_len=4096` 명시 | [configs/policy.yaml](../configs/policy.yaml), [src/rollout/mu_sampler.py](../src/rollout/mu_sampler.py) |
| 4 | 8 | `OSError: I/O error (no space)` 호스트 1.3GB만 남음 | E: 드라이브(`./.hf_cache` 경로) 100% 사용 | HF 캐시를 WSL Linux fs(`/home/lkmisem/pav_hf_cache`)로 이전, 879GB 여유 | [.env](../.env) |
| 5 | 8 | `TypeError: GRPOConfig got unexpected keyword 'epsilon'` | TRL 0.15.2가 `epsilon` 대신 `epsilon_low/high`로 분리 | `inspect.signature`로 supported args만 자동 필터링 | [src/train/grpo_trainer.py](../src/train/grpo_trainer.py) |

부수 — 모든 GPU service에 `src/`, `scripts/`, `configs/` host bind mount 추가 → 코드 수정 시 이미지 재빌드 없이 컨테이너 `--force-recreate`만으로 반영 (개발 hot-reload).

---

## VRAM 사용 추이 (단일 24GB)

| 단계 | 떠있는 GPU 컨테이너 | 측정 사용량 |
|---|---|---:|
| Stage 1~2 | broker | 2GB (idle) |
| Stage 3~4 | broker + prm-worker | ~3GB + idle |
| Stage 5~6 | broker + mu-worker | ~19GB (mu vLLM 0.75 × 24GB ≈ 18GB) |
| Stage 7 | broker + prm + mu | ~22GB (mu 18 + prm 3 + overhead) |
| Stage 8 (옵션 C) | broker + prm + trainer | ~15GB (trainer 1.5B LoRA + colocate vLLM) |

24GB 카드에 모든 단계 안전 적재. Stage 8을 정책 7B로 끌어올리면 옵션 A(워커를 다른 PC로) 필요.

---

## 시간 측정

| 단계 | 소요 시간 |
|---|---|
| 0.3 이미지 빌드 (cache hit) | < 1분 |
| 0.4 import smoke 3개 | ~2분 |
| 1 broker 기동 | 20초 (이미지 pull 제외) |
| 3 PRM 워커 첫 ready | ~9분 (모델 다운 + 로딩) |
| 4 PRM E2E (RPC 4건) | < 1.5초 총 |
| 5 μ 워커 첫 ready | ~25분 (7B 다운 + vLLM init) |
| 6 μ E2E (n=4) | 5.7초 (첫 호출) |
| 7 MCRolloutPAV K=8 | 4.7초 |
| 8 GRPO 50 step | 8분 12초 |

---

## 결론

1. **분산 아키텍처가 의도대로 동작** — 본체가 publish, 워커가 consume, reply queue로 응답, correlation_id 매칭 모두 broker를 거쳐 실시간 라운드트립.
2. **단일 24GB GPU 호스트에서 Phase 0 학습까지 검증** — μ를 끄고 정책을 1.5B로 다운그레이드하면 모든 컨테이너 동시 적재 가능.
3. **Phase 1 분포 신호도 본체 단독으로 검증** (Stage 7) — 실제 학습은 옵션 A(분산 PC)로 갈 때 동일 패턴 재사용.
4. **`PAVMethod` Protocol + 동일 인터페이스(RemotePRM/RemoteMuSampler)** 덕분에 `score.py` 1군데 fix(architecture 분기) 외에는 호출 코드 0줄 수정 — RL 코드의 `pav` 인스턴스 교체만으로 Phase 0/1 swap.

다음 권장 단계 — [DOCKER_STAGE_TESTS.md](DOCKER_STAGE_TESTS.md)의 옵션 A 분산 시나리오로 정책 7B + Phase 1 + 본 학습(5000 step) 진입.
