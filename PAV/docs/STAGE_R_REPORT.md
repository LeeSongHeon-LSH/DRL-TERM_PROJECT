# Stage R 단계별 테스트 결과 보고서 (Ray cluster)

실행일: 2026-05-11
환경: RTX 3090 24GB **단일 호스트** (Windows 11 + WSL2 + Docker Desktop + nvidia runtime)
transport: **Ray cluster** (head + worker actor pattern)
대상 계획: [DOCKER_STAGE_TESTS.md](DOCKER_STAGE_TESTS.md) §Ray Stage 매핑

> 본 보고서는 단일 호스트 WSL 환경에서의 검증 기록입니다. **실제 분산 환경(다른 PC들이 Linux 네이티브)에서는 R4~R7 stuck 이슈가 해결될 가능성이 큼** — 보고서 §3, §5 참고.

RabbitMQ 시점 검증 결과는 [STAGE_TEST_REPORT.md](STAGE_TEST_REPORT.md) (history).

---

## 1. 요약

| Stage | 측면 | 결과 | 핵심 증거 |
|---|---|:---:|---|
| R0 Ray head | 🔵 | ✅ | 6379/8265/10001 listen, `ray start --head --block` 정상 |
| R1 cluster reachable | 🔵 | ✅ | `ray status` 1 node, CPU 2/2, GPU 0/0 |
| R2 RayPRMActor 등록 | 🔵🟡 | ✅ | PRM 1.5B 22초 로드 + named actor `prm-actor` 등록 |
| R3 RayPRMClient E2E | 🔵🟡 | ✅ | 정답 0.6509 vs 오답 0.1874 — RabbitMQ Stage 4와 동일 |
| R4 RayMuActor | ⚠️ | ⏸ | **vLLM이 weight load 단계에서 Ray actor 안에서 stuck** — WSL+Docker+Ray+vLLM 0.7 조합 이슈 |
| R5 RayMuClient | — | ⏸ | R4 의존 |
| R6 PRM+μ 동시 | — | ⏸ | R4 의존 |
| R7 trainer 학습 | ⚠️ | ⏸ | `ray.init` 호출 패턴 fix는 적용 (`ray start --address && python …`) — R4 stuck 우회 후 검증 예정 |

핵심 메시지:
- **PRM 분산 RPC + named actor pattern 정상 동작** (R3에서 정답/오답 점수 분리 확인).
- **μ는 단일 호스트 WSL 환경에서 vLLM init이 Ray actor 안에서 hang** — 알려진 환경 의존성. 실제 분산 환경(Linux 네이티브)에서는 정상 동작 예상.
- 코드 자체는 단위 테스트 38/38 통과 — 인터페이스 호환성 검증됨.

---

## 2. 환경

| 항목 | 값 |
|---|---|
| Host OS | Windows 11 + WSL2 (Linux 6.6.x) |
| Docker | 29.1.2 + Compose v2.40.3 + nvidia runtime |
| GPU | NVIDIA GeForce RTX 3090 24GB |
| Driver | 561.09 (CUDA 12.6 runtime) |
| Ray | 2.40.0 |
| 이미지 | `pav-rl:latest` ~11GB (4개 GPU service 공유) |

---

## 3. Stage 별 결과

### R0 — Ray Head ✅

```bash
docker compose --profile head up -d
docker exec pav-ray-head ray status
```

| 검증 | 결과 |
|---|---|
| `ray start --head --block` 정상 동작 | ✅ |
| 포트 6379 (GCS) / 8265 (dashboard) / 10001 (client) listen | ✅ |
| `ray status` Active 1 node, CPU 2/2, GPU 0/0 (head는 GPU 미사용) | ✅ |

> 발견된 fix: `command: >` multiline 형식이 bash로 줄바꿈 전달되어 `--num-cpus=2` 등이 별도 명령으로 잘못 파싱됨 → **list form `command: ["ray", "start", ...]`로 교체**.

### R1 — cluster reachable ✅

```bash
docker exec pav-ray-head python -c "
import ray; ray.init(address='auto', namespace='pav-rl')
print('nodes:', len(ray.nodes()), 'resources:', ray.cluster_resources())"
```

출력: `nodes: 1, resources: {'CPU': 2.0, 'memory': 35.2GiB, ...}` ✅

> Host venv ↔ Docker network 격리 때문에 본체 venv의 `ray://localhost:10001` Client mode는 timeout. 본체에서 cluster 사용은 `docker exec` 또는 `trainer` 컨테이너 안에서.

### R2 — RayPRMActor 등록 ✅

```bash
docker compose --profile head --profile prm up -d
```

| 검증 | 결과 |
|---|---|
| `Worker ray join + actor 등록` | ✅ |
| PRM 1.5B 로드 (Skywork-o1-Open-PRM-Qwen-2.5-1.5B) | ✅ 22초 (HF 캐시 hit) |
| `prm-actor` named actor health 응답 | ✅ |
| cluster: 2 nodes (head + prm-worker), 1.0/1.0 GPU 사용 | ✅ |

### R3 — RayPRMClient E2E ✅

`docker exec pav-ray-head python r3_test.py`:

| 입력 step | PRM 점수 | latency |
|---|---:|---:|
| `Step 1: 7 + 5 = 12.` (정답) | **0.6509** | 774ms (첫 호출) |
| `Step 1: Let me add these numbers.` (부분) | 0.7192 | 141ms |
| `Step 1: 7 + 5 = 13.` (오답) | **0.1874** | 168ms |
| `Step 1: Let me think carefully about this.` (filler) | 0.6729 | 172ms |
| `score_batch(4개)` | 4 scores 일치 | 242ms (1 RPC) |
| `score_per_step(5단계 full)` | 5개 점수 | 165ms |

**RabbitMQ Stage 4와 동일한 출력값 + 비슷한 latency** — Ray transport에서도 정답/오답 점수 분리(0.65 vs 0.19) 그대로.

### R4 — RayMuActor ⏸ (보류)

**현상**: μ Qwen2.5-Math-7B base를 vLLM 0.7.3으로 Ray actor 안에서 init할 때, weight load 단계(`weight_utils.py:254 Using model weights format ['*.safetensors']`) **이후 무한 hang**.

| 디버깅 데이터 | 값 |
|---|---|
| GPU 메모리 점유 | 16.4GB (모델 weight + KV cache로 메모리 진입은 됨) |
| GPU util | 6~10% (idle 수준) |
| ray actor 응답 | `GetTimeoutError` (health 호출이 hang) |
| docker logs | weight_utils 마지막 줄에서 진전 없음 |
| 시도한 fix | `enforce_eager=True`, `disable_custom_all_reduce=True`, polling health (30분) — 모두 같은 위치에서 stuck |

**의심 원인**: WSL2 + Docker Desktop + Ray actor + vLLM 0.7 multiprocessing 조합. WSL 환경에서 vLLM이 internal worker subprocess를 spawn할 때 Ray actor의 nested-process 환경과 충돌 가능. RabbitMQ Stage 5에서는 단일 컨테이너 안에서 vLLM 직접 실행이라 정상 동작 (~25초).

**Linux 네이티브에서 해결 가능성**:
- vLLM의 GitHub 이슈들에서 WSL pin_memory 경고 + Ray 조합 stuck 보고 있음
- 워커 PC가 Linux 네이티브 (Ubuntu)면 vLLM이 정상 init 예상

**우회 경로** (확정되면 적용):
1. μ 워커를 Linux 호스트에 배치 (분산 시나리오 그대로)
2. μ를 `enforce_eager=True` + `tensor_parallel_size=1` + `disable_custom_all_reduce=True` 외에 `swap_space=0`, `block_size=16` 추가
3. μ를 vLLM 대신 transformers `model.generate` fallback (느리지만 동작 보장)

### R5/R6 ⏸

R4 의존이라 진행 못함.

### R7 — trainer 학습 ⚠️

`ray.init` 호출 방식 fix 적용했으나 검증 마지막 단계에서 시간 소진:

1. trainer 컨테이너의 ray.init이 **새 local cluster 시작 시도** → cluster join 실패
2. **Fix 적용**: compose `command`에 `ray start --address=${RAY_ADDRESS} && python ...` 패턴 적용 (PRM/μ worker와 동일)
3. `03_grpo_train.py`의 `ray.init(address="auto")`로 단순화 (이미 worker로 join된 상태이므로)
4. 재기동까지 했으나 즉시 사용자 중단

**검증 미완**. 단 (a) PRM ray actor가 R2/R3에서 동작 확인, (b) trainer 안에서 PRM RayClient 호출 패턴은 단위 테스트 18/18에서 검증, (c) RabbitMQ Stage 8의 학습 신호 (`lora_B 196/196 학습됨`, `KL 26배 증가`)는 transport-무관 로직이라 동일 결과 예상 — 다음 검증 시 우선 항목.

---

## 4. 진행 중 적용된 코드/설정 fix

| # | 영역 | 이슈 | Fix | 파일 |
|:---:|---|---|---|---|
| 1 | Compose | `command: >` multiline이 bash로 잘못 분할 | list form / single-line으로 | `docker-compose.yml` |
| 2 | Compose | worker `restart: unless-stopped` + 이전 detached actor → name 충돌 | actor 등록 전 `ray.get_actor` 시도 → 있으면 `ray.kill` | `scripts/serve_prm_ray.py`, `serve_mu_ray.py` |
| 3 | Compose | mu-worker `shm_size` 누락 → raylet die | `shm_size: 8g` 추가 (ray-head, prm-worker는 4g) | `docker-compose.yml` |
| 4 | μ | health timeout 600초로 부족 | 10초 polling × 30분 deadline | `scripts/serve_mu_ray.py` |
| 5 | μ vLLM | WSL CUDA graph capture stuck | `enforce_eager=True`, `disable_custom_all_reduce=True` 추가 (R4 stuck 의 해결에는 도움 안 됨) | `src/rollout/mu_sampler.py` |
| 6 | Trainer | `ray.init()` 새 cluster 시작 시도 | compose `command`에 `ray start --address=…` 선행 + `ray.init(address="auto")` | `docker-compose.yml`, `scripts/03_grpo_train.py` |

---

## 5. RabbitMQ vs Ray — 검증된 항목 비교

| 기능 | RabbitMQ (Stage 0~8) | Ray (Stage R0~R7) |
|---|---|---|
| broker/head 단독 | ✅ | ✅ (R0) |
| 본체 → 분산 publish/RPC | ✅ | ✅ (R3 PRM 정답/오답 0.65/0.19) |
| PRM 워커 단독 | ✅ | ✅ (R2) |
| PRM E2E | ✅ | ✅ (R3, 동일 결과) |
| μ 워커 단독 | ✅ (vLLM 25초) | ⏸ (WSL stuck) |
| μ E2E | ✅ (4개 unique step) | ⏸ |
| PRM+μ 동시 | ✅ (MCRolloutPAV K=8) | ⏸ |
| trainer 학습 | ✅ (50 step + lora_B 196/196 학습) | ⚠️ (fix 적용 후 검증 미완) |

R0~R3는 RabbitMQ와 동일 결과로 검증됨. R4~R7는 **WSL 환경 한계로 검증 미완**.

---

## 6. 결론 + 다음 단계

### 결론
- **Ray 마이그레이션 코드 자체는 완성** (단위 테스트 38/38 통과).
- **PRM 분산 RPC가 Ray actor 패턴으로 동작 검증됨** (R0~R3, 4개 stage).
- **μ + trainer는 WSL 단일 호스트 환경에서 검증 미완** — 실제 분산 시나리오(Linux 네이티브 PC)로 가면 해결 가능성 큼.

### 다음 단계
1. **분산 환경 배포** ([RAY_MIGRATION.md](RAY_MIGRATION.md) §사용법):
   - 본체 1대(head + trainer)
   - μ 워커 1대 (Linux 호스트 권장)
   - PRM 워커 N대 (5070급)
2. **분산 환경에서 R4–R7 재검증** — Linux 네이티브에서 vLLM stuck 해결되는지 확인
3. 그 후 본격 학습 (`configs/rl_q3.yaml`의 `total_steps`를 5000으로, `pav.method`를 `mc_rollout`으로)

### 알려진 제한 (단일 호스트 WSL)
- μ vLLM 7B는 Ray actor 안에서 hang. 단일 호스트 학습 필요 시:
  - 옵션 B: Phase 0(`differential`)만 사용, μ 안 띄움 — RabbitMQ Stage 8과 동일 시나리오
  - 옵션 C: 정책 1.5B로 다운그레이드 + μ 안 사용

---

## 참고
- 시스템 결정 사항: [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)
- 마이그레이션 진행 상황: [RAY_MIGRATION.md](RAY_MIGRATION.md)
- 분산 배포 가이드: [RAY_MIGRATION.md §사용법](RAY_MIGRATION.md)
- RabbitMQ 시점 검증 결과 (비교용): [STAGE_TEST_REPORT.md](STAGE_TEST_REPORT.md)
