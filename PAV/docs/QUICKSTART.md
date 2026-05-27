# Quickstart — 2 PC 분산 (3090 학습 + 3090 Ti 추론)

> **단일 PC만 있는 경우**: [Swap Pipeline](#0-단일-pc-swap-pipeline) section 참고.
> 24GB GPU 한 대로 Phase 1 (μ K=16 rollout)까지 가능 — π/PRM/μ를 dynamic swap.

---

## 0. 단일 PC Swap Pipeline (옵션, 24GB GPU 1대만 있을 때)

분산 학습 (2 PC) 대신, **한 GPU에서 π / PRM / μ를 동적 swap**해서 학습.

### 동작 흐름 (한 step)

```
1. π vLLM.generate(prompts)            ← π wake, rollout (KV cache 활성)
2. reward 계산:
   각 trajectory의 각 step마다:
     [π sleep(level=1)] [μ → GPU] μ.generate(K=16) [μ → CPU]
     [PRM → GPU] PRM.score(K+1개) [PRM → CPU]
   → π wake (sleep 풀고 KV cache 다시)
3. forward + backward + GaLore optimizer step
```

| 시점 | GPU에 있는 모델 | 메모리 |
|---|---|---|
| π rollout | π weight + KV cache + GaLore state | ~10 GB |
| μ rollout | π weight (sleep) + μ HF + activations | ~9 GB |
| PRM score | π weight (sleep) + PRM 8bit | ~8 GB |
| forward/backward | π weight + grad + GaLore + activations | ~12 GB |

→ peak **~12 GB / 24 GB** ✅. Phase 1 K=16도 가능.

### 사용

```bash
cd PAV
cp .env.example .env   # PRM_ENDPOINT/MU_ENDPOINT 무시됨 (swap pipeline은 모두 local)

# 빌드 + 시작 (trainer + dashboard 둘 다)
docker compose -f docker-compose.single.yml up -d --build
docker compose -f docker-compose.single.yml logs -f trainer

# Dashboard: http://localhost:8501
```

설정은 [configs/rl_q3_swap.yaml](../configs/rl_q3_swap.yaml) 사용 (vllm.gpu_memory_utilization=0.15).

### Swap Pipeline 구성 파일

| 파일 | 역할 |
|---|---|
| [src/swap/orchestrator.py](../src/swap/orchestrator.py) | SwapOrchestrator — 모델 swap 관리 |
| [src/swap/swap_prm.py](../src/swap/swap_prm.py) | PRM CPU/GPU swap wrapper |
| [src/swap/swap_mu.py](../src/swap/swap_mu.py) | μ HF model wrapper (vLLM 안 씀) |
| [src/swap/reward_fn.py](../src/swap/reward_fn.py) | swap-aware reward function |
| [src/swap/trainer.py](../src/swap/trainer.py) | build_grpo_trainer_swap (vLLM sleep mode) |
| [scripts/03_grpo_train_swap.py](../scripts/03_grpo_train_swap.py) | entry point |
| [configs/rl_q3_swap.yaml](../configs/rl_q3_swap.yaml) | swap 모드 yaml |
| [docker-compose.single.yml](../docker-compose.single.yml) | trainer + dashboard 통합 compose |

### 트러블슈팅 (Swap 전용)

| 증상 | 해결 |
|---|---|
| `vLLM sleep/wake_up 메서드 없음` 경고 | vLLM 버전 0.6+ 필요 (우리 0.7.3 OK). 버전 낮으면 PRM/μ만 swap (π는 GPU 항상 유지) |
| μ HF generate 매우 느림 | μ HF는 vLLM 없이 .generate() 사용 (token-by-token). 정상. Phase 1 step time 60-90초가 이 때문 |
| PRM/μ load 시 GPU OOM | `configs/rl_q3_swap.yaml`의 `vllm.gpu_memory_utilization` 0.15 → 0.12로 낮춤 |

---


현재 default: **π / μ = Qwen2.5-Math-1.5B Full FT**, PRM = Skywork 1.5B int8.

---

## 0. 사전 준비 (두 PC 공통, 1회)

### 0-1) NVIDIA Container Toolkit (두 PC 모두)

설치: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

### 0-2) 네트워크 모드 선택

- **같은 LAN** — 사설 IP로 직접 통신 (가장 단순). FRP 셋업 불필요 → 바로 [Section 1](#1-추론-pc-3090-ti--5070-등--μ--prm-서빙) 진행.
- **다른 위치 / 학습 PC 공인 IP 보유, 추론 PC NAT 뒤** — FRP TCP tunnel 사용 → 아래 0-3) 먼저 진행.

### 0-3) FRP server 셋업 (학습 PC, FRP 모드 사용 시만)

**Section 1 (추론 PC) 에서 `FRPS_TOKEN`, `FRPS_ADDR` 입력이 필요한데 — 이건 학습 PC 에서 미리 frps 띄워야 발급됨**. 순서: 학습 PC frps → 추론 PC frpc → 학습 PC trainer.

#### Step 1: 라우터 포트포워딩 + DDNS

- 라우터: **TCP 7000** → 학습 PC LAN IP
- 방화벽: TCP 7000 inbound 허용
- (공유기 IP 자주 바뀌면) DDNS: DuckDNS / no-ip 등 무료. 예: `myhost.duckdns.org`

도달성 검증 (외부 LTE 등 **다른 네트워크에서**):
```bash
nc -vz <학습PC 공인IP 또는 DDNS> 7000     # "succeeded" 떠야 OK
```

#### Step 2: Config 검증 — frps 만 임시 부팅 (test token)

```bash
cd PAV
cp .env.example .env
FRPS_TOKEN=test-validation FRPS_DASHBOARD_PW=test docker compose up -d frps
sleep 3 && docker logs pav-frps
```

정상 출력:
```
frps tcp listen on 0.0.0.0:7000          ← control port 정상
dashboard listen on 0.0.0.0:7500          ← web UI 정상
```

문제 시 진단:
- `json: unknown field "XXX"` — frp 버전 / [frp/frps.toml](../frp/frps.toml) 키 불일치
- `bind: address already in use` — 7000/7500/18001/18002 점유. `lsof -i :7000`
- `failed to listen` — Docker 권한

#### Step 3: Test 정리 + 본 frps 기동 (실 토큰)

```bash
docker stop pav-frps && docker rm pav-frps

# 실 토큰 발급 (random 32 chars) + dashboard 비번
export FRPS_TOKEN=$(openssl rand -hex 32)
export FRPS_DASHBOARD_PW=<원하는 비번>
echo "FRPS_TOKEN=$FRPS_TOKEN"             # ← 이 값 메모. 추론 PC + Section 2 에서 사용

# frps 만 우선 띄움 (trainer 는 Section 2 에서)
docker compose up -d --build frps
docker logs pav-frps | tail                # tcp listen 7000 + dashboard 7500 확인
curl -s localhost:7500 | head              # dashboard HTML 응답 (admin / FRPS_DASHBOARD_PW)
```

→ 이 시점부터 학습 PC frps 가 추론 PC frpc 접속 대기. Section 1 진행 가능.

---

## 1. 추론 PC — μ + PRM 서빙

### 1-A. Ampere / Ada GPU (3090, 3090 Ti, 4090 등)

PyTorch 2.5.1 + CUDA 12.4 기본 이미지. `docker-compose.inference.yml` 사용.

#### μ + PRM 동시 실행 (24GB+ VRAM)

한 PC에 μ와 PRM을 모두 GPU에 올립니다.

```bash
cd PAV
cp .env.example .env

# 같은 LAN:
docker compose -f docker-compose.inference.yml up -d --build

# FRP TCP tunnel:
FRPS_ADDR=<학습PC 공인IP/DDNS> FRPS_TOKEN=<frps 와 동일 토큰> NODE_NAME=<유일라벨> \
  docker compose -f docker-compose.inference.yml up -d --build
```

VRAM 점유: **~18 GB / 24 GB** (μ 1.5B @ 0.85 → ~20 GB + PRM int8 ~4 GB).

#### μ만 실행

```bash
docker compose -f docker-compose.inference.yml up -d --build mu-server
```

#### PRM만 실행

```bash
docker compose -f docker-compose.inference.yml up -d --build prm-server
```

상태 확인:
```bash
docker compose -f docker-compose.inference.yml logs -f
curl http://localhost:8001/v1/models     # μ vLLM
curl http://localhost:8002/health         # PRM
```

### 1-B. Blackwell GPU (RTX 5070 등) ⭐

PyTorch 2.7+ / CUDA 12.8 필요. `docker-compose.inference-blackwell.yml` 사용.
Dockerfile의 `BLACKWELL=1` 빌드 인자로 자동 분기 (베이스 이미지 + cu128 패키지).

**아키텍처:**
```
┌──────────────┐  HTTP  ┌──────────────┐  HTTP  ┌──────────────┐
│ 학습 PC 3090  │◄──────►│ 5070 PC #1   │       │ 5070 PC #2   │
│ PyTorch 2.5.1 │       │ μ vLLM 전용  │       │ PRM 전용     │
│ 기존 스택 그대로│       │ PyTorch 2.7+ │       │ PyTorch 2.7+ │
└──────────────┘       └──────────────┘       └──────────────┘
```

> 학습 PC(3090)는 기존 스택 그대로 — HTTP 통신이므로 버전 무관.

#### μ + PRM 동시 실행 (24GB+ VRAM)

```bash
cd PAV
cp .env.example .env

# 같은 LAN:
docker compose -f docker-compose.inference-blackwell.yml up -d --build

# FRP TCP tunnel:
FRPS_ADDR=<학습PC 공인IP/DDNS> FRPS_TOKEN=<frps 와 동일 토큰> NODE_NAME=<유일라벨> \
  docker compose -f docker-compose.inference-blackwell.yml up -d --build
```

#### μ만 실행

```bash
FRPS_ADDR=... FRPS_TOKEN=... NODE_NAME=5070-mu-01 \
  docker compose -f docker-compose.inference-blackwell.yml up -d --build mu-server
```

#### PRM만 실행

```bash
FRPS_ADDR=... FRPS_TOKEN=... NODE_NAME=5070-prm-01 \
  docker compose -f docker-compose.inference-blackwell.yml up -d --build prm-server
```

상태 확인:
```bash
docker compose -f docker-compose.inference-blackwell.yml logs -f
curl http://localhost:8001/v1/models     # μ vLLM
curl http://localhost:8002/health         # PRM
```

VRAM 점유: μ+PRM **~22 GB / 24 GB** | μ만 **~10 GB / 12 GB** | PRM만 **~2 GB / 12 GB** ✅

**⚠️ Blackwell 호환성:**
- RTX 5070은 SM 12.0 (Blackwell). 기존 PyTorch 2.5.1은 미지원.
- compose 파일이 `BLACKWELL=1` 빌드 인자로 자동 처리:
  - 베이스 이미지: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`
  - uv sync: `--index-url https://download.pytorch.org/whl/cu128`
  - 환경변수: `TORCH_CUDA_ARCH_LIST=12.0`, `CUDA_MODULE_LOADING=LAZY`

**트러블슈팅 (Blackwell 전용):**

| 증상 | 해결 |
|---|---|
| vLLM 시작 시 `CUDA error: no kernel image` | `BLACKWELL=1` 빌드 인자 누락. `docker compose -f docker-compose.inference-blackwell.yml build` 로 재빌드 |
| μ vLLM OOM (12GB) | `MU_GPU_MEM`을 0.65로 낮춤. `MU_MAX_LEN`을 1024로 축소 |
| PRM OOM | `prm.yaml`의 `quantization`을 `8bit`로 설정 (기본값) |
| 빌드 시 torch 버전 충돌 | `BLACKWELL=1`이면 cu128 인덱스에서 설치. `BLACKWELL=0`(기본)은 기존 cu124 |

---

## 2. 학습 PC (3090) — trainer

```bash
cd PAV
cp .env.example .env
```

### 옵션 A: 같은 LAN (직접 IP)

`.env`에 추론 PC LAN IP 입력:
```bash
PRM_ENDPOINT=http://192.168.1.10:8002
MU_ENDPOINT=http://192.168.1.10:8001
```

기동:
```bash
# 같은 LAN 이라 frps tunnel 불필요 — trainer + dashboard 만 (frps 안 띄움)
docker compose up -d --build trainer dashboard
docker compose logs -f trainer
```

### 옵션 B: FRP TCP tunnel (학습 PC 공인 IP, 추론 PC NAT 뒤)

`.env`에 frps endpoint (default 그대로 — `frps` 컨테이너의 18001/18002):
```bash
PRM_ENDPOINT=http://frps:18002
MU_ENDPOINT=http://frps:18001
```

기동:
```bash
# 사전: 라우터 포트포워딩 TCP 7000 → 학습 PC LAN IP
FRPS_TOKEN=<random-32+chars> docker compose up -d --build   # frps + trainer + dashboard
# 추론 PC 에서 docker compose -f docker-compose.inference.yml up -d (위 Section 1) →
# frpc 가 자동으로 frps 에 등록, mu/PRM 가용 → 학습 즉시 시작 가능

# FRP dashboard — 모든 추론 PC 상태 실시간
# http://localhost:7500 (admin / FRPS_DASHBOARD_PW)
```

VRAM 점유: **~18 GB / 24 GB** (π 1.5B Full FT + GaLore 8bit layerwise + vLLM colocate 0.30 = 7.2 GB).

실시간 학습 진행 보려면 `tmux attach -t <세션명>` (docker logs는 `\r` carriage return 처리 못해 metrics만 보임).

---

## 3. 검증 명령어 모음

| 동작 | 명령 |
|---|---|
| 추론 PC 로그 follow | `docker compose -f docker-compose.inference.yml logs -f` |
| 학습 PC 로그 follow | `docker compose logs -f trainer` |
| PRM health (직접) | `curl http://<inference-ip>:8002/health` |
| μ vLLM 모델 목록 (직접) | `curl http://<inference-ip>:8001/v1/models` |
| **FRP server 로그** | `docker logs -f pav-frps` (frpc 접속/health/LB 상태) |
| **FRP dashboard** | `http://<학습PC>:7500` (admin / FRPS_DASHBOARD_PW) — 모든 추론 PC 실시간 |
| **FRP client 로그 (추론 PC)** | `docker logs -f pav-frpc` (학습 PC 와의 tunnel 상태) |
| **frps 통한 mu/PRM 호출** | `curl http://localhost:18001/v1/models` / `curl http://localhost:18002/health` (학습 PC 에서) |
| **추론 PC mu/PRM 직접** | `curl http://<추론PC LAN IP>:8001/v1/models` (같은 LAN 일 때) |
| GPU 모니터링 | `nvidia-smi -l 1` |
| 학습 중단 | `docker compose stop trainer` |
| 학습 재개 (체크포인트에서) | trainer 컨테이너 재시작 (`resume_from_checkpoint=True` 자동) |
| 추론 서비스 종료 | `docker compose -f docker-compose.inference.yml down` |

---

## 4. yaml 핵심 키 (현재 default 상태)

[configs/prm.yaml](../configs/prm.yaml):
```yaml
mode: remote                       # local → remote 자동 swap
remote:
  endpoint: http://localhost:8002  # PRM_ENDPOINT 환경변수로 override
  timeout: 300                     # K=16 score_batch 안정성 (default 120 → 300)
  batch_size: 16                   # 32는 추론 PC OOM 위험
quantization: 8bit                 # bnb LLM.int8
```

[configs/policy.yaml](../configs/policy.yaml):
```yaml
model_id: Qwen/Qwen2.5-Math-1.5B-Instruct
quantization: none                 # Full FT는 양자화 없음
full_ft: true                      # ⭐ peft_config=None → 모든 weight 학습

mu:
  mode: remote
  model_id: Qwen/Qwen2.5-Math-1.5B-Instruct
  remote:
    endpoint: http://localhost:8001 # MU_ENDPOINT 환경변수로 override
    timeout: 600                    # K=16 generation batch (default 180 → 600)
```

[configs/rl_q3.yaml](../configs/rl_q3.yaml):
```yaml
pav:
  method: mc_rollout               # Phase 1 (μ K개 alternative rollout, counterfactual advantage)
                                   # smoke test는 differential (Phase 0, μ 안 씀)
  K: 16
grpo:
  learning_rate: 5.0e-7            # Full FT 스케일 (LoRA는 5e-6)
  group_size: 4                    # GRPO trajectory 수
  optim: galore_adamw_8bit_layerwise  # ⭐ 24GB GPU + 1.5B Full FT 권장
                                      # paged_adamw_8bit은 step scratch 12GB로 OOM
                                      # CAME은 RAM 폭주 hang (검증 실패)
                                      # adafactor도 작동 (대안, LR 민감)
vllm:
  colocate: true
  gpu_memory_utilization: 0.30     # 7.2 GB — KV cache 여유, π rollout 안정
```

[docker-compose.yml](../docker-compose.yml) environment:
```yaml
PYTHONUNBUFFERED: "1"              # stdout/stderr 즉시 flush (metrics 실시간)
PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"  # CUDA fragmentation 완화 (WSL2는 무시)
```

[추론 PC `.env`](../.env.inference.example):
```bash
MU_GPU_MEM=0.6                     # vLLM KV cache (default 0.25 → 0.6)
```

---

## 5. 흔한 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| trainer가 `Connection refused` | 추론 PC의 vLLM/PRM 서버 아직 not ready. `docker compose logs -f` 로 "Application startup complete" 확인 후 trainer 재시작 |
| trainer 첫 호출에서 timeout | μ 모델 lazy 로드(~1–3 min). `.env`의 `MU_GPU_MEM`/PRM init time 고려. config의 `mu.remote.timeout`/`remote.timeout` 늘리기 |
| `httpx.RemoteProtocolError: Server disconnected` | 추론 PC에서 PRM 또는 μ가 batch 처리 중 OOM/crash. `prm.yaml`의 `remote.batch_size`를 16으로, `.env`의 `MU_GPU_MEM`을 0.5로 낮춤 |
| `PRM_ENDPOINT`가 yaml override 안 됨 | 환경변수가 trainer 프로세스에 전달됐는지 확인. docker면 compose의 `environment:` 섹션, native면 `export PRM_ENDPOINT=...` |
| 학습 PC OOM (`pythonInterface.cpp` 같은 bitsandbytes 에러) | `paged_adamw_8bit`의 step scratch dequantize가 1.5B에서 ~12GB 사용 → OOM. `optim: galore_adamw_8bit_layerwise`로 변경 (혹은 adafactor) |
| 학습 첫 step에서 매우 오래 (~60-90초) hang처럼 보임 | GaLore의 layerwise projection matrix 초기화. 정상 — step 2부터 정상 속도 |
| 학습이 0/50에서 진행 안 됨 (GPU 100% but step 안 늘어남) | CAME optimizer의 RAM 폭주 (96%). `optim: galore_adamw_8bit_layerwise`로 변경 |
| 학습 진행률(`tqdm` progress bar)이 docker logs에 안 보임 | `docker logs`는 line-based(`\n`만 새 줄)라 tqdm의 `\r` 갱신이 학습 끝에 한 번에 flush됨. `tmux attach -t pav`로 직접 보면 실시간 보임. metrics(`logging_steps`)는 정상 출력 |
| μ 응답이 비어있음 | `policy.yaml`의 `mu.step_stop`이 정책 출력 boundary와 안 맞을 수 있음. `["\n\n", "\n"]` 둘 다 시도 |
| 추론 PC vLLM 시작 시 OOM | `.env.inference.example`의 `MU_GPU_MEM` 낮추거나 `MU_MAX_LEN` 줄임 (default 4096) |
| trainer가 `μ HTTP 502` / `ReadTimeout` 반복 (600s 마다 1회), httpx pool 깨진 keepalive 소켓 재사용 | (이전 ZeroTier RELAY 시절 이슈 — FRP 도입으로 거의 발생 안 함). 발생 시: FRP dashboard 에서 frpc 연결 상태 확인 → 죽었으면 추론 PC 의 frpc 컨테이너 재시작. trainer 의 `_reset_client()` 로직이 retry 1~2회 안에 회복 |
| frpc 가 frps 에 접속 못함 (`dial tcp: i/o timeout` 또는 `connection refused`) | 학습 PC 라우터의 TCP 7000 포트포워딩 안 됨, 또는 학습 PC 방화벽 inbound 차단. 외부에서 `nc -vz <학습PC 공인IP> 7000` 으로 도달성 확인 |
| FRPS_TOKEN mismatch (`login to server failed: authorization failed`) | frpc 와 frps 의 토큰 불일치. 둘 다 같은 `FRPS_TOKEN=` 명령어 inline env 로 띄움 |

---

## 6. 분산 최적화 (Phase 1 학습 가속)

Phase 1 (`pav.method: mc_rollout`)은 K=16 μ rollout + ~200 PRM 호출 / step → 직렬 처리 시 step time 60-90초. 적용된 최적화 / 검증된 trade-off:

### 6.1 trajectory 처리는 직렬 (ThreadPool 시도 → 실패)

[src/train/grpo_trainer.py:`_adapt_reward_for_trl`](../src/train/grpo_trainer.py)는 group_size trajectory를 **for loop 직렬**로 처리.

**ThreadPoolExecutor 검증 결과 — 추론 PC vLLM이 concurrent K=16 batch generation을 못 견딤**:

| max_workers | 결과 |
|---|---|
| 4 | step 4에서 `RemoteProtocolError: Server disconnected` |
| 2 | step 6에서 동일 disconnect |
| **1 (직렬)** | **50 step 완주 ✅ (~54분)** |

vLLM의 동시 sequence scheduling/KV cache가 N×K (N=동시 trajectory, K=16)을 못 견디면 응답 중 끊김. 직렬이 안정. 추론 PC vLLM의 `max_num_seqs` 늘리거나 별도 inference replica 추가하면 ThreadPool 활성화 가능하나 현재 stack에서는 직렬 유지.

### 6.2 timeout/batch_size 보수적 조정

| key | 값 | 이유 |
|---|---|---|
| `mu.remote.timeout` | 600s | K=16 batch generation은 추론 PC에서 ~60-120초 |
| `remote.timeout` (PRM) | 300s | score_batch 16 안정 |
| `remote.batch_size` (PRM) | 16 | 32는 PRM 8bit Skywork OOM |
| `vllm.gpu_memory_utilization` | 0.30 | 학습 PC 7.2 GB (KV cache 여유) |
| `MU_GPU_MEM` (추론 PC) | 0.6 | μ KV cache 14.4 GB (n=16 batch 안정) |

### 6.3 가시성 (PYTHONUNBUFFERED + tmux)

docker-compose.yml의 environment:
```yaml
PYTHONUNBUFFERED: "1"   # metrics 즉시 flush
```

tqdm progress bar 실시간 보려면 **`tmux attach -t <세션명>`** (docker logs는 line-based라 `\r` 처리 못 함 → 학습 끝에 한꺼번에 flush).

### 6.4 한 step 시간 분해 (현재 default — 직렬)

```
시간 (초)    0         15        30        45        60        75
──────────────────────────────────────────────────────────────────
trajectory 1 [μ→][PRM→][μ→][PRM→]
trajectory 2                       [μ→][PRM→][μ→][PRM→]
trajectory 3                                              [μ→][PRM→]...
trajectory 4                                                          ...
                                                                       │
backward+opt:                                                          └►[backward]

학습 PC GPU:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████  ← rollout 대기 중 idle
추론 PC GPU:  ▆▆░░▆▆░░▆▆░░▆▆░░▆▆░░▆▆░░▆▆░░▆▆░░▆▆░░▆▆░░░░░░░░░░░░░░░░░░░░░░  ← spike-idle
```

50 step 실측: **~54분** (step time 60-90초, GaLore 첫 step 60-90초 포함).

---

## 7. 학습 logs metrics 해석

매 `logging_steps`마다 docker logs에 출력되는 라인 예시:

```python
{
  'loss': 0.0,
  'grad_norm': 0.0,
  'learning_rate': 1.67e-07,
  'rewards/pav_Q3_reward': 1.017,
  'reward': 1.017,
  'reward_std': 0.869,
  'completion_length': 246.8,
  'kl': 7.96e-06,
  'epoch': 0.01,
}
```

| 필드 | 의미 | 정상 범위 / 해석 |
|---|---|---|
| `loss` | GRPO loss (`= − E[A · log π/π_old] + β·KL`) | **거의 0** — GRPO 특성. group-baseline 빼서 평균 0 근처. **학습 신호는 loss가 아닌 reward 추이로 봐야 함** |
| `grad_norm` | parameter gradient의 L2 norm (clipping 전) | 정상: 0.1 ~ 10. **0.0이면** ① GaLore layerwise는 fused norm을 reporting 안 함 (artifact 가능) ② grad clipping 작동 ③ 진짜 grad 거의 없음 — checkpoint diff로 확인 권장 |
| `learning_rate` | 현재 step의 LR (cosine schedule 적용 후) | `5e-7` (initial) → `0` (final). `warmup_steps=5`로 step 0~5는 ramp-up, 이후 cosine decay |
| `rewards/pav_Q3_reward` | PAV (Q3 risk-seeking mode) reward 평균 | `α=3.0, λ=-0.5`로 계산. trajectory 1개당 `Σ_t [PRM(s_t+a_t) − E_K[PRM(s_t+a'_k)]]`의 mode-weighted 합. **값보다는 step별 추이가 중요** |
| `reward` | reward_funcs 평균 (현재 1개라 `pav_Q3_reward`와 동일) | 동일 |
| `reward_std` | group_size 내 trajectory들 reward 표준편차 | GRPO advantage 정규화에 사용. **0에 가까우면** trajectory들이 다 비슷 (μ가 다양성 못 만듦 또는 K=16 alternative 비슷) → 학습 신호 약함. 0.5~1.5가 healthy |
| `completion_length` | 생성된 completion 평균 token 수 | `max_new_tokens=256` 가까이면 truncate 의심 (논리 미완성). 100~250 normal |
| `kl` | π 와 reference model 사이 KL divergence | `β·KL`이 loss에 추가되어 π가 ref에서 너무 멀리 안 가게 함. **너무 작음 (~1e-5)** = π 거의 안 바뀜 = learning rate 작거나 학습 미미. **너무 큼 (>0.1)** = π가 ref에서 크게 이탈, 안정성 우려 |
| `epoch` | dataset 한 바퀴 진행률 | 50 step smoke = 0.01~0.03 (dataset 매우 큼). 5000 step 본격 학습 = 1 epoch 근처 |

### 주의해야 할 패턴

- `reward`가 random walk처럼 변동만 하고 trend 없음 → **50 step은 학습 효과 보기 너무 짧음**. 5000+ step 필요
- `grad_norm = 0.0`이 지속되고 `kl` 매우 작음 → π 실제로 안 바뀜. **checkpoint weight diff**로 진위 확인
- `reward_std`가 매우 작아짐 → μ가 다양성 잃음. `mu.temperature`나 `top_p` 조정 검토
- `completion_length`가 256 도달 빈번 → `max_new_tokens` 늘림 (메모리 영향)

### 시각화 (3가지 옵션, 자동/수동 둘 다)

학습 시작하면 `outputs/<run_name>/`에 다음 파일들이 자동 적재됨:
- `metrics.jsonl` ← `JsonlMetricsCallback`이 매 log마다 한 줄씩 append
- `runs/` (TensorBoard event 파일) ← HF Trainer 기본
- `checkpoint-*/` ← 학습 중간 weight 저장

#### A. Streamlit 실시간 대시보드 (가장 인터랙티브) ⭐

```bash
# 학습 PC host에서 (Docker 컨테이너 밖)
uv pip install streamlit pandas matplotlib   # 1회만
uv run streamlit run scripts/dashboard.py
# 브라우저: http://localhost:8501
# 또는 다른 머신에서: http://<학습PC IP>:8501
```

- 5초마다 jsonl 재로드 → 학습 진행 실시간 그래프
- reward / reward_std / kl / lr / grad_norm / completion_length 6 panel
- 최근 N step 필터 + raw table view

#### B. TensorBoard (HF Trainer 자동 통합)

```bash
uv pip install tensorboard
uv run tensorboard --logdir outputs/ --bind_all   # :6006
```

브라우저에서 다중 run 비교 가능.

#### C. matplotlib PNG 그래프 (스크립트, 학습 끝나면)

```bash
uv run python scripts/plot_metrics.py
# → outputs/stage8_smoke/plots/overview.png + 개별 png 6개
```

논문/리포트용 정적 그래프 생성.

### wandb 활성화 (cloud)

[configs/rl_q3.yaml](../configs/rl_q3.yaml)의 `logging.wandb_project: pav-rl`로 설정하면 cloud로도 push (tensorboard와 병행 가능). `WANDB_API_KEY` env 필요.

---

## 8. FRP TCP tunnel cluster (5070 × N대 추론, 학습 PC 공인 IP 활용)

학습 PC가 공인 IP 1개 보유 + 추론 PC들이 NAT/CGNAT 뒤에 있어도 FRP single persistent TCP tunnel 로 묶어 분산 학습.

> **이전: ZeroTier mesh + nginx-lb (폐기)** — 둘 다 NAT 뒤면 PLANET RELAY 의 packet loss 로 K=16 long-lived TCP 자주 끊김. MOON root 자체 호스팅도 NAT punching 협상 실패. 학습 PC 공인 IP 있으면 FRP 가 단순/안정.

### 1) 사전 작업 + frps 기동 (학습 PC)

→ [Section 0-3 (FRP server 셋업)](#0-3-frp-server-셋업-학습-pc-frp-모드-사용-시만) 참고. 라우터 포트포워딩 + config 검증 + 실 토큰 기동 단계까지 거기서 다룸.

### 2) 학습 PC trainer 시작

frps 까지 띄운 상태에서 trainer + dashboard 추가:
```bash
FRPS_TOKEN=<위 0-3 에서 발급한 토큰> docker compose up -d --build trainer dashboard
docker compose logs -f trainer
```

이 시점: trainer 시작되지만 추론 PC 등록 전이라 첫 mu/PRM 호출에서 502 → retry 모드. 다음 step (3) 끝나면 자동 학습 시작.

### 3) 5070 추론 PC (× N대)

각 PC에서 (NODE_NAME 만 다르게):
```bash
git clone <repo> && cd PAV
cp .env.example .env                  # MU_MODEL_ID, MU_GPU_MEM 등 조정

# 학습 PC 의 frps 토큰 (학습 PC 에서 띄울 때 사용한 동일 값) 와 공인 IP/DDNS,
# 노드 고유 라벨 inline 전달
FRPS_ADDR=myhost.duckdns.org FRPS_TOKEN=<학습 PC 와 동일 토큰> NODE_NAME=5070-pc-01 \
  docker compose -f docker-compose.inference.yml up -d --build
```

확인:
```bash
docker logs pav-frpc                  # "start proxy success" 보이면 학습 PC 와 tunnel 성공
docker logs pav-mu-server | tail      # vLLM "Application startup complete"
docker logs pav-prm-server | tail     # uvicorn "Application startup complete"
```

### 4) Cluster — 자동 등록 + load balancing

별도 discovery 스크립트 불필요. frpc 가 frps 에 등록하면 자동으로 `mu_cluster` / `prm_cluster` group 에 추가됨.

frps 의 자동 처리:
- **Round-robin** — 학습 PC `localhost:18001` 요청 → 살아있는 mu replica 중 1개 선택
- **Health check** — 매 10초 `/v1/models` (mu), `/health` (prm) HTTP probe. 3회 실패 시 LB pool 에서 자동 제거
- **자동 복구** — 죽었던 frpc 가 다시 올라오면 health 통과 후 자동 pool 추가
- **노드 추가/제거** — 새 추론 PC 띄우거나 끄면 frps 가 즉시 반영, 학습 PC 쪽 무수정

**노드 변경 시**: 그냥 추론 PC 에서 docker compose up/down 하면 끝. nginx config 갱신 같은 작업 0.

### 5) 학습 시작

```bash
FRPS_TOKEN=<같은 토큰> docker compose up -d
docker compose logs -f trainer
```

학습 PC trainer가 `http://frps:18001/18002` → frps → 살아있는 추론 PC frpc → mu/PRM.

### 6) FRP Dashboard (운영 시각화)

```
http://<학습PC>:7500
(로그인: admin / FRPS_DASHBOARD_PW)
```

- 모든 frpc 의 connection 상태 / 트래픽 / 에러 / health 실시간
- proxy group 별 현황 (`mu_cluster` 에 몇 replica 살아있나)

### 장점

- 공인 IP **0개의 추론 PC** 들로 cluster 운영 (학습 PC 만 공인 IP 필요)
- 1대 추론 PC 죽어도 fail-over (frps native load balancing)
- 추론 throughput N배 → 학습 시간 N배 단축
- step time ~50초 → **~5-10초** 가능 (10 replica 기준)
- 학습 PC ↔ 추론 PC 단일 영구 TCP, multiplexed — ZT RELAY 의 packet loss 문제 사라짐

### 보안

- `FRPS_TOKEN` 은 .env 저장 X, 명령어 inline env (예: `FRPS_TOKEN=... docker compose up -d`)
- frps 토큰 일치 안 하는 frpc 는 거부 (인증 실패)
- (선택) frps 에 TLS 적용 가능 (`transport.tls.enable = true`)
- frp dashboard 는 학습 PC 내부망/VPN 만 노출 권장 (admin 비밀번호 noise 보호)

---

## 9. 7B로 확장 (옵션)

| 파일 | 변경 키 |
|---|---|
| `configs/policy.yaml` | `model_id: Qwen/Qwen2.5-Math-7B-Instruct`, `quantization: 4bit`, `full_ft: false`, `mu.model_id` → 7B |
| `configs/rl_q3.yaml` | `learning_rate: 5e-6` (LoRA 스케일), `vllm.colocate: false` |
| `.env.inference.example` (추론 PC) | `MU_MODEL_ID=Qwen/Qwen2.5-Math-7B-Instruct`, `MU_GPU_MEM=0.65` |

→ 학습 PC ~13 GB, 추론 PC ~17 GB (마진 충분).
