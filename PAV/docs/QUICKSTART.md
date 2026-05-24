# Quickstart — 2 PC 분산 (3090 학습 + 3090 Ti 추론)

> **단일 PC만 있는 경우**: [docker-compose.single.yml](../docker-compose.single.yml) 사용.
> Phase 0 (μ 안 씀) + PRM local로 24GB GPU에 모두 fit (~20GB 사용, 마진 4GB).
> 설정: `configs/prm.yaml`에 `mode: local`, `configs/rl_q3.yaml`에 `pav.method: differential` + `vllm.gpu_memory_utilization: 0.20`.
> 실행: `docker compose -f docker-compose.single.yml up -d --build`
> 장점: HTTP RPC 0회 → disconnect/stall 없음, 추론 PC 불필요. 단점: Phase 1 K=16 rollout은 메모리 안 들어감 (LoRA 모드라야 가능).


현재 default: **π / μ = Qwen2.5-Math-1.5B Full FT**, PRM = Skywork 1.5B int8.

---

## 0. 사전 준비 (두 PC 공통, 1회)

- NVIDIA Container Toolkit 설치:
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
- 두 PC가 같은 LAN (RTT < 1 ms 권장, 100 Mbps도 OK)
- 추론 PC의 inbound 포트 **8001 (μ)**, **8002 (PRM)** 열기
- 추론 PC의 IP 확인 — 예: `192.168.1.10`

---

## 1. 추론 PC (3090 Ti) — μ + PRM 서빙

```bash
cd PAV
cp .env.inference.example .env       # 그대로 두면 1.5B μ 자동 적용
docker compose -f docker-compose.inference.yml up -d
```

상태 확인:
```bash
docker compose -f docker-compose.inference.yml logs -f      # μ "ready" + PRM "Application startup complete" 대기
curl http://localhost:8002/health                            # PRM   → {"ok": true, ...}
curl http://localhost:8001/v1/models                         # μ vLLM → {"data": [...]}
```

VRAM 점유: **~18 GB / 24 GB** (μ 1.5B @ MU_GPU_MEM=0.6 → 14.4 GB + PRM int8 ~4 GB).
KV cache가 크게 잡혀야 Phase 1 K=16 batch가 안정적이라 0.6 권장 (이전 0.25는 disconnect 빈발).

---

## 2. 학습 PC (3090) — trainer

```bash
cd PAV
cp .env.example .env
# .env 파일에서 PRM_ENDPOINT / MU_ENDPOINT를 추론 PC IP로 수정:
#   PRM_ENDPOINT=http://192.168.1.10:8002
#   MU_ENDPOINT =http://192.168.1.10:8001

docker compose up -d                  # 또는 native: bash run_train.sh --mode phase1
docker compose logs -f
```

VRAM 점유: **~18 GB / 24 GB** (π 1.5B Full FT + GaLore 8bit layerwise + vLLM colocate 0.30 = 7.2 GB).
GaLore optimizer로 1.5B Full FT가 24GB에 들어옴 (`paged_adamw_8bit`은 step scratch 12GB로 OOM, `CAME`은 RAM 폭주 hang).
실시간 학습 진행 보려면 `tmux attach -t <세션명>` (docker logs는 `\r` carriage return 처리 못해 metrics만 보임).

---

## 3. 검증 명령어 모음

| 동작 | 명령 |
|---|---|
| 추론 PC 로그 follow | `docker compose -f docker-compose.inference.yml logs -f` |
| 학습 PC 로그 follow | `docker compose logs -f` |
| PRM health | `curl http://<inference-ip>:8002/health` |
| μ vLLM 모델 목록 | `curl http://<inference-ip>:8001/v1/models` |
| 학습 PC에서 PRM 직접 호출 (smoke) | `uv run python scripts/00_smoke_prm.py --config configs/prm.yaml` (mode=remote 자동) |
| GPU 모니터링 | `nvidia-smi -l 1` |
| 학습 중단 | `docker compose stop` |
| 학습 재개 (체크포인트에서) | trainer 컨테이너 재시작 (TRL `save_strategy=steps`로 자동 저장된 ckpt 사용) |
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

## 8. 7B로 확장 (옵션)

| 파일 | 변경 키 |
|---|---|
| `configs/policy.yaml` | `model_id: Qwen/Qwen2.5-Math-7B-Instruct`, `quantization: 4bit`, `full_ft: false`, `mu.model_id` → 7B |
| `configs/rl_q3.yaml` | `learning_rate: 5e-6` (LoRA 스케일), `vllm.colocate: false` |
| `.env.inference.example` (추론 PC) | `MU_MODEL_ID=Qwen/Qwen2.5-Math-7B-Instruct`, `MU_GPU_MEM=0.65` |

→ 학습 PC ~13 GB, 추론 PC ~17 GB (마진 충분).
