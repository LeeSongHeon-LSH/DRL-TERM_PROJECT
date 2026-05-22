# Quickstart — 2 PC 분산 (3090 학습 + 3090 Ti 추론)

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

VRAM 점유: **~8 GB / 24 GB** (μ 1.5B 6 GB + PRM int8 1.7 GB).

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

VRAM 점유: **~17 GB / 24 GB** (π 1.5B Full FT + 8bit Adam + vLLM colocate 0.20).

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
```

[configs/rl_q3.yaml](../configs/rl_q3.yaml):
```yaml
grpo:
  learning_rate: 5.0e-7            # Full FT 스케일 (LoRA는 5e-6)
  optim: paged_adamw_8bit          # ⭐ 24GB GPU Full FT 필수
vllm:
  colocate: true
  gpu_memory_utilization: 0.20     # 4.8 GB만 예약
```

---

## 5. 흔한 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| trainer가 `Connection refused` | 추론 PC의 vLLM/PRM 서버 아직 not ready. `docker compose logs -f` 로 "Application startup complete" 확인 후 trainer 재시작 |
| trainer 첫 호출에서 timeout | μ 모델 lazy 로드(~1–3 min). `.env`의 `MU_GPU_MEM`/PRM init time 고려. config의 `rpc_timeout` 늘리기 |
| `PRM_ENDPOINT`가 yaml override 안 됨 | 환경변수가 trainer 프로세스에 전달됐는지 확인. docker면 compose의 `environment:` 섹션, native면 `export PRM_ENDPOINT=...` |
| 학습 PC OOM | `rl_q3.yaml`의 `vllm.gpu_memory_utilization`을 0.15로 낮춤. 그래도 OOM이면 `grpo.gradient_accumulation` 2로 늘리고 `group_size` 4로 |
| μ 응답이 비어있음 | `policy.yaml`의 `mu.step_stop`이 정책 출력 boundary와 안 맞을 수 있음. `["\n\n", "\n"]` 둘 다 시도 |
| 추론 PC vLLM 시작 시 OOM | `.env.inference.example`의 `MU_GPU_MEM` 낮추거나 `MU_MAX_LEN` 줄임 (default 4096) |

---

## 6. 7B로 확장 (옵션)

| 파일 | 변경 키 |
|---|---|
| `configs/policy.yaml` | `model_id: Qwen/Qwen2.5-Math-7B-Instruct`, `quantization: 4bit`, `full_ft: false`, `mu.model_id` → 7B |
| `configs/rl_q3.yaml` | `learning_rate: 5e-6` (LoRA 스케일), `vllm.colocate: false` |
| `.env.inference.example` (추론 PC) | `MU_MODEL_ID=Qwen/Qwen2.5-Math-7B-Instruct`, `MU_GPU_MEM=0.65` |

→ 학습 PC ~13 GB, 추론 PC ~17 GB (마진 충분).
