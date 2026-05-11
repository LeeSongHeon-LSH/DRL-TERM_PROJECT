# Docker 단계별 테스트 플랜

## 이 테스트의 목적

두 가지 측면을 **분리해서** 검증:

1. **🔵 AMQP 통신** — 본체와 워커들이 RabbitMQ를 통해 publish / consume / reply가 정확히 매칭되어 도착하는가
2. **🟡 실제 추론** — 컨테이너 안에서 PRM(점수)·μ(다음 step rollout) 모델 forward가 합리적인 출력을 내는가
3. **🔴 실제 학습** — trainer 컨테이너에서 GRPO loss가 흐르고 LoRA 가중치가 실제로 변경되는가

본체 PC → RabbitMQ → 워커들 → RabbitMQ → 본체 PC 의 흐름을 **각 화살표 단위로 끊어서** 도커로 검증.
실패 시 어느 지점에서 깨졌는지 한 단계 안에서 좁힐 수 있게 설계.

### 측면별 단계 매트릭스

| Stage | 🔵 AMQP | 🟡 추론 | 🔴 학습 | 메모 |
|---|:---:|:---:|:---:|---|
| 0 | — | — | — | 환경/빌드/import만 |
| 1 broker up | ✓ (idle) | — | — | broker가 5672 listen |
| 2 본체 → broker | ✓ publish | — | — | 큐에 메시지 적재 |
| 3 PRM 워커 단독 | ✓ consume+reply | ✓ PRM forward | — | 가짜 publisher → 실제 점수 |
| 4 본체 ↔ PRM E2E | ✓ round-trip | ✓ PRM forward | — | RemotePRM full RPC |
| 5 μ 워커 단독 | ✓ consume+reply | ✓ μ rollout | — | 실제 step 생성 |
| 6 본체 ↔ μ E2E | ✓ round-trip | ✓ μ rollout | — | RemoteMu full RPC |
| 7 PRM + μ 동시 | ✓ 두 큐 동시 | ✓ PRM + μ | — | MCRolloutPAV 분포 [K] |
| 8 trainer + 학습 | ✓ heavy | ✓ rollout 중 추론 | ✓ LoRA update | smoke 50~1k step |

추론은 Stage 3에서 **처음 등장**, 학습은 Stage 8에서만. 각 단계가 어느 신호를 검증하는지 헤더의 🔵 / 🟡 / 🔴 로 표시했습니다.

### 왜 끊어서 테스트하는가 — VRAM 제약

본체가 **RTX 3090 24GB 한 장**이라 컴포넌트를 한꺼번에 다 띄울 수 없습니다.
모두 동시 적재하면 ~40GB+ 가 필요해 OOM. 따라서 **단계마다 필요한 컨테이너만 켜고, 끝나면 down**.

각 단계의 누적 VRAM:

| Stage | 띄워두는 GPU 컨테이너 | 누적 VRAM | 24GB 한 장? |
|---|---|---:|:---:|
| 0 | (없음, `--rm` 으로 한 번씩) | ~0 | ✅ |
| 1 | broker | 0 (CPU만) | ✅ |
| 2 | broker | 0 | ✅ |
| 3 | broker + **prm-worker** | ~3GB | ✅ |
| 4 | broker + prm-worker | ~3GB | ✅ |
| 5 | broker + **mu-worker** (prm 끄기) | ~14GB | ✅ |
| 6 | broker + mu-worker | ~14GB | ✅ |
| 7 | broker + prm-worker + mu-worker | ~17GB | ✅ |
| 8 | + **trainer** (LoRA 학습 + vLLM colocate ~22GB) | **~40GB** | ❌ OOM |

**단계 전환 패턴** — 다음 단계로 갈 때 이전 단계의 무거운 컨테이너부터 정리:

```bash
# 예: Stage 4 (PRM E2E) → Stage 5 (μ 단독) 전환
docker compose stop prm-worker        # 3GB 회수
docker compose --profile mu up -d     # mu-worker 14GB 새로 적재
```

broker는 VRAM 0이라 끝까지 켜둬도 무방.

### Stage 8의 OOM 우회 (단일 24GB 호스트)

Stage 8은 단일 호스트로는 못 들어가므로 셋 중 하나:

| 옵션 | 방법 | 결과 |
|---|---|---|
| **A. 진짜 분산** | PRM·μ 워커를 다른 PC로 옮김 (분산 시나리오 그대로) | 본체 22GB만 사용 ✅ — **권장** |
| B. Phase 0만 smoke | `pav.method: differential` (μ 안 씀), prm-worker만 켜고 mu-worker 끔 | 본체 22GB + PRM 3GB = 25GB → 빡빡, gpu_mem_util 더 낮춤 |
| C. 더 작은 정책 | 학습 정책을 Qwen2.5-Math-1.5B로 임시 다운그레이드 | trainer 본체 ~6GB, 모든 컨테이너 동시 적재 가능 |

Stage 7까지는 **단일 호스트로 충분히 검증 가능**, Stage 8만 분산이 필요한 게 핵심입니다.

```mermaid
flowchart LR
    subgraph S0["Stage 0: 빌드 + 기동 smoke (6 sub-steps)"]
        direction TB
        S0_1["0.1 호스트 환경<br>(GPU / driver / docker / runtime)"]
        S0_2["0.2 .env 설정"]
        S0_3["0.3 모든 이미지 빌드<br>pav-rl:latest 공유 (prm/mu/trainer)"]
        S0_4["0.4 컨테이너 import smoke<br>(모델 X, src 모듈만)"]
        S0_5["0.5 GPU 패스스루 검증<br>(컨테이너 안 nvidia-smi + torch.cuda)"]
        S0_6["0.6 정리<br>compose --profile all down"]
        S0_1 --> S0_2 --> S0_3 --> S0_4 --> S0_5 --> S0_6
    end

    subgraph S1["Stage 1 🔵: broker"]
        S1a[("RabbitMQ 컨테이너")]
    end

    subgraph S2["Stage 2 🔵: 본체 → broker"]
        H1["본체 venv가 publish<br>(pika 스크립트)"]
    end

    subgraph S3["Stage 3 🔵🟡: PRM 워커 단독"]
        W1["broker → PRM worker(컨테이너) → broker<br>실제 PRM forward 점수 검증"]
    end

    subgraph S4["Stage 4 🔵🟡: 본체 ↔ PRM E2E"]
        E1["RemotePRM.score → broker → 워커 → reply → 본체<br>점수 라운드트립"]
    end

    subgraph S5["Stage 5 🔵🟡: μ 워커 단독"]
        W2["broker → μ worker(컨테이너) → broker<br>실제 step 생성 검증"]
    end

    subgraph S6["Stage 6 🔵🟡: 본체 ↔ μ E2E"]
        E2["RemoteMu.sample → broker → 워커 → reply → 본체<br>K개 alt step 라운드트립"]
    end

    subgraph S7["Stage 7 🔵🟡: 두 워커 동시 (Phase 1 한 step)"]
        ALL["MCRolloutPAV 1회 → PRM + μ 양쪽 RPC<br>advantage 분포 [K]"]
    end

    subgraph S8["Stage 8 🔵🟡🔴: trainer 통합 + 학습"]
        TR["GRPO smoke 50~1k step<br>LoRA gradient 흐름 + weight 변경 검증"]
    end

    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8

    classDef amqp fill:#cfe5ff,stroke:#1565c0
    classDef infer fill:#fff3cd,stroke:#e8a317
    classDef train fill:#ffcdd2,stroke:#c62828
    class S1,S2 amqp
    class S3,S4,S5,S6,S7 infer
    class S8 train
```

색 의미: 🔵 파랑 = AMQP만, 🟡 노랑 = AMQP + 실제 추론, 🔴 빨강 = AMQP + 추론 + 학습.

각 단계는 **이전 단계가 통과했음을 가정**. 한 단계 실패 시 그 단계의 "흔한 원인"으로 디버깅하고, 다음 단계는 시도하지 않습니다.

---

## Stage 0 — 모든 워커의 이미지 빌드 + 기동 smoke

5개 sub-step. **각 워커가 컨테이너 안에서 시작은 된다**는 사실을 보장 (실제 모델 로딩 검증은 Stage 3/5/8).

### 0.1 호스트 환경 확인

| 항목 | 명령 | 통과 기준 |
|---|---|---|
| GPU + driver | `nvidia-smi` | RTX 3090 24GB 인식, CUDA ≥ 12.4 |
| Docker + nvidia runtime | `docker info \| grep -E "Server Version\|Runtimes"` | `nvidia` runtime 등록 |
| Docker Compose v2 | `docker compose version` | v2.x |
| 디스크 여유 | `df -h /var/lib/docker` (또는 Docker Desktop 설정) | ≥ 50GB free (이미지 + HF 캐시) |

**흔한 원인** — Docker Desktop WSL 통합 꺼짐 / NVIDIA Container Toolkit 미설치 / WSL2 GPU passthrough 비활성.

### 0.2 `.env` 설정

```bash
cp .env.example .env
# 단일 호스트면 그대로 두고, 분산이면 AMQP_URL을 broker IP로 수정
```

| 키 | 값 (단일 호스트) | 분산 시 |
|---|---|---|
| `AMQP_URL` | `amqp://guest:guest@rabbitmq:5672/` | `amqp://guest:guest@<broker-host>:5672/` |
| `HF_HOME_HOST` | (빈칸 → `./.hf_cache`) | 호스트 절대 경로 권장 |
| `HF_TOKEN` | (선택) | 동일 |
| `WANDB_API_KEY` | (trainer 기동 시만 필요) | 동일 |

### 0.3 모든 이미지 빌드

```bash
docker compose build              # 4개 service 중 GPU 3개 (prm/mu/trainer)는 같은 pav-rl:latest 공유
```

| 검증 | 통과 기준 |
|---|---|
| `docker images pav-rl:latest` | tag 존재, 크기 ~15~20GB |
| `docker images rabbitmq:3.13-management` | (자동 pull 안 했으면 `docker pull rabbitmq:3.13-management`) |
| 빌드 로그 마지막 | `=> => writing image sha256:...` |
| service별 이미지 동일성 | `docker compose images` → prm-worker / mu-worker / trainer 모두 같은 IMAGE ID |

> 빌드 시간 첫 회 15~25분 (torch + vLLM + trl + peft 다운로드). 두 번째부터는 cache hit으로 분 단위.
> 빌드 진행이 안 보이면 `docker compose build --progress=plain` 으로 stdout에 단계별 출력.

**흔한 원인** — base 이미지 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` pull 실패 (네트워크 / Docker Hub 인증) / `uv sync --extra gpu` 실패 (lock 파일 mismatch) / disk full.

### 0.4 각 컨테이너 기동 smoke (모델 로딩 X, Python import만)

각 워커 image가 `docker compose run`으로 정상 부팅 + 우리 src 모듈을 import 할 수 있는지만 검증.

```bash
# PRM 워커 image — handlers / worker 모듈 import
docker compose run --rm --no-deps prm-worker \
    python -c "from src.prm.remote_worker import serve, WorkerSettings; \
               from src.prm.handlers import handle_request; \
               print('prm-worker imports OK')"

# μ 워커 image
docker compose run --rm --no-deps mu-worker \
    python -c "from src.rollout.mu_worker import serve, WorkerSettings; \
               from src.rollout.mu_handlers import handle_request; \
               print('mu-worker imports OK')"

# trainer image
docker compose run --rm --no-deps trainer \
    python -c "from src.train.grpo_trainer import build_grpo_trainer; \
               from src.train.policy_data import build_policy, build_train_dataset, build_eval_dataset; \
               from src.train.callbacks import PAVMonitorCallback; \
               print('trainer imports OK')"
```

| 검증 | 통과 기준 |
|---|---|
| 각 명령 stdout 마지막 줄 | `... imports OK` |
| exit code | 0 |
| 컨테이너 자동 정리 | `--rm` 덕분에 끝나면 사라짐 |

**흔한 원인** — `PYTHONPATH=/app` 미설정 / `/app/src` 누락 / `uv sync`가 venv를 다른 경로에 만들어 `/app/.venv/bin/python`이 PATH에 없음.

### 0.5 GPU 컨테이너 안에서 보이는지

```bash
docker compose run --rm --no-deps prm-worker nvidia-smi
docker compose run --rm --no-deps mu-worker  nvidia-smi
docker compose run --rm --no-deps trainer    nvidia-smi
```

| 검증 | 통과 기준 |
|---|---|
| 각 출력 | RTX 3090 24GB가 컨테이너 안에서도 표시 |
| `torch.cuda.is_available()` | True (아래 명령) |

```bash
docker compose run --rm --no-deps trainer \
    python -c "import torch; print('cuda:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0))"
```

**흔한 원인** — `docker compose run`이 `deploy.resources.reservations.devices`를 무시할 수 있음 (compose 버전에 따라). 그 경우 임시로 `docker run --gpus all pav-rl:latest nvidia-smi` 로 직접 확인. compose가 GPU 못 잡으면 daemon.json의 `"default-runtime": "nvidia"` 추가 또는 `services.<>.runtime: nvidia` 명시.

### 0.6 정리

```bash
docker compose --profile all down                # 떠 있는 컨테이너 정리 (이미지/볼륨은 보존)
```

→ Stage 0 통과: **빌드 OK + GPU 패스스루 OK + Python import OK**. 다음 단계부터 실제 워커 기동.

---

## Stage 1 🔵 — broker 띄우기 (AMQP 기반 가용)

> 🟢 켬: broker — 🔴 끔: (없음) — 누적 VRAM: 0

```bash
docker compose --profile broker up -d
```

| 검증 | 통과 기준 |
|---|---|
| `docker compose ps` | `pav-rabbitmq` Up |
| `docker logs pav-rabbitmq \| grep "Server startup complete"` | 출력 있음 |
| Management UI | `http://localhost:15672` (guest/guest) 로그인 가능 |
| AMQP 포트 | `nc -zv localhost 5672` OK |

**흔한 원인** — 5672/15672 포트 충돌 / Docker network 충돌.

---

## Stage 2 🔵 — 본체 → broker (publish 도달 검증)

> 🟢 켬: broker — 🔴 끔: (없음) — 누적 VRAM: 0

본체 venv에서 단순 publish가 broker에 도달하는지만 검증. 워커 없음.

```bash
cd PAV
uv run python - <<'PY'
import pika, json, uuid
conn = pika.BlockingConnection(pika.URLParameters("amqp://guest:guest@localhost:5672/"))
ch = conn.channel()
ch.queue_declare(queue="prm.requests", durable=False)
ch.basic_publish(
    exchange="", routing_key="prm.requests",
    properties=pika.BasicProperties(correlation_id=str(uuid.uuid4())),
    body=json.dumps({"op": "health"}).encode(),
)
print("published")
conn.close()
PY
```

| 검증 | 통과 기준 |
|---|---|
| 위 스크립트 | "published" 출력, 에러 없음 |
| Management UI → Queues → `prm.requests` | "Ready" 메시지 1건 (consumer 없으니 큐에 적체) |
| `docker exec pav-rabbitmq rabbitmqctl list_queues name messages` | `prm.requests 1` |

**흔한 원인** — `AMQP_URL` 잘못 / WSL ↔ Docker Desktop 네트워크 (`localhost` vs `host.docker.internal`).

---

## Stage 3 🔵🟡 — PRM 워커 컨테이너 단독 (실제 추론 첫 검증)

> 🟢 켬: broker + **prm-worker** — 🔴 끔: (없음) — 누적 VRAM: ~3GB

워커가 큐의 메시지를 consume하고 reply queue로 답할 수 있는지만 검증. 본체는 가짜 publisher 역할.

```bash
docker compose --profile prm up -d
docker compose logs -f prm-worker          # "Worker ready" 라인까지 대기
```

검증용 가짜 publisher (본체 venv):
```bash
uv run python - <<'PY'
import pika, json, uuid, time
conn = pika.BlockingConnection(pika.URLParameters("amqp://guest:guest@localhost:5672/"))
ch = conn.channel()
result = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
reply_q = result.method.queue
out = {}
def _cb(c, m, p, b): out["body"] = b; out["cid"] = p.correlation_id
ch.basic_consume(queue=reply_q, on_message_callback=_cb, auto_ack=True)

cid = str(uuid.uuid4())
ch.basic_publish(
    exchange="", routing_key="prm.requests",
    properties=pika.BasicProperties(reply_to=reply_q, correlation_id=cid),
    body=json.dumps({"op": "health"}).encode(),
)
t0 = time.time()
while "body" not in out and time.time()-t0 < 30:
    conn.process_data_events(time_limit=0.1)
print(out)
conn.close()
PY
```

| 검증 | 통과 기준 |
|---|---|
| `docker compose logs prm-worker` | `Worker ready. Listening on queue 'prm.requests'` |
| 위 health 호출 | `{"ok": true, "name": "skywork-o1-prm", ...}` 응답 |
| 빈 prefix score 호출 (`{"op":"score", "problem":"...", "solution_prefix":""}`) | `{"score": 0.5}` (PRM의 빈 prefix 0.5 fallback) |

**흔한 원인** — 모델 가중치 다운로드 실패 / GPU 패스스루 미반영 (`docker run --gpus all` vs compose `deploy.devices`) / Skywork 1.5B의 `Qwen2ForRewardModel` 호환 이슈 (현재 보류 중) / VRAM 부족.

---

## Stage 4 🔵🟡 — 본체 ↔ PRM 워커 E2E (RPC 라운드트립)

> 🟢 켬: broker + prm-worker (Stage 3 그대로) — 🔴 끔: (없음) — 누적 VRAM: ~3GB

`RemotePRM` 인터페이스가 실제 워커와 정상 RPC 라운드트립을 도는지.

```bash
uv run python scripts/_smoke_remote_prm.py \
    --amqp-url amqp://guest:guest@localhost:5672/ \
    --queue prm.requests
```

| 검증 | 통과 기준 |
|---|---|
| `health` | `ok=true` |
| `score(toy 정답 prefix)` | 점수 (보통 > 0.5) |
| `score(toy 오답 prefix)` | 점수 (보통 < 정답 점수) |
| `score_batch(4개)` | shape [4], 단일 호출보다 약 4배 안 느림 |
| `score_per_step(full solution)` | step 수만큼 list 반환 |

**흔한 원인** — correlation_id 미매칭 / reply_queue exclusive 충돌 / RPC timeout 너무 짧음.

---

## Stage 5 🔵🟡 — μ 워커 컨테이너 단독 (실제 step 생성)

> 🟢 켬: broker + **mu-worker** — 🔴 **끔: prm-worker** (3GB 회수) — 누적 VRAM: ~14GB

```bash
docker compose stop prm-worker             # ⚠️ Stage 4까지 켜둔 PRM 끄고 VRAM 회수
docker compose --profile mu up -d
docker compose logs -f mu-worker           # "μ worker ready" 까지 대기
```

가짜 publisher (Stage 3과 동일 패턴, queue를 `mu.requests`로 변경):
```python
# body: {"op": "sample", "problem":"...", "prefix":"...", "n": 4}
# expected reply: {"steps": ["...", ..., "..."]}
```

| 검증 | 통과 기준 |
|---|---|
| `docker compose logs mu-worker` | `Loading μ weights (Qwen/Qwen2.5-Math-7B-Instruct)…` 후 ready |
| health → `{"ok": true, "model_id": "Qwen/..."}` |
| sample(n=4) → 4개 string 반환, 각 짧은 reasoning step |

**흔한 원인** — 7B base ~14GB라 12GB GPU에서 OOM / vLLM build 호환 (CUDA, torch 버전) / step_stop 토큰 미인식.

---

## Stage 6 🔵🟡 — 본체 ↔ μ 워커 E2E (RPC 라운드트립)

> 🟢 켬: broker + mu-worker (Stage 5 그대로) — 🔴 끔: (없음) — 누적 VRAM: ~14GB

```bash
uv run python - <<'PY'
import sys; sys.path.insert(0, ".")
from src.rollout.remote_mu import RemoteMuConfig, RemoteMuSampler
cli = RemoteMuSampler(RemoteMuConfig(
    amqp_url="amqp://guest:guest@localhost:5672/", request_queue="mu.requests"))
print(cli.health())
steps = cli.sample_step_batch("Solve x^2=9.", "", n=4)
for i, s in enumerate(steps): print(f"[{i}] {s!r}")
cli.close()
PY
```

| 검증 | 통과 기준 |
|---|---|
| 4개 step 모두 길이 > 10자 |
| 4개가 서로 다름 (temperature=1.0) |
| 1회 호출 latency < 5초 (3090 기준) |

**흔한 원인** — chat template 미반영 → 빈 출력 / vLLM prefix caching 비활성 → 느림.

---

## Stage 7 🔵🟡 — 본체 + PRM + μ 동시 (Phase 1 한 step, 두 큐 동시)

> 🟢 켬: broker + **prm-worker (재시작)** + mu-worker — 🔴 끔: (없음) — 누적 VRAM: ~17GB

```bash
docker compose --profile prm up -d         # ⚠️ Stage 5에서 끈 PRM 다시 켬 (3GB 추가)
# (mu-worker는 Stage 5/6에서 켜진 상태 그대로)
uv run python - <<'PY'
import sys; sys.path.insert(0, ".")
from src.prm import load_prm
from src.rollout.mu_sampler import build_mu_from_policy_yaml
from src.pav import MCRolloutPAV

# 본체에서 모드 remote로 강제 (yaml은 그대로 둬도 무방)
import yaml
prm = load_prm("configs/prm.yaml", mode="remote",
               amqp_url="amqp://guest:guest@localhost:5672/", request_queue="prm.requests")
# μ는 policy.yaml의 mu.mode를 remote로 하거나 아래처럼 직접
# (간략화 — 실제로는 build_mu_from_policy_yaml + yaml 수정)
from src.rollout.remote_mu import RemoteMuConfig, RemoteMuSampler
mu = RemoteMuSampler(RemoteMuConfig(
    amqp_url="amqp://guest:guest@localhost:5672/", request_queue="mu.requests"))

pav = MCRolloutPAV(prm, mu, K=8)
out = pav("Solve x^2=9.", "", "Step 1: x = ±3.\n")
print("A_samples:", out["advantage_samples"].tolist())
print("A_mean:", out["advantage_scalar"].item())
PY
```

| 검증 | 통과 기준 |
|---|---|
| K=8 advantage_samples 정상 반환 | shape [8] |
| 정답 step일 때 mean > 0 (대체 step보다 PRM 점수가 높음) |
| Management UI에서 두 큐(`prm.requests`, `mu.requests`) 모두 메시지가 흘러간 흔적 |

**흔한 원인** — `MCRolloutPAV.__call__`가 `mu.sample_step_batch`만 호출 — 시그니처 일치 확인 / `score_batch`가 한 RPC로 묶이는지 확인.

---

## Stage 8 🔵🟡🔴 — trainer 통합 + 학습 신호 검증

> 🟢 켬: broker + prm-worker + mu-worker + **trainer** — ⚠️ **단일 24GB 호스트는 OOM** (40GB 필요)<br>
> → 위 §"Stage 8 OOM 우회" 옵션 A/B/C 중 선택 후 진행

### A. 분산 (권장) — PRM·μ 워커를 다른 PC로

```bash
# 본체에서 PRM/μ 컨테이너 끄고 (VRAM 회수)
docker compose stop prm-worker mu-worker

# 다른 PC에서 동일 compose로 워커 띄움 (.env의 AMQP_URL을 본체 IP로)
# (PC 2)  docker compose --profile prm up -d
# (PC 3)  docker compose --profile mu  up -d

# 본체에서 trainer만 띄움 → 22GB
docker compose --profile trainer up -d
docker compose logs -f trainer
```

### B. Phase 0만 smoke (단일 호스트)

```bash
# configs/rl_q3.yaml에서 일시적으로:
#   pav.method: differential        # μ 안 씀 → mu-worker 끔
docker compose stop mu-worker        # 14GB 회수
docker compose --profile trainer up -d
# trainer 22GB + prm-worker 3GB = 25GB → 빡빡, configs/rl_q3.yaml의 vllm.gpu_memory_utilization 0.25로 더 낮춤
```

### C. 1.5B 정책으로 임시 다운그레이드

```bash
# configs/policy.yaml에서:
#   model_id: Qwen/Qwen2.5-Math-1.5B-Instruct
#   mu.model_id: 동일
docker compose --profile all up -d
# trainer ~6GB + mu ~3GB + prm 3GB = ~12GB ✅
```

이 단계의 핵심 — **LoRA가 실제로 학습되는지** 가시적으로 보여야 함.

| 검증 측면 | 명령 / 산출물 | 통과 기준 |
|---|---|---|
| 🔵 AMQP throughput | Management UI → `prm.requests`, `mu.requests` 큐 메시지 rate | 0이 아닌 publish/consume rate, idle 큐 적체 < 100 |
| 🟡 추론 (rollout) | `docker compose logs trainer` | step별 trajectory 생성 + reward signal 매 step 도착 |
| 🟡 PAV 신호 | W&B `pav/A_mean`, `pav/A_std`, `pav/p_q_mean` | 0 아닌 값, std > 0 (분포 신호 살아있음) |
| 🔴 **학습 (loss)** | W&B `train/grpo_loss`, `train/policy_kl` | 50 step 이내에 loss 단조 감소 추세 (반드시 monotonic은 X, 평균적 ↓) |
| 🔴 **학습 (LoRA weight 변경)** | 학습 전/후 LoRA `.safetensors`의 norm 비교 (아래 스크립트) | weight L2 norm이 step 0과 step N 사이에 측정 가능한 차이 |
| 🔴 GPU 메모리 | 본체 `nvidia-smi -l 1` | ≤ 22GB (24GB 카드 안전선) |

LoRA weight 변경 검증 스크립트 (학습 전/후 LoRA 디렉토리에서 실행):
```python
# scripts/_check_lora_changed.py
import torch
from safetensors.torch import load_file
import sys
before = load_file(sys.argv[1])  # 학습 전 LoRA (step 0)
after  = load_file(sys.argv[2])  # 학습 후 LoRA (step N)
total_diff = 0.0; total_norm = 0.0
for k in before:
    diff = (after[k] - before[k]).norm().item()
    total_diff += diff
    total_norm += before[k].norm().item()
print(f"L2(diff) / L2(before) = {total_diff/total_norm:.4f}")
# 0.001 이상이면 학습이 실제로 weight를 변경 중
```

> smoke 단계라 학습 step 50~100이면 충분 (수렴이 아니라 *학습 신호 자체가 흐름*을 검증).

**흔한 원인** — vLLM colocate gpu_mem_util 너무 큼(0.30 초과 시 OOM) / TRL ↔ peft 버전 호환 / chat template 적용 실패 / reward function이 항상 0 반환(verifier가 ground truth 못 매칭)<br>→ 학습이 안 됨의 가장 흔한 원인은 **reward가 모두 0** 또는 **모두 같은 값** — `pav/A_std=0`이면 group baseline이 advantage 0으로 만들어 loss가 흐르지 않음.

---

## 단계 통과 후 보존 / 정리

### 전환 시 VRAM 회수 패턴

| 전환 | 명령 |
|---|---|
| Stage 4 → 5 | `docker compose stop prm-worker` (3GB 회수, mu-worker 띄울 자리) |
| Stage 6 → 7 | `docker compose --profile prm up -d` (PRM 다시 켬, mu는 그대로) |
| Stage 7 → 8 | (옵션 A) `docker compose stop prm-worker mu-worker` 후 다른 PC 워커 사용<br>(옵션 B/C) configs 일부 수정 후 trainer up |

> `stop`은 컨테이너만 멈추고 이미지/볼륨/HF 캐시는 보존 — 다음 단계에서 즉시 재기동 가능.
> 워커 모델 가중치는 HF 캐시(호스트 bind mount)에 있으므로 stop/start 비용은 모델 로딩 시간(~30초)뿐.

### 최종 정리

| 시점 | 명령 | 보존 |
|---|---|---|
| 매 단계 사이 | `docker compose stop <service>` | 모든 데이터/캐시/이미지 |
| 모든 단계 끝 | `docker compose --profile all down` | broker volume(`rabbit_data`), HF 캐시(`./.hf_cache`), 이미지 |
| 완전 초기화 | `docker compose down -v && docker rmi pav-rl:latest` | (HF 캐시는 호스트라 별도 삭제) |

각 단계 사이에 broker는 **계속 켜두는 게 편함** (VRAM 0, 큐 재생성 비용 0).
