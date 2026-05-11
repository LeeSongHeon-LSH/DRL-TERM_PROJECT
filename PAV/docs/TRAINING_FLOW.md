# 학습 흐름 다이어그램

PAV-RL 파이프라인의 **시스템 구조 → 한 step 데이터 흐름 → 한 step RPC 시퀀스 → 4주 실행 단계**.
모든 다이어그램은 mermaid (GitHub / VSCode 미리보기에서 자동 렌더).

---

## 1. 시스템 구조 (Ray cluster, 마이그레이션 중)

> 이전 RabbitMQ 구조는 [STAGE_TEST_REPORT.md](STAGE_TEST_REPORT.md)에서 Stage 0~8 검증 끝남(history). Ray 전환 진행 상황 — [RAY_MIGRATION.md](RAY_MIGRATION.md).

```mermaid
flowchart LR
    subgraph Main["PC A 본체: RTX 3090 24GB (Ray Head)"]
        direction TB
        TRAIN["TorchTrainer wrap (Ray Train)<br>GRPOTrainer (TRL)<br>π = Qwen2.5-Math-7B + LoRA r=64"]
        VLLM["vLLM colocate rollout<br>gpu_mem_util=0.30"]
        REWARD["PAVRewardFn<br>(stats/sample buffer)"]
        RPRM["RayPRMClientPool<br>(N replicas 라운드로빈)"]
        RMU["RayMuClient (예정)"]
        TRAIN --- VLLM
        TRAIN --- REWARD
        REWARD --- RPRM
        REWARD --- RMU
    end

    subgraph RAY["Ray Cluster (head: PC A)"]
        direction TB
        HEAD[("Ray Head<br>scheduler + GCS<br>port 6379/8265/10001")]
    end

    subgraph PRMW["PRM Ray Workers (PC C…G: RTX 5070 12GB × 5+)"]
        direction TB
        PW1["RayPRMActor #1<br>@ray.remote(num_gpus=1)<br>Skywork-PRM 1.5B"]
        PW2["RayPRMActor #2"]
        PWN["RayPRMActor #N"]
    end

    subgraph MUW["μ Ray Worker (PC B: RTX 3090 24GB)"]
        MW["RayMuActor<br>Qwen2.5-Math-7B + vLLM"]
    end

    subgraph Data["데이터"]
        GSM["GSM8K train"]
        MN["MathNet eval (n=200)"]
    end

    GSM --> TRAIN
    MN -. "20_eval_mathnet.py" .-> VLLM

    RPRM -- "actor.method.remote() + ray.get()" --> HEAD
    RMU  -- "ray.get()" --> HEAD
    HEAD -- "schedule (round-robin)" --> PW1
    HEAD -- "schedule" --> PW2
    HEAD -- "schedule" --> PWN
    HEAD -- "schedule" --> MW

    classDef main fill:#fff3cd,stroke:#e8a317
    classDef ray fill:#e8eaf6,stroke:#3949ab
    classDef worker fill:#e3f2fd,stroke:#1565c0
    class TRAIN,VLLM,REWARD,RPRM,RMU main
    class HEAD ray
    class PW1,PW2,PWN,MW worker
```

**핵심**:
- 본체는 `actor.method.remote()` + `ray.get()`로 동기 결과 (기존 PRM 인터페이스 보존).
- Ray scheduler가 N replicas에 자동 라운드로빈. `RayPRMClientPool.score_batch`는 N-way 자동 분산.
- 5070 ×5이면 PRM RPC throughput 5배 → 본체 step latency의 PRM 부분 ~1/5로.
- 본체 VRAM 약 22GB로 3090 한 장 (PRM/μ는 0).
- μ는 7B라 별도 3090(24GB) 단일 인스턴스 (PC B).

---

## 1-legacy. 시스템 구조 (RabbitMQ 분산 큐, history)

<details>
<summary>이전 RabbitMQ 다이어그램 (참고용 — 클릭해서 펼치기)</summary>

```mermaid
flowchart LR
    subgraph Main["본체 PC: RTX 3090 24GB"]
        direction TB
        TRAIN["GRPOTrainer (TRL)<br>π = Qwen2.5-Math-7B + LoRA r=64"]
        VLLM["vLLM colocate rollout<br>gpu_mem_util=0.30"]
        REWARD["PAVRewardFn<br>(stats/sample buffer)"]
        RPRM["RemotePRM (pika RPC client)"]
        RMU["RemoteMuSampler (pika RPC client)"]
        TRAIN --- VLLM
        TRAIN --- REWARD
        REWARD --- RPRM
        REWARD --- RMU
    end

    subgraph Broker["메시지 브로커"]
        direction TB
        RQ["RabbitMQ<br>docker compose up -d"]
        Q1[("prm.requests<br>queue")]
        Q2[("mu.requests<br>queue")]
        QR[("reply queues<br>(per-client, exclusive)")]
        RQ --- Q1
        RQ --- Q2
        RQ --- QR
    end

    subgraph PRMW["PRM 워커들 (각 PC에서 동일 명령으로)"]
        direction TB
        PW1["RTX 5070 12GB+<br>serve_prm.py<br>Skywork-PRM 1.5B (~3GB)"]
        PW2["RTX 5070 12GB+<br>serve_prm.py (replica)"]
    end

    subgraph MUW["μ 워커 (16GB+)"]
        MW["serve_mu.py<br>Qwen2.5-Math-7B base"]
    end

    subgraph Data["데이터"]
        GSM["GSM8K train"]
        MN["MathNet eval (English text-only, n=200)"]
    end

    GSM --> TRAIN
    MN -. "20_eval_mathnet.py" .-> VLLM

    RPRM -- "publish (op, payload, reply_to, correlation_id)" --> Q1
    RMU  -- "publish" --> Q2
    Q1 -- "consume (round-robin)" --> PW1
    Q1 -- "consume" --> PW2
    Q2 -- "consume" --> MW
    PW1 -- "publish reply" --> QR
    PW2 -- "publish reply" --> QR
    MW  -- "publish reply" --> QR
    QR -. "consume by correlation_id" .-> RPRM
    QR -. "consume by correlation_id" .-> RMU

    classDef main fill:#fff3cd,stroke:#e8a317
    classDef broker fill:#fce4ec,stroke:#c2185b
    classDef worker fill:#e3f2fd,stroke:#1565c0
    class TRAIN,VLLM,REWARD,RPRM,RMU main
    class RQ,Q1,Q2,QR broker
    class PW1,PW2,MW worker
```

- 본체는 publish만, 워커들이 consume — 워커 추가/제거 시 본체 코드 무수정.
- RabbitMQ가 `prm.requests`/`mu.requests` 큐를 워커들에게 라운드로빈 분산.
- 표준 AMQP RPC 패턴 (`reply_to` + `correlation_id`)으로 응답 매칭.

</details>

---

## 2. 한 GRPO step 안의 데이터 흐름

```mermaid
flowchart TB
    Q["GSM8K 문제 q"]:::data
    Q --> PI["π (LoRA) — vLLM colocate<br>group_size=8 trajectory 생성"]:::main

    PI --> SPLIT["split_steps<br>trajectory를 step 단위로 분할"]:::main
    SPLIT --> LOOP{"각 step h = 1..H<br>prefix s_h, action a_h"}:::main

    LOOP -->|"P0/P1 공통"| PRM_PQ["PRM(s_h + a_h) → p_q<br>(remote)"]:::remote

    LOOP -->|"Phase 0: Differential"| PRM_PV["PRM(s_h) → p_v<br>(remote)"]:::remote
    LOOP -->|"Phase 1 ⭐: MC rollout"| MU_K["μ.sample_step(s_h) × K=16<br>(remote)"]:::remote
    MU_K --> PRM_K["PRM(s_h + a_k) × K → p_v_samples<br>(remote, batch)"]:::remote

    PRM_PQ --> ADV["advantage:<br>A = p_q − p_v (스칼라)<br>또는 A_k = p_q − p_v_k (분포 [K])"]:::main
    PRM_PV --> ADV
    PRM_K --> ADV

    ADV --> RED["reduce_advantage<br>B1 / Q1 / Q3 ⭐ / Q4"]:::main
    RED --> R["r_h = R_ex·𝟙[h=H] + α·Ã_h"]:::main
    R --> SUM["Σ r_h<br>trajectory scalar"]:::main
    SUM --> GRPO["GRPO loss<br>group baseline + KL β=0.04 + clip ε=0.2"]:::main
    GRPO -.->|"LoRA 가중치 업데이트"| PI

    classDef data fill:#f0f0f0,stroke:#666
    classDef main fill:#fff3cd,stroke:#e8a317
    classDef remote fill:#e3f2fd,stroke:#1565c0
```

`pav.method`(`differential` ↔ `mc_rollout`) 한 줄만 바꾸면 위 두 분기 사이를 swap.

---

## 3. 한 step의 RPC 시퀀스 (Phase 1, K=16) — RabbitMQ AMQP

```mermaid
sequenceDiagram
    autonumber
    participant Tr as 본체: GRPOTrainer
    participant Pi as 본체: π vLLM<br>(colocate)
    participant RF as 본체: PAVRewardFn
    participant PRMcli as 본체: RemotePRM
    participant MUcli as 본체: RemoteMuSampler
    participant MQ as RabbitMQ
    participant PRMw as PRM 워커 (1..N)
    participant MUw as μ 워커

    Tr->>Pi: rollout(group_size=8 problems)
    Pi-->>Tr: 8 trajectories

    loop 각 trajectory의 각 step h
        RF->>PRMcli: score(problem, prefix+step)
        PRMcli->>MQ: publish prm.requests (op=score, reply_to, cid)
        MQ->>PRMw: deliver (round-robin)
        PRMw->>MQ: publish reply (cid → reply_queue)
        MQ-->>PRMcli: deliver reply
        PRMcli-->>RF: p_q

        RF->>MUcli: sample_step_batch(problem, prefix, n=16)
        MUcli->>MQ: publish mu.requests (op=sample, n=16, reply_to, cid)
        MQ->>MUw: deliver
        MUw->>MQ: publish reply
        MQ-->>MUcli: deliver reply
        MUcli-->>RF: 16 alt steps

        RF->>PRMcli: score_batch(problem, [prefix+a_k]×16)
        PRMcli->>MQ: publish prm.requests (op=score_batch)
        MQ->>PRMw: deliver
        PRMw->>MQ: publish reply
        MQ-->>PRMcli: deliver reply
        PRMcli-->>RF: p_v_samples [16]

        Note over RF: A_k = p_q − p_v_k<br>reduce_advantage(Q3, λ=−0.5)<br>r_h = R_ex·𝟙[h=H] + α·Ã_h
    end

    RF-->>Tr: trajectory rewards [8]
    Tr->>Tr: GRPO loss + LoRA update
```

**핵심 최적화**:
- **vLLM prefix caching** — 같은 prefix에서 K=16 sampling이 single forward에 가깝게 빠름.
- **score_batch endpoint** — 16개 prefix를 한 AMQP 요청에 묶어 워커가 한 번의 PRM batch forward.
- **워커 풀 라운드로빈** — RabbitMQ가 자동으로 idle 워커에 분산. PRM 워커 N개 띄우면 throughput N배.

---

## 4. 4주 실행 단계와 게이트

```mermaid
flowchart TB
    subgraph W1["W1 — 차분 PAV (Phase 0)"]
        W1a["00_smoke_prm.py<br>PRM toy 점수 검증"]
        W1b["10_label_steps.py<br>GSM8K test → sanity 라벨"]
        W1c["01_phase0_diff.py<br>S1~S4 측정"]
        W1d{{"G0:<br>S1~S4 통과 +<br>BoN-PAV ≥ BoN-PRM"}}:::gate
        W1a --> W1b --> W1c --> W1d
    end

    subgraph W2["W2 — 분포형 PAV (Phase 1) ⭐"]
        W2a["02_phase1_mc.py<br>K ∈ {4,8,16,32} 비교"]
        W2b["BoN-PAV(분포) vs BoN-PAV(스칼라)"]
        W2c{{"G1:<br>분포 ≥ 스칼라 +<br>corr(Q1,Q3) < 0.95"}}:::gate
        W2a --> W2b --> W2c
    end

    subgraph W3["W3 — GRPO 학습 셋업"]
        W3a["보상모델 PC: serve_prm/serve_mu 기동"]
        W3b["본체 3090: 03_grpo_train smoke 1k step"]
        W3c["B1 baseline + Q1 학습"]
    end

    subgraph W4["W4 — 메인 실험 + ablation"]
        W4a["Q3 (λ=−0.5) ⭐ 본 학습"]
        W4b["Q4 (CVaR α=0.2) 학습"]
        W4c["20_eval_mathnet.py<br>pass@1 / pass@N (MathNet)"]
        W4d{{"G2:<br>pass@N +3%p<br>또는 entropy decay 50% 완화"}}:::gate
        W4e{{"G3:<br>A.mean only vs A.mean+std<br>유의차"}}:::gate
        W4a --> W4c
        W4b --> W4c
        W4c --> W4d
        W4c --> W4e
    end

    W1d --> W2a
    W2c --> W3a
    W3a --> W3b --> W3c --> W4a

    classDef gate fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## 5. PAVMethod Protocol 단일화 — 왜 swap이 자유로운가

```mermaid
flowchart LR
    A1["DifferentialPAV<br>A = PRM(s+a) − PRM(s)"]
    A2["MCRolloutPAV<br>A_k = PRM(s+a) − PRM(s+a_k)<br>k=1..K"]
    A3["BetaPosteriorPAV<br>(미래 옵션)"]
    A4["LookaheadPAV<br>(미래 옵션)"]
    A5["EnsemblePAV<br>(미래 옵션)"]

    P[["PAVMethod Protocol<br>(problem, prefix, step) → dict"]]:::protocol
    A1 --> P
    A2 --> P
    A3 --> P
    A4 --> P
    A5 --> P

    P --> R["reduce_advantage<br>B1 / Q1 / Q3 / Q4"]
    R --> RW["PAVRewardFn"]
    RW --> G["GRPOTrainer"]

    PRM_LOC["PRM (local)"]
    PRM_REM["RemotePRM<br>(HTTP)"]
    MU_LOC["MuSampler (local)"]
    MU_REM["RemoteMuSampler<br>(HTTP)"]

    PRM_LOC -. "동일 인터페이스" .-> A1
    PRM_REM -. "동일 인터페이스" .-> A1
    PRM_LOC -. "동일 인터페이스" .-> A2
    PRM_REM -. "동일 인터페이스" .-> A2
    MU_LOC  -. "동일 인터페이스" .-> A2
    MU_REM  -. "동일 인터페이스" .-> A2

    classDef protocol fill:#fff3cd,stroke:#e8a317,stroke-width:2px
```

- **추출 방식 추가** (BetaPosterior, Lookahead, Ensemble …): `PAVMethod` 만족 → RL 코드 0줄 수정
- **로컬 ↔ 원격(RabbitMQ) 분산 swap**: `mode: local|remote` yaml 키 한 줄 → `PAVRewardFn` 0줄 수정
- **transport 변경** (HTTP → AMQP → gRPC …): handlers.py만 그대로 두고 client/worker 교체 → handlers 0줄 수정

---

## 6. 각 워커 / 본체 최소·권장 사양

### 6.1 한눈에 비교

| 컴포넌트 | GPU 최소 | GPU 권장 | CPU | RAM | 디스크 | 비고 |
|---|---|---|---|---|---|---|
| **RabbitMQ broker** | — (CPU만) | — | 2 core | 2GB | 5GB | 분산 시 LAN 1Gbps 권장 |
| **PRM 워커** | 6GB VRAM (RTX 3060 12G / RTX 4060 8G) | 12~16GB (RTX 4060 Ti 16G / **RTX 5070 12G**) | 4 core | 8GB | 10GB | PRM 1.5B fp16 ~3GB. 여러 PC 분산 가능 |
| **μ 워커** | 16GB VRAM (RTX 4060 Ti 16G) | **24GB** (RTX 3090/4090) | 4~8 core | 16GB | 30GB | Qwen2.5-Math-7B bf16 ~14GB + KV cache |
| **본체 (trainer)** | **24GB** (RTX 3090/4090) | 48GB (A6000) ~ 80GB (H100) | 8~16 core | 32GB+ | 100GB+ | π 7B + LoRA + vLLM colocate ≈ 22GB |

> 기준: PRM=1.5B, π/μ=Qwen2.5-Math-7B-Instruct, LoRA r=64, Phase 1 K=16.
> "GPU 최소" = 띄우는 데 OOM 안 나는 한계, "GPU 권장" = batch/throughput 안정.

---

### 6.2 RabbitMQ broker (GPU 불필요)

| 항목 | 최소 | 권장 |
|---|---|---|
| CPU | 2 core | 4 core |
| RAM | 2GB | 4GB |
| 디스크 | 5GB | 10GB (메시지 영속화 시 ↑) |
| 네트워크 | 100Mbps | **1Gbps LAN** |
| OS | Docker 호스트면 무관 (Linux/Windows/Mac) | Linux 권장 |

- 메시지 자체는 작음 (~수 KB) — broker가 throughput 병목이 되는 일은 거의 없음.
- 본체와 같은 PC에 띄워도 무방. 별도 PC면 LAN 라운드트립 ≤2ms 권장.

---

### 6.3 PRM 워커 (Skywork-PRM 1.5B)

| 항목 | 최소 | 권장 |
|---|---|---|
| **GPU** | RTX 3060 12GB / RTX 4060 8GB | **RTX 4060 Ti 16GB / RTX 5070 12GB** |
| VRAM | 6GB (모델 ~3GB + activation + KV) | 12~16GB |
| CPU | 4 core | 8 core |
| RAM | 8GB | 16GB |
| 디스크 | 10GB (모델 + HF 캐시) | 30GB (학습 데이터 캐시 공유 시) |
| 네트워크 | 100Mbps | 1Gbps |

- **여러 PC에 동시 분산 가능** — RabbitMQ가 라운드로빈. N대면 throughput N배.
- prefetch=1 (compose 기본) → 워커 1대당 in-flight 1개. GPU OOM 절대 안 남.
- 모델 가중치 ~2.9GB는 첫 호출 시 lazy 다운 → HF 캐시 영속화로 재기동 비용 0.

---

### 6.4 μ 워커 (Qwen2.5-Math-7B base, vLLM)

| 항목 | 최소 | 권장 |
|---|---|---|
| **GPU** | RTX 4060 Ti 16GB / RTX 4070 12GB(빡빡) | **RTX 3090 24GB / RTX 4090 24GB** |
| VRAM | 16GB (모델 14GB + KV cache 좁게) | 24GB (K=16 batch 안정 + prefix caching 여유) |
| CPU | 4 core | 8 core |
| RAM | 16GB | 32GB |
| 디스크 | 30GB | 50GB |
| 네트워크 | 100Mbps | 1Gbps (응답 길이 ~K개×수십토큰) |

- 7B 모델이라 12GB GPU(RTX 4070)에서는 KV cache 압박 → max_model_len 줄여야 함.
- prefix caching 효과로 K=16 sampling이 K=1과 비슷한 비용.
- μ가 학습 중 변하지 않으므로 weight load는 1회.

---

### 6.5 본체 trainer (3090 24GB 기준)

| 항목 | 최소 | 권장 | 충분 |
|---|---|---|---|
| **GPU** | **RTX 3090 / 4090 24GB** | A6000 Ada 48GB | H100 80GB ×2 |
| VRAM | 22GB (가용 한계) | 40GB+ | 80GB+ (멀티 정책 동시 실험) |
| CPU | 8 core | 16 core | 32 core |
| RAM | 32GB | 64GB | 128GB |
| 디스크 | 100GB (체크포인트 + W&B logs) | 500GB SSD | 1TB NVMe |
| 네트워크 | 1Gbps (broker LAN + W&B push) | 1Gbps | 1Gbps |
| Shared mem | 8GB (`shm_size: 8g` compose 설정) | — | — |

본체 VRAM 분해 (3090 24GB / PRM·μ remote 가정):

| 항목 | 메모리 |
|---|---|
| π base 7B (bf16, frozen) | ~14GB |
| LoRA r=64 + Adam optimizer states | ~1.2GB |
| vLLM colocate rollout (`gpu_mem_util=0.30`) | ~7~9GB |
| **합계** | **~22GB** ✅ |

> 24GB GPU에서 안전 마진은 ~2GB. group_size를 8 → 16으로 늘리거나 `max_completion_length`를 1024로 키우면 OOM 위험.
> RTX 4090 / 3090 모두 ECC 없는 일반 GPU라 장시간 학습 시 ECC 메모리 GPU 권장.

---

### 6.6 분산 토폴로지 예시 3가지

#### A. 미니멈 (1 PC, 단일 GPU 24GB)
```
[3090 24GB] — broker(docker) + PRM worker + μ worker + trainer
```
모든 service `--profile all up`. 단 vLLM colocate가 8~9GB만 가져감 + μ 7B(14GB) + PRM 1.5B(3GB) 동시는 빡빡 → **사실상 Phase 0만 가능**, Phase 1은 K=4 정도.

#### B. 작은 분산 (2 PC) ⭐ 가장 균형
```
[PC 1: 3090 24GB] — broker + trainer
[PC 2: 4060 Ti 16GB] — PRM 워커 + μ 워커  (또는 PRM만)
```
Phase 1 K=16 가능. 본체에 broker 같이 띄움 → LAN 의존성 ↓.

#### C. 풀 분산 (3+ PC, 권장)
```
[PC 1: 미니PC]    — broker
[PC 2: 5070 12GB] — PRM 워커 (replica 가능)
[PC 3: 4090 24GB] — μ 워커
[PC 4: 3090 24GB] — trainer
```
PRM 워커 PC를 늘리면 throughput 선형 증가. broker는 idle 시 자원 거의 0.

---

## 참고
- 시스템 결정 사항: [IMPLEMENTATION_REPORT.md §3](IMPLEMENTATION_REPORT.md)
- 가중치 다운로드: `scripts/download_models.py`
- 분산 모드 사용법: [README.md](../README.md) § "분산 구조"
