# 구현 보고서 — PAV-RL (Phase 0 + Phase 1)

작성일: 2026-05-10 (분산 셋업 갱신: 2026-05-20)
대상 계획서: [구현 계획 — 차분 PAV + 분포형 보상 (Phase 0) 4dc18b40f5ed47259166ff0f0c0f8086.md](구현%20계획%20—%20차분%20PAV%20+%20분포형%20보상%20(Phase%200)%204dc18b40f5ed47259166ff0f0c0f8086.md)

---

## 1. 요약

천공 PRM(Skywork-o1-Open-PRM-Qwen-2.5-1.5B; 7B 변형도 호환)을 활용하는 **차분 PAV(Phase 0) + 분포형 MC-rollout PAV(Phase 1)** 파이프라인의
코드 골격을 모두 작성하고, 다음을 검증했습니다.

- **단일 `PAVMethod` Protocol** — 두 추출 방식이 동일 인터페이스를 만족, RL 코드 수정 없이 swap.
- **천공 PRM 정합성** — Skywork 공식 inference 레포에서 `PRM_MODEL` + IO 헬퍼를 vendoring하여
  step boundary(`\n`) / sigmoid value head 추출 / batch padding을 정확히 재현.
- **TRL ≥0.13 GRPO API** — `vllm_mode="colocate"`, `peft_config`, dataset 컬럼 → reward kwargs 통신을
  실제 시그니처에 맞춰 구현.
- **W&B 로깅 + 함정 모니터링 콜백** — PAV 통계 / 분포 corr / sample dump를 자동화.
- **회귀 테스트 19/19 통과** — uv venv (torch 2.5.1+cu124, transformers 4.46) 위에서.
- **분산 옵션 (HTTP transport)** — μ + PRM을 다른 PC로 분리. trainer는 단일 24 GB GPU에서도
  1.5B Full FT (default) 또는 7B QLoRA 가능. weight broadcast 0, RPC 부하 ~6 MB/step (100 Mbps도 충분).
- **PRM 8-bit 양자화** — bitsandbytes LLM.int8()로 Skywork 1.5B를 ~1.7 GB로 압축.
- **현재 default: π/μ = Qwen2.5-Math-1.5B-Instruct + Full FT** (`full_ft: true`, `optim: galore_adamw_8bit_layerwise`).
  7B + QLoRA로 확장은 yaml 키 변경만으로 가능.
- **Optimizer 검증 결과** (1.5B Full FT, 24GB GPU):
  - ❌ `paged_adamw_8bit`: step scratch dequantize 12GB → OOM
  - ❌ `CAME`: factored variance + confidence guidance가 RAM 96% 폭주, step 0/50 hang
  - ✅ `adafactor`: factored variance, state ~1.5GB, 50 step 596초 완주 (Phase 0)
  - ✅ `galore_adamw_8bit_layerwise` ⭐: gradient low-rank projection + 8bit + layer 단위, 50 step 553초 (Phase 0). 권장
- **분산 최적화**:
  - `_adapt_reward_for_trl`는 **직렬 처리** (ThreadPoolExecutor 2/4 동시 시도 → 추론 PC vLLM이 concurrent K=16 batch generation 못 견디고 `RemoteProtocolError: Server disconnected`, 직렬이 안정).
  - HTTP timeout: μ 600s / PRM 300s (K=16 batch 안정성).
  - 추론 PC `MU_GPU_MEM=0.6` (14.4GB) — KV cache 여유로 n=16 batch generation 안정.
  - 학습 PC `vllm.gpu_memory_utilization=0.30` (7.2GB) — π colocate rollout.
- **가시성**: `PYTHONUNBUFFERED=1` (docker-compose env)로 metrics 실시간 flush. tqdm progress bar는 `\r` carriage return이라 `docker logs`엔 한꺼번에 flush → `tmux attach`로 직접 봐야 실시간.

가중치만 받으면 `00_smoke_prm → 10_label_steps → 01_phase0_diff → 02_phase1_mc → 03_grpo_train`
순서로 즉시 학습 진입 가능합니다.

---

## 2. 디렉토리

```
PAV/
├── pyproject.toml                 # uv canonical (torch 2.5.1+cu124 + extras=gpu,serve,awq)
├── .python-version                # 3.11
├── README.md
├── run_train.sh                   # one-shot 런처 (smoke / phase0 / phase1)
├── Dockerfile                     # 단일 이미지 (trainer + 추론 서버 공통)
├── docker-compose.yml             # 학습 PC — trainer 1개 서비스
├── docker-compose.inference.yml   # 추론 PC — μ vLLM + PRM FastAPI 2개 서비스
├── .env.example / .env.inference.example
├── docs/
│   ├── IMPLEMENTATION_REPORT.md       # ← 본 문서
│   ├── TRAINING_FLOW.md
│   └── 구현 계획 — 차분 PAV + 분포형 보상 (Phase 0) ….md
├── configs/
│   ├── prm.yaml                   # Skywork PRM 1.5B int8 (mode: local|remote)
│   ├── policy.yaml                # Qwen2.5-Math-1.5B + Full FT (default) | LoRA/QLoRA, μ.mode 3가지
│   └── rl_q3.yaml                 # 메인 RL 조건 (Q3, λ=-0.5, K=16, optim=paged_adamw_8bit, lr=5e-7)
├── src/
│   ├── prm/
│   │   ├── loader.py              # PRMConfig + load_prm (local|remote 분기)
│   │   ├── score.py               # PRM(score, score_batch, score_per_step) + bnb 8bit
│   │   ├── remote_client.py       # ★ RemotePRM HTTP 클라이언트
│   │   ├── skywork_rm.py          # Skywork 1.5B (Qwen2ForPrmModel/RewardModel) wrapper
│   │   └── skywork/               # vendored: PRM_MODEL + ValueHead + io_utils
│   ├── pav/
│   │   ├── base.py                # PAVMethod Protocol
│   │   ├── differential.py        # Phase 0: A = PRM(s+a) - PRM(s)
│   │   ├── mc_rollout.py          # Phase 1: K개 μ-rollout으로 V 분포 추정
│   │   └── reduce.py              # B1 / Q1 / Q3 / Q4 reducer
│   ├── rollout/
│   │   ├── parser.py              # split_steps (헤더/blank-line/single-NL)
│   │   ├── mu_sampler.py          # μ — MuSampler(vllm) / SharedHFMuSampler / RemoteMuSampler
│   │   ├── remote_mu.py           # ★ RemoteMuSampler (vLLM OpenAI API 클라이언트)
│   │   └── vllm_rollout.py        # π trajectory rollout (BoN/eval용)
│   ├── train/
│   │   ├── reward_fn.py           # PAVRewardFn + stats/sample buffer
│   │   ├── policy_data.py         # build_policy / build_train_dataset
│   │   ├── grpo_trainer.py        # build_grpo_trainer (TRL GRPOTrainer + optim 키)
│   │   └── callbacks.py           # PAVMonitorCallback (W&B + dump)
│   └── eval/
│       ├── sanity.py              # S1~S4 자동 검증
│       └── bon_pav.py             # BoN-PAV vs BoN-PRM
├── scripts/
│   ├── download_models.py         # PRM + 정책 가중치 HF 다운로드
│   ├── 00_smoke_prm.py            # PRM smoke (toy 4문장)
│   ├── 01_phase0_diff.py          # Phase 0 + S1~S4
│   ├── 02_phase1_mc.py            # Phase 1 K 비교
│   ├── 03_grpo_train.py           # GRPO+LoRA/QLoRA/FullFT 학습 entry
│   ├── 10_label_steps.py          # MATH-500 → sanity 라벨 jsonl
│   ├── 20_eval_mathnet.py         # MathNet pass@1/N 평가
│   └── serve_prm_http.py          # ★ PRM HTTP 서버 (FastAPI + uvicorn)
└── tests/
    ├── test_pav_swap.py           # Protocol/reducer/swap 동작
    ├── test_parser.py             # split_steps 회귀
    └── test_imports.py            # 모든 모듈 import smoke
```

★ = 분산 셋업 (HTTP transport)을 위한 컴포넌트.

---

## 3. 핵심 설계 결정

### 3.1 `PAVMethod` Protocol (단일 인터페이스)

[src/pav/base.py](../src/pav/base.py) — 추출 방식별 차이를 dict 반환 형식으로 흡수.

```python
@runtime_checkable
class PAVMethod(Protocol):
    name: str
    def __call__(self, problem, prefix, step) -> dict:
        # 필수: advantage_scalar (0-d tensor)
        # 선택: advantage_samples ([K] tensor)  ← 없으면 reducer가 scalar fallback
        # 디버깅: p_q, p_v, p_v_samples
        ...
```

`reduce_advantage(out, mode="Q3", lam=-0.5)`가 분포/스칼라를 모두 받아 동일하게 처리 →
**Phase 0 ↔ Phase 1 swap은 `pav` 객체 교체 한 줄**.

### 3.2 천공 PRM 정합성

초기 구현은 표준 `AutoModelForCausalLM` + 마지막 logit 채널을 sigmoid한 가정이었으나,
실측 결과 Skywork PRM은 다음 구조였음:

| 항목 | 사양 |
|------|------|
| 클래스 | TRL `PreTrainedModelWrapper` 패턴의 커스텀 `PRM_MODEL` |
| Head  | `ValueHead = Linear(hidden, 1)` (per-token scalar) |
| Forward | `(lm_logits, loss, value)` 반환. `return_probs=True`이면 sigmoid 적용 |
| Step boundary | **단일 `\n`** |
| 입력 포맷 | `bos + problem + "\n" + (step + step_token) × N` |
| Reward 위치 | 각 step 마지막 토큰 위치 (reward_flag=1) |

→ Skywork 공식 inference 레포의 `prm_model.py` / `modeling_base.py` / `io_utils.py`를
[src/prm/skywork/](../src/prm/skywork/)에 **verbatim vendoring** (Apache 2.0).

[src/prm/score.py](../src/prm/score.py)는 정책이 출력하는 `\n\n` 구분 step을 내부에서
`\n`으로 자동 정규화(`_normalize_for_prm`)하여 사용자 코드는 boundary 차이를 신경쓰지 않아도 됨.

### 3.3 TRL GRPO 정합성

TRL ≥0.13의 실제 시그니처에 맞춤:

```python
GRPOConfig(
    use_vllm=True, vllm_mode="colocate",         # 0.13+ 표준
    num_generations=8, beta=0.04, epsilon=0.2,    # group, KL, clip
    max_completion_length=512,
    ...
)
GRPOTrainer(
    model=policy_model,
    processing_class=tokenizer,                   # 0.12+에서 tokenizer→processing_class
    args=grpo_cfg,
    train_dataset=train_dataset,                  # "prompt" + "answer" 컬럼
    reward_funcs=[trl_reward],                    # (prompts, completions, **cols)
    peft_config=peft_config,                      # LoRA 직접 주입
)
```

reward func 시그니처가 `(prompts, completions, **dataset_columns) -> list[float]`이라는 점이
초기 구현(positional `solution=...` 가정)과 다름 → [grpo_trainer.py](../src/train/grpo_trainer.py)에서
`kwargs["answer"]`로 정답을 받도록 정정.

### 3.4 보상 함수 — step-wise → trajectory scalar

GRPO는 trajectory당 scalar 1개를 받음. 따라서:

```
trajectory r = Σ_h (R_ex · 𝟙[h=H] + α · Ã_h)
```

step-wise credit assignment는 **GRPO의 group baseline (advantage normalization)**이
대신 처리하므로 별도 step-level discount는 추가하지 않음.

### 3.5 단일 호스트 / 분산 (2 PC) 양쪽 지원

`PRM` / `MuSampler` 클래스의 인터페이스(`score` / `score_batch` / `sample_step_batch`)를 **그대로**
보존하고, 내부 transport만 HTTP로 교체 가능. yaml의 `mode` 키 한 줄로 swap.

| 컴포넌트 | 단일 호스트 | 분산 (2 PC) |
|---|---|---|
| PRM | `PRM` ([src/prm/score.py](../src/prm/score.py)) — local GPU 적재 | `RemotePRM` ([src/prm/remote_client.py](../src/prm/remote_client.py)) — HTTP |
| PRM 서버 | — | [scripts/serve_prm_http.py](../scripts/serve_prm_http.py) (FastAPI + uvicorn, port 8002) |
| μ | `MuSampler` (vLLM) / `SharedHFMuSampler` (PEFT 재활용) | `RemoteMuSampler` ([src/rollout/remote_mu.py](../src/rollout/remote_mu.py)) — vLLM OpenAI API |
| μ 서버 | — | `python -m vllm.entrypoints.openai.api_server` (stock, port 8001) |
| 분기 | `loader.load_prm`, `build_mu_from_policy_yaml` | 동일 — yaml의 `mode: local|remote` 키로 자동 선택 |
| 환경변수 override | — | `PRM_ENDPOINT` / `MU_ENDPOINT` (배포 시 yaml 수정 X) |

특징:
- **표준 HTTP/JSON** — broker 없음. μ는 vLLM 기본 OpenAI 호환 API 그대로 사용 (서버 코드 0줄).
- **weight broadcast 없음** — π는 학습 PC 안에서 trainer + (선택적) vLLM colocate가 공유.
  μ/PRM은 frozen이므로 1회 로드 후 호출만.
- **PAVMethod / PAVRewardFn / GRPOTrainer 미수정** — 인스턴스 타입을 신경쓰지 않음.
- **VRAM 분리** — μ 7B(15 GB)를 학습 PC에서 빼면 24 GB GPU 1장으로 7B QLoRA 또는 1.5B Full FT 가능.

VRAM 빠듯한 단일 24GB 호스트에서는 다음 중 선택:
- **2 PC 분산** ⭐ — μ + PRM을 추론 PC로 분리 (이 셋업이 권장 default)
- 정책을 1.5B로 다운그레이드 (모두 적재 가능)
- Phase 0(`pav.method: differential`)만 학습 — μ 불필요
- `vllm.gpu_memory_utilization` 축소 또는 K 축소(예: K=4)

### 3.6 W&B + 함정 모니터링

[callbacks.PAVMonitorCallback](../src/train/callbacks.py)가 `PAVRewardFn.stats_buffer / sample_buffer`를
attach. 매 reward 계산마다 push되는 dict를 모아:

- `pav/A_mean`, `A_std`, `A_q05/q95`
- `pav/p_q_mean`, `pav/p_v_mean`
- `pav/corr_Q1_Q3` (분포가 의미있는지 — < 0.95 권장)
- 1k step마다 5 sample dump (trivial step 붕괴 검사)

`logs[]`에도 같이 넣어 progress bar에서도 확인 가능.

---

## 4. 모듈별 구현 내역

| 모듈 | 책임 | 핵심 객체 |
|------|------|------|
| [src/prm/loader.py](../src/prm/loader.py) | YAML → `PRMConfig` → `PRM` 또는 `RemotePRM` (mode 분기) | `load_prm`, `PRMConfig` |
| [src/prm/score.py](../src/prm/score.py) | Skywork PRM forward + step 보상 추출 + bnb 8bit | `PRM.score` / `score_batch` / `score_per_step` |
| [src/prm/remote_client.py](../src/prm/remote_client.py) | **★ HTTP 클라이언트** — PRM과 동일 인터페이스 | `RemotePRM`, `RemotePRMConfig` |
| [src/prm/skywork_rm.py](../src/prm/skywork_rm.py) | 1.5B(Qwen2ForPrmModel/RewardModel) wrapper | `SkyworkRMWrapper` |
| [src/prm/skywork/*](../src/prm/skywork/) | Skywork 공식 inference 코드 vendoring (7B 경로) | `PRM_MODEL`, `prepare_input`, `derive_step_rewards` |
| [src/pav/base.py](../src/pav/base.py) | PAV 추출 방식 통일 인터페이스 | `PAVMethod`, `is_distributional` |
| [src/pav/differential.py](../src/pav/differential.py) | Phase 0 차분 PAV | `DifferentialPAV` |
| [src/pav/mc_rollout.py](../src/pav/mc_rollout.py) | Phase 1 K개 μ-rollout 분포 | `MCRolloutPAV` |
| [src/pav/reduce.py](../src/pav/reduce.py) | B1/Q1/Q3/Q4 통합 reducer | `reduce_advantage` |
| [src/rollout/parser.py](../src/rollout/parser.py) | step 경계 분할 | `split_steps`, `normalize_step`, `join_steps` |
| [src/rollout/mu_sampler.py](../src/rollout/mu_sampler.py) | μ — vLLM / shared(disable_adapter) / remote 3가지 | `MuSampler`, `SharedHFMuSampler`, `build_mu_from_policy_yaml` |
| [src/rollout/remote_mu.py](../src/rollout/remote_mu.py) | **★ vLLM OpenAI API 클라이언트** | `RemoteMuSampler`, `RemoteMuConfig` |
| [src/rollout/vllm_rollout.py](../src/rollout/vllm_rollout.py) | π trajectory rollout (BoN) | `VLLMRollout`, `Trajectory` |
| [src/train/reward_fn.py](../src/train/reward_fn.py) | PAV → step reward + stats push | `PAVRewardFn`, `build_pav_from_config` |
| [src/train/policy_data.py](../src/train/policy_data.py) | Qwen2.5-Math + LoRA/QLoRA/FullFT, MATH/GSM8K | `build_policy`, `build_train_dataset` |
| [src/train/grpo_trainer.py](../src/train/grpo_trainer.py) | TRL GRPOTrainer 빌드 + `optim` 키 (8bit Adam 등) | `build_grpo_trainer`, `load_rl_config` |
| [src/train/callbacks.py](../src/train/callbacks.py) | W&B + 샘플 dump | `PAVMonitorCallback` |
| [src/eval/sanity.py](../src/eval/sanity.py) | S1~S4 자동 검증 | `run_sanity_checks`, `SanityItem/Result` |
| [src/eval/bon_pav.py](../src/eval/bon_pav.py) | BoN-PAV vs BoN-PRM | `bon_pav`, `bon_prm` |
| [scripts/download_models.py](../scripts/download_models.py) | HF 가중치 다운로드 | — |
| [scripts/00_smoke_prm.py](../scripts/00_smoke_prm.py) | PRM smoke (toy 4문장) | — |
| [scripts/10_label_steps.py](../scripts/10_label_steps.py) | MATH-500 자동 라벨링 | — |
| [scripts/01_phase0_diff.py](../scripts/01_phase0_diff.py) | Phase 0 + S1~S4 | — |
| [scripts/02_phase1_mc.py](../scripts/02_phase1_mc.py) | Phase 1 K 비교 | — |
| [scripts/03_grpo_train.py](../scripts/03_grpo_train.py) | GRPO 학습 entry | — |
| [scripts/20_eval_mathnet.py](../scripts/20_eval_mathnet.py) | MathNet pass@1/N 평가 | — |
| [scripts/serve_prm_http.py](../scripts/serve_prm_http.py) | **★ PRM HTTP 서버 (FastAPI)** | — |

---

## 5. 환경 (uv)

| 항목 | 결정 |
|------|------|
| 패키지 매니저 | uv 0.11.2 (`pyproject.toml` canonical) |
| Python | 3.11 (`.python-version`) |
| torch | 2.5.1+cu124 (cu126 드라이버 forward-compat — cu126 wheel은 torch 2.5에 없음) |
| transformers | 4.46.x (Skywork PRM_MODEL은 4.5x major rewrite와 충돌, `<4.50` 핀) |
| vLLM | 0.7.3 (torch 2.5.1 호환) |
| TRL | 0.15.2 (GRPO + vllm_mode=colocate + peft_config) |
| PEFT | 0.14+ |
| accelerate | 1.0+ (Skywork PRM_MODEL의 PartialState 의존) |

의존성 extras:

```bash
uv sync                       # base (+ httpx — RemotePRM/RemoteMuSampler 클라이언트)
uv sync --extra gpu           # +vllm/trl/peft/bitsandbytes/wandb/math-verify
uv sync --extra serve         # +fastapi/uvicorn[standard]  (추론 PC의 PRM HTTP 서버)
uv sync --extra awq           # +autoawq (옵션)
uv sync --group dev           # +pytest/ruff
```

Dockerfile은 기본으로 `--extra gpu --extra serve`를 sync하여 trainer/추론서버 둘 다 쓸 수 있는 단일 이미지 생성.

---

## 6. 검증 결과

### 6.1 자동 테스트 — 19/19 통과

```
tests/test_imports.py     7 tests   (pav, prm, skywork, train, rollout, eval, grpo_trainer)
tests/test_parser.py      6 tests   (헤더/blank-line/single-NL/empty/normalize/join)
tests/test_pav_swap.py    6 tests   (Protocol, output shape × 2, reducer 호환,
                                     reward_fn swap, Q3 std 보너스)
─────────────────────────────────────────────
                          19 passed in 56.78s
```

### 6.2 수동 검증

- `uv run python scripts/download_models.py --list-only` → 대상 모델 정상 출력.
- `uv sync --extra gpu --extra awq --dry-run` → trl 0.15.2, vllm 0.7.3 등 lock 해상도 OK.
- `torch 2.5.1+cu124`, `cuda 12.4` (cu126 드라이버에서 forward-compat 동작 예상).

### 6.3 미검증 (외부 자원 필요)

- 실제 PRM forward → step 보상 분리 (`scripts/00_smoke_prm.py`)
- vLLM μ rollout (`scripts/02_phase1_mc.py`)
- GRPO 학습 1k smoke step (`scripts/03_grpo_train.py`)

가중치 다운로드(`scripts/download_models.py`) 후 GPU 환경에서 위 3개 스크립트가 첫 실행 검증 항목.

---

## 7. 검증 게이트 매핑

| Gate | 자동화 위치 | 통과 기준 |
|------|------|------|
| **G0** Phase 0 | `scripts/01_phase0_diff.py` + `scripts/10_label_steps.py` (GSM8K test) | S1~S4 모두 + BoN-PAV ≥ BoN-PRM |
| **G1** 분포 | `scripts/02_phase1_mc.py` + `src/eval/bon_pav.py` | BoN-PAV(분포) ≥ BoN-PAV(스칼라), corr(Q1,Q3) < 0.95 |
| **G2** RL 효과 | `scripts/20_eval_mathnet.py` (MathNet) + W&B 모니터링 | pass@N +3%p **또는** entropy decay 50% 완화 |
| **G3** ablation | `configs/rl_q3.yaml`의 `reward.mode` 변경 (Q1 vs Q3) + `20_eval_mathnet.py` 비교 | A.mean only vs A.mean+std 유의차 |

---

## 8. 알려진 한계 / 남은 작업

### 8.0 컨테이너 (Docker) — 단일 PC / 2 PC 양쪽 지원

**공통**
- [Dockerfile](../Dockerfile) — `pytorch:2.5.1-cuda12.4-cudnn9-runtime` + uv + `--extra gpu --extra serve`.
  trainer / μ vLLM / PRM HTTP 서버 모두 같은 이미지 사용.

**단일 PC** ([docker-compose.yml](../docker-compose.yml))
- `trainer` 서비스 1개 (GPU, shm_size=8g)
- yaml에서 `mode: local` 또는 `mode: remote + endpoint: http://localhost:...` 둘 다 가능
- [.env.example](../.env.example) — `HF_HOME_HOST` / `HF_TOKEN` / `WANDB_API_KEY` / `PRM_ENDPOINT` / `MU_ENDPOINT`

**2 PC 분산** — 추론 PC에 [docker-compose.inference.yml](../docker-compose.inference.yml) 추가
- `mu-server` — `python -m vllm.entrypoints.openai.api_server` (Qwen2.5-Math-7B base, port 8001)
- `prm-server` — `python scripts/serve_prm_http.py` (Skywork PRM 1.5B int8, port 8002)
- [.env.inference.example](../.env.inference.example) — `HF_HOME_HOST` / `HF_TOKEN` / `MU_MODEL_ID` / `MU_GPU_MEM` / `MU_MAX_LEN`

학습 PC의 `.env`에 `PRM_ENDPOINT=http://<inference-IP>:8002`, `MU_ENDPOINT=http://<inference-IP>:8001`만
넣으면 [src/prm/loader.py](../src/prm/loader.py) / [src/rollout/mu_sampler.py](../src/rollout/mu_sampler.py)가
yaml의 `remote.endpoint`를 자동 override.

호스트 NVIDIA Container Toolkit이 깔린 상태에서 `docker compose up -d`로 기동.
HF 캐시는 호스트 bind mount로 영속화.

### 8.1 실모델 검증 부재
- 본 보고서 작성 시점까지 천공 PRM·정책 가중치를 다운로드하지 않아 실제 forward 검증은 불완전.
- 첫 GPU 실행에서 다음 항목 점검 필요:
  - PRM `score_batch`의 device placement (accelerate `device_map="auto"`가 multi-GPU에서 ValueHead와 unify되는지)
  - vLLM colocate가 RTX 4090 24GB × 2 또는 H100 80GB × 2에서 OOM 없이 동작하는지

### 8.2 라벨 데이터 품질
- `scripts/10_label_steps.py`는 휴리스틱(filler 패턴 + final-step 정답 매칭) 자동 라벨링.
- G0 게이트의 신뢰성을 높이려면 LLM-judge(예: GPT-4o-mini) 라벨을 추가 권장.

### 8.3 PRM 사이즈 / 양자화
- 공식 PRM 사이즈는 **1.5B와 7B 두 가지뿐** (32B/72B 없음).
- 현재 default는 **1.5B + 8bit** (~1.7 GB) — bitsandbytes LLM.int8.
  - `configs/prm.yaml`의 `quantization`: `"none"` (fp16, ~2.9 GB) | `"8bit"` (bnb int8, ~1.7 GB) | `"awq"` (4-bit AWQ).
  - `model_id`를 `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`로 바꾸면 7B 변형 사용 (fp16 ~14 GB / int8 ~8 GB).
- AWQ 4-bit 공식 변환본 미제공 — autoawq 별도 설치 + 사용자가 직접 변환 시만 활용.

### 8.4 데이터셋 (확정)

| 용도 | HF dataset | 처리 |
|------|------|------|
| **학습** | `openai/gsm8k` (config: `main`, split: `train`) | `#### N` 정답 추출, `{prompt, answer}` |
| **Sanity (G0)** | `openai/gsm8k` test (default) — `--dataset {gsm8k\|math500\|mathnet}`로 변경 가능 | filler/correct/wrong 자동 라벨 |
| **검증/평가** | `ShadenA/MathNet` train split | `language=English` + `len(images)==0` + `final_answer not null`, subset 200 |

근거:
- GSM8K는 정책 base의 정답률이 충분히 높아 sanity의 `correct/wrong` sample이 균형있게 나옴.
- MathNet은 Olympiad-level (SOTA 78.4%), train split만 존재 → 평가에만 사용. 멀티모달이라 텍스트 전용 필터 필수.
- MATH/MATH500은 legacy 호환 옵션으로만 유지 (`build_train_dataset`가 `hendrycks_math` 키도 처리).

### 8.5 KL 모니터링
- PAVMonitorCallback에 `KL(π‖μ_base)` 추적은 미구현 (TRL이 GRPO loss 안에서 KL은 계산하나 외부 push는 안 함).
- 필요 시 callbacks.py에 evaluation hook 추가.

---

## 9. 다음 단계 (권장 순서)

### 단일 PC

1. `scripts/download_models.py` 실행 → 가중치 캐시.
2. `scripts/00_smoke_prm.py` → PRM이 toy 입력에서 정답/오답을 분리하는지 1차 확인.
3. `scripts/10_label_steps.py --dataset gsm8k --n-problems 200` → sanity 라벨 jsonl.
4. `scripts/01_phase0_diff.py --items-jsonl data/sanity_items.jsonl` → S1~S4 측정 → **G0 판정**.
5. `scripts/02_phase1_mc.py --ks 4 8 16 32` → K 결정 + 분포 신호 확인 → **G1 판정**.
6. `scripts/03_grpo_train.py --rl-config configs/rl_q3.yaml` → GSM8K로 GRPO 학습.
7. `scripts/20_eval_mathnet.py --lora ./outputs/.../checkpoint-N --N 64` → MathNet pass@1 / pass@N → **G2 / G3**.

### 2 PC 분산 (3090 학습 + 3090 Ti 추론)

상세 quickstart + 트러블슈팅 → **[QUICKSTART.md](QUICKSTART.md)**.

요약:
1. **추론 PC**:
   ```bash
   cp .env.inference.example .env
   docker compose -f docker-compose.inference.yml up -d
   curl http://localhost:8002/health        # PRM 확인
   curl http://localhost:8001/v1/models     # μ vLLM 확인
   ```
2. **학습 PC**:
   ```bash
   cp .env.example .env
   # PRM_ENDPOINT / MU_ENDPOINT를 추론 PC IP로 수정
   docker compose up -d
   docker compose logs -f
   ```
   또는 docker 없이:
   ```bash
   PRM_ENDPOINT=http://<inference-ip>:8002 MU_ENDPOINT=http://<inference-ip>:8001 \
     bash run_train.sh --mode phase1
   ```
3. **검증**: 학습 PC에서 `scripts/01_phase0_diff.py`, `scripts/03_grpo_train.py` 등은 yaml의
   `mode: remote` 분기로 자동 RemotePRM / RemoteMuSampler 사용.

---

## 10. 참고

- 원본 계획서: [구현 계획 — 차분 PAV + 분포형 보상 (Phase 0) 4dc18b40f5ed47259166ff0f0c0f8086.md](구현%20계획%20—%20차분%20PAV%20+%20분포형%20보상%20(Phase%200)%204dc18b40f5ed47259166ff0f0c0f8086.md)
- Skywork PRM 추론 레포: <https://github.com/SkyworkAI/skywork-o1-prm-inference>
- TRL GRPOTrainer 도큐: <https://huggingface.co/docs/trl/en/grpo_trainer>
- DeepSeekMath GRPO 논문: <https://arxiv.org/abs/2402.03300>
