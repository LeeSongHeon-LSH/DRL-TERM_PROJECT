# DRL Term Project — GRPO on GSM8K with Qwen2.5-7B

## Overview

Qwen2.5-7B-Instruct를 **GRPO**(Group Relative Policy Optimization)로 GSM8K 수학 데이터셋에 파인튜닝합니다.  
추론(rollout)은 vLLM으로 가속하고, LoRA로 파라미터 효율적 학습을 수행합니다.  
벤치마크는 MATH(Hendrycks et al., 2021) — "MathNet" — 으로 평가합니다.

---

## Architecture

```
GSM8K Dataset
     │
     ▼
vLLM (Qwen2.5-7B)  ──►  G responses per prompt
     │
     ▼
Rule-based Reward  ──►  answer correct? → 1.0 / 0.0
     │
     ▼
GRPO Advantage  ──►  within-group normalization: (R_i - mean) / std
     │
     ▼
HF Model + LoRA  ──►  clipped policy gradient + KL penalty
     │
     ▼
Weight Sync  ──►  merged LoRA weights pushed back to vLLM
```

### Why GRPO over PPO

| | PPO | GRPO |
|---|---|---|
| Critic model | 필요 (추가 메모리) | 불필요 |
| Advantage | GAE (value network) | 그룹 내 reward 정규화 |
| 수학 태스크 적합성 | 보통 | 높음 (DeepSeek-R1 방식) |

---

## Project Structure

```
DRL-TERM_PROJECT/
├── config/
│   └── config.yaml          # 모든 하이퍼파라미터
├── data/
│   └── gsm8k.py             # GSM8K 데이터셋 로더
├── reward/
│   └── math_reward.py       # 규칙 기반 reward (정답 일치 여부)
├── trainer/
│   └── grpo_trainer.py      # GRPO 학습 루프 (vLLM + HF + LoRA)
├── eval/
│   └── math_eval.py         # MATH 벤치마크 평가 (MathNet)
├── scripts/
│   ├── train.py             # 학습 진입점
│   └── evaluate.py          # 평가 진입점
└── requirements.txt
```

---

## Environment

### 권장 사양
- GPU: A100 80GB (vLLM + HF policy + HF ref 동시 로드 시 ~44GB 필요)
- Python 3.10+
- CUDA 12.1+

### 설치
```bash
pip install -r requirements.txt
```

---

## Usage

### 학습
```bash
python scripts/train.py
```

### 평가 (MATH 벤치마크)
```bash
python scripts/evaluate.py outputs/final
# 또는 특정 체크포인트
python scripts/evaluate.py outputs/checkpoint-200
```

---

## Key Config (`config/config.yaml`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `grpo.num_generations` | 8 | 프롬프트당 생성 응답 수 (G) |
| `grpo.batch_size` | 4 | 스텝당 프롬프트 수 |
| `grpo.kl_coef` | 0.01 | KL 페널티 계수 |
| `grpo.clip_epsilon` | 0.2 | PPO-style clipping 범위 |
| `grpo.weight_sync_steps` | 20 | vLLM 가중치 동기화 주기 |
| `vllm.gpu_memory_utilization` | 0.20 | vLLM GPU 메모리 할당 비율 |

---

## Notes

- **MathNet**: Hendrycks et al.의 [MATH 벤치마크](https://arxiv.org/abs/2103.03874)(`hendrycks/competition_math`)로 해석합니다.  
  다른 데이터셋을 의도하신 경우 `eval/math_eval.py`의 `_MATH_DATASET`을 수정하세요.
- **vLLM 버전**: weight sync는 vLLM 내부 API(`driver_worker.model_runner.model`)를 사용합니다. vLLM ≥ 0.6.0 권장.
- **메모리 절약**: KL 페널티가 불필요하면 `kl_coef: 0.0`으로 설정하면 ref_model 로드를 생략하도록 수정 가능합니다.
