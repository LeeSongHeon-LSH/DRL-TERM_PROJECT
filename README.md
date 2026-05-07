# DRL Term Project — LLM RLHF Pipeline

## Overview

LLM을 강화학습(RL)으로 fine-tuning하는 파이프라인입니다.  
Policy LLM이 생성한 출력을 Judge LLM이 평가하여 reward signal을 제공하고, 이를 바탕으로 PPO로 학습합니다.

> **TBD**: Policy LLM, Judge LLM, Benchmark는 추후 확정 예정입니다.  
> 확정되는 대로 이 README를 업데이트합니다.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Pipeline                             │
│                                                             │
│  Dataset ──► Policy LLM ──► Response                        │
│               (학습 대상)         │                          │
│                                  ▼                          │
│                          Judge LLM (Reward Model)           │
│                          확률 분포 기반 reward 계산           │
│                                  │                          │
│                                  ▼                          │
│                          PPO Trainer                        │
│                          (정책 업데이트)                      │
│                                  │                          │
│                                  ▼                          │
│                          Benchmark Evaluator                │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

| 모듈 | 역할 | 상태 |
|------|------|------|
| `models/policy.py` | 학습 대상 LLM 래퍼 | TBD (모델 미확정) |
| `models/reward_model.py` | Judge LLM 래퍼 (reward 계산) | TBD (모델 미확정) |
| `reward/llm_judge.py` | Judge LLM의 확률 분포를 reward로 변환 | TBD |
| `trainer/ppo_trainer.py` | PPO 학습 루프 | TBD (알고리즘 검토 중) |
| `eval/evaluator.py` | 벤치마크 평가 러너 | TBD (벤치마크 미확정) |
| `data/dataset.py` | 데이터셋 로딩 및 전처리 | TBD |

---

## Training Methodology

### Reward Signal
Judge LLM의 출력 **확률 분포**를 reward signal로 사용합니다.  
(구체적인 reward 계산 방식은 확정 후 업데이트 예정)

### Algorithm
- **후보**: PPO (Proximal Policy Optimization)  
- **미확정 사항**: 알고리즘 최종 결정 후 업데이트 예정

---

## Models (TBD)

- **Policy LLM**: 미확정
- **Judge LLM**: 미확정

---

## Benchmarks (TBD)

- 미확정

---

## Environment

- Single GPU 환경 기준 설계
- Python 3.10+
- PyTorch

---

## Project Structure

```
DRL-TERM_PROJECT/
├── config/
│   └── base_config.yaml       # 전체 실험 설정
├── data/
│   └── dataset.py             # 데이터셋 인터페이스
├── models/
│   ├── base.py                # 추상 LLM 인터페이스
│   ├── policy.py              # Policy LLM 래퍼
│   └── reward_model.py        # Judge LLM 래퍼
├── reward/
│   ├── base.py                # 추상 Reward 인터페이스
│   └── llm_judge.py           # LLM-as-Judge reward 계산
├── trainer/
│   ├── base.py                # 추상 Trainer 인터페이스
│   └── ppo_trainer.py         # PPO Trainer
├── eval/
│   ├── base.py                # 추상 Benchmark 인터페이스
│   └── evaluator.py           # 평가 러너
├── utils/
│   └── logging.py             # 로깅 유틸리티
├── scripts/
│   ├── train.py               # 학습 진입점
│   └── evaluate.py            # 평가 진입점
└── requirements.txt
```
