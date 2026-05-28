# FRP — TCP Tunnel Hub

PAV, PRM, step-wise-PRM 이 공유하는 **FRP 서버**를 관리하는 디렉토리입니다.

## 구조

```
FRP/
├── docker-compose.yml    # FRP 서버 (frps) — 학습 PC에서 실행
└── frps.toml             # FRP 서버 설정
```

## 사용법

### 1. FRP 서버 시작 (학습 PC)

```bash
cd FRP
FRPS_TOKEN=your-random-32-char-token FRPS_DASHBOARD_PW=your-password \
  docker compose up -d
```

- dashboard: http://localhost:7500
- `pav-shared-network` 브리지 네트워크가 자동 생성됨

### 2. 추론 PC 등록

각 프로젝트 디렉토리에서 클라이언트를 실행합니다:

**PAV (μ + PRM):**
```bash
cd PAV
FRPS_ADDR=<학습PC-공인IP> FRPS_TOKEN=<동일토큰> NODE_NAME=pc-01 \
  docker compose -f docker-compose.inference.yml up -d
```

**PRM (PRM만):**
```bash
cd PRM
FRPS_ADDR=<학습PC-공인IP> FRPS_TOKEN=<동일토큰> NODE_NAME=pc-01 \
  docker compose -f docker-compose.inference.yml up -d
```

**step-wise-PRM (PRM만):**
```bash
cd step-wise-PRM
FRPS_ADDR=<학습PC-공인IP> FRPS_TOKEN=<동일토큰> NODE_NAME=pc-01 \
  docker compose -f docker-compose.inference.yml up -d
```

### 3. 학습 시작

PAV, PRM, step-wise-PRM은 각자의 디렉토리에서 학습을 시작합니다.
`pav-shared-network`를 통해 FRP 서버에 자동 연결됩니다.

## 포트

| 포트 | 용도 |
|------|------|
| 7000 | FRP control (frpc → frps) |
| 7500 | FRP dashboard |
| 18001 | μ vLLM cluster |
| 18002 | PRM 서버 cluster |

## 네트워크

`pav-shared-network`는 `FRP/docker-compose.yml`에서 생성되며,
PAV/PRM/step-wise-PRM의 `docker-compose.yml`에서 `external: true`로 연결합니다.
