# FRP — 추론 서버 로드밸런서 (frps + frpc)

본 프로젝트의 다이어그램·문서에서 말하는 **"로드밸런서(Load Balancer)"가 바로 이 FRP 셋업**이다.
학습 PC(보유 RTX 3090, 공인 IP)와 클라우드 **T4 추론 인스턴스들** 사이를 **단일 TCP 터널**로 잇고,
같은 역할의 서버가 여러 대일 때 **round-robin + health-check로 부하를 분산**한다.

```
[학습 PC · 공인 IP]                         [추론 cloud T4 ×N · NAT 뒤]
 trainer ─ http://frps:18001 (μ) ┐          frpc ── outbound TCP 1개 ──┐
          http://frps:18002 (PRM)│                                     │
                                 ▼                                     ▼
        ┌──────── frps (로드밸런서) ────────┐        mu-server  :8001  (mu_cluster)
        │ control :7000 · dashboard :7500   │◄──────  prm-server :8002  (prm_cluster)
        │ round-robin · 10s health-check    │
        └───────────────────────────────────┘
```

## 왜 FRP가 "로드밸런서"인가
- **N개 replica 분산** — 여러 추론 T4의 frpc가 같은 `loadBalancer.group`(`mu_cluster` / `prm_cluster`)으로 등록되면,
  학습 PC의 `localhost:18001`(μ) / `localhost:18002`(PRM) 호출을 frps가 **살아있는 replica 하나로 round-robin** 라우팅.
- **health-check** — 매 10초 HTTP probe(μ `/v1/models`, PRM `/health`), **3회 실패 시 자동으로 LB pool에서 제거**, 복구되면 재등록.
- **무중단 스케일** — 추론 인스턴스를 `docker compose up/down` 하면 frps가 즉시 반영, 학습 PC 쪽 수정 0.
- **NAT 무관** — frpc는 **outbound TCP만** 사용 → 클라우드 T4가 NAT/방화벽 뒤에 있어도 OK (공인 IP는 학습 PC 1개만 필요).
- **weight broadcast 0** — μ·PRM은 frozen, 요청/응답만 오감 (step당 RPC ~6MB).

## 구성 파일

| 파일 | 위치 | 역할 |
|---|---|---|
| [`frps.toml`](frps.toml) | 학습 PC | control :7000 수신 · dashboard :7500 · remote port 18001~18099 허용 · token 인증 |
| [`frpc.toml`](frpc.toml) | 각 추론 T4 | outbound 터널 1개 · `mu-server:8001→18001`, `prm-server:8002→18002` · group + health-check |

> 두 파일은 docker-compose가 환경변수를 주입하는 **템플릿**(`{{ .Envs.XXX }}`)이다. 직접 값을 적지 말 것.

## 환경변수

| 변수 | 어디서 | 설명 |
|---|---|---|
| `FRPS_TOKEN` | frps · frpc 공통 | 인증 토큰(random 32+ chars). **불일치 frpc는 거부**. git 저장 X — 명령어 inline env |
| `FRPS_ADDR` | frpc | 학습 PC 공인 IP 또는 DDNS (예: `myhost.duckdns.org`) |
| `NODE_NAME` | frpc | 추론 인스턴스 고유 라벨 (예: `t4-mu-01`) — proxy name 충돌 방지 |
| `FRPS_DASHBOARD_PW` | frps | dashboard(`http://학습PC:7500`, admin) 비번 |

## 실행

```bash
# 학습 PC — frps(로드밸런서) 기동
FRPS_TOKEN=$(openssl rand -hex 32) FRPS_DASHBOARD_PW=<비번> \
  docker compose up -d frps                       # ../docker-compose.yml

# 각 추론 cloud T4 — frpc + μ/PRM 서버 (NODE_NAME만 다르게)
FRPS_ADDR=myhost.duckdns.org FRPS_TOKEN=<위와 동일> NODE_NAME=t4-01 \
  docker compose -f docker-compose.inference.yml up -d
```

- **dashboard**: `http://<학습PC>:7500` (admin / `FRPS_DASHBOARD_PW`) — 모든 frpc 상태·트래픽·health 실시간
- **학습 PC에서 LB 통한 호출 확인**: `curl localhost:18001/v1/models` (μ) · `curl localhost:18002/health` (PRM)

## 보안
- `FRPS_TOKEN`은 `.env` 저장 X → 명령어 inline env로만 전달. 토큰 불일치 frpc는 인증 거부.
- frps는 `allowPorts`로 **18001~18099만** 노출. dashboard는 내부망/VPN 노출 권장.

> 참고: 이전엔 ZeroTier mesh + nginx-lb를 시도했으나, 두 PC 모두 NAT 뒤일 때 RELAY 패킷 손실로 long-lived TCP가 자주 끊겼다. **학습 PC가 공인 IP를 가지면 FRP가 더 단순·안정**이라 교체. 상세: [../docs/QUICKSTART.md](../docs/QUICKSTART.md) · [../docs/TRAINING_FLOW.md](../docs/TRAINING_FLOW.md).
