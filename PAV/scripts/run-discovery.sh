#!/bin/bash
# ZeroTier cluster 자동 discovery + nginx config 생성 + nginx 재시작.
#
# 사용: bash scripts/run-discovery.sh
# 결과: nginx/inference-cluster.conf 자동 생성, nginx-lb 재시작
#
# 추론 PC 추가/제거 시 학습 PC에서 이 스크립트 한 번 실행하면 자동 반영.

set -e
cd "$(dirname "$0")/.."

# 1. zerotier 컨테이너에서 학습 PC의 ZeroTier IP 추출
SELF_IP=$(docker exec pav-zerotier zerotier-cli listnetworks 2>/dev/null \
    | awk 'END {print $NF}' | cut -d/ -f1)
if [ -z "$SELF_IP" ] || [ "$SELF_IP" = "-" ]; then
    echo "❌ ZeroTier IP 없음. ZTNCUI에서 노드 Authorize + IP 부여 확인."
    exit 1
fi
echo "🔎 ZeroTier self IP: $SELF_IP"

# 2. discovery 컨테이너 (python:slim ad-hoc) 실행 → nginx config stdout
docker run --rm \
    --network container:pav-zerotier \
    -e SELF_IP="$SELF_IP" \
    -v "$(pwd)/scripts:/s:ro" \
    python:3.11-slim python3 /s/cluster_discovery.py > nginx/inference-cluster.conf

echo "✅ nginx config 생성: nginx/inference-cluster.conf"
cat nginx/inference-cluster.conf | grep -E '^\s+server|upstream' | head -20

# 3. nginx-lb 재시작 (구동 중이면)
if docker ps --filter name=pav-nginx-lb --format '{{.Names}}' | grep -q pav-nginx-lb; then
    docker compose restart nginx-lb
    echo "✅ nginx-lb 재시작 완료"
else
    echo "ℹ nginx-lb 미구동 — 학습 시작 시 자동 적용"
fi
