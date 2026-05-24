#!/bin/sh
# ZeroTier entrypoint with optional MOON orbit
#
# Env vars:
#   $1 (positional)  = ZT_NETWORK_ID (16 hex chars) — network to join
#   MOON_IP          = MOON 공인 IP (set 되면 orbit 실행, value 자체는 로깅용)
#
# MOON_ID 는 ZT_NETWORK_ID 의 앞 10 hex 로 자동 derive (= controller address)
set -e

mkdir -p /var/lib/zerotier-one
zerotier-one -d

# Wait for daemon socket
TIMEOUT=20
until [ "$TIMEOUT" -le 0 ] || zerotier-cli info >/dev/null 2>&1; do
  sleep 1
  TIMEOUT=$((TIMEOUT - 1))
done

NET_ID="${1:-}"
if [ -n "${NET_ID}" ]; then
  zerotier-cli join "${NET_ID}" || true
fi

if [ -n "${MOON_IP:-}" ] && [ -n "${NET_ID}" ]; then
  # MOON_IP value 자체는 사용 안 함 (orbit 은 MOON_ID + seed peer 만 필요).
  # 단지 "MOON 활성화 플래그" 역할 — 어떤 non-empty 값이든 OK.
  # 실제 MOON endpoint 는 PLANET 통해 fetch 되는 .moon 정의에 들어있음.
  MOON_ID=$(echo "${NET_ID}" | cut -c1-10)
  echo "[zt] MOON orbit requested (moon_id=${MOON_ID})"
  zerotier-cli orbit "${MOON_ID}" "${MOON_ID}" || echo "[zt] orbit failed (may already be orbited)"
fi

exec tail -f /dev/null
