#!/bin/bash
# MOON 생성/갱신 — IP 는 환경변수 MOON_IP 로 받음 (파일/git 에 IP 안 남김)
#
# 사용:
#   MOON_IP=<학습PC 공인IP> ./PAV/scripts/setup-moon.sh
#
# 결과:
#   - pav-ztncui:/var/lib/zerotier-one/moon.json (서명 secret 포함, gitignored)
#   - pav-ztncui:/var/lib/zerotier-one/moons.d/000000<moonid>.moon
#   - PAV/zerotier/moons.d/000000<moonid>.moon (학습 PC member 용)
#   - 추론 PC orbit 명령어 출력
set -euo pipefail

if [[ -z "${MOON_IP:-}" ]]; then
  echo "ERROR: MOON_IP env var not set. Usage: MOON_IP=<public_ip> $0" >&2
  exit 1
fi

# 0. IP 형식 간단 검증 (echo 안 함 — 로그에 IP 남기지 않음)
if ! [[ "${MOON_IP}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: MOON_IP must be IPv4 (got: <redacted>)" >&2
  exit 1
fi

# 1. ztncui 컨테이너 안에서 moon.json 생성/갱신 + genmoon
docker exec -e MOON_IP="${MOON_IP}" pav-ztncui sh -c '
  set -e
  cd /var/lib/zerotier-one
  if [ ! -f moon.json ]; then
    zerotier-idtool initmoon identity.public > moon.json
  fi
  python3 <<PYEOF
import json, os
p = "moon.json"
with open(p) as f: d = json.load(f)
d["roots"][0]["stableEndpoints"] = [os.environ["MOON_IP"] + "/9993"]
with open(p, "w") as f: json.dump(d, f, indent=1)
PYEOF
  rm -f 000000*.moon moons.d/*.moon
  zerotier-idtool genmoon moon.json
  mkdir -p moons.d
  mv 000000*.moon moons.d/
' >/dev/null

MOONFILE=$(ls PAV/ztncui/ztone/moons.d/000000*.moon | head -1)
MOON_ID=$(basename "${MOONFILE}" .moon | sed 's/^000000//')

# 2. 학습 PC member (pav-zerotier) 로 복사
mkdir -p PAV/zerotier/moons.d
cp "${MOONFILE}" PAV/zerotier/moons.d/

# 3. member 재시작 (pav-zerotier 네임스페이스 공유: trainer, nginx-lb 도 함께 재시작 필요)
echo "[MOON] file generated (moon_id=${MOON_ID})"
echo "[MOON] To activate locally, restart containers sharing pav-zerotier network ns:"
echo "       docker stop pav-trainer pav-nginx-lb pav-zerotier"
echo "       docker start pav-zerotier && sleep 5 && docker start pav-nginx-lb pav-trainer"
echo
echo "[MOON] On INFERENCE PC, run (no file transfer needed):"
echo "       docker exec pav-zerotier zerotier-cli orbit ${MOON_ID} ${MOON_ID}"
echo
echo "[MOON] Verify after activation:"
echo "       docker exec pav-zerotier zerotier-cli listmoons"
echo "       docker exec pav-zerotier zerotier-cli peers   # MOON 행이 DIRECT 로 승급되면 성공"
