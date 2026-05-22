#!/usr/bin/env bash
# 분산 추론 서버 (μ vLLM + PRM FastAPI) 더미 데이터 smoke test.
#
# 추론 PC 자체에서:
#   bash scripts/smoke_inference.sh
#
# 학습 PC에서 추론 PC 호출 (env 또는 인자로 override):
#   PRM_ENDPOINT=http://192.168.1.10:8002 \
#   MU_ENDPOINT=http://192.168.1.10:8001 \
#     bash scripts/smoke_inference.sh
#
# 또는:
#   bash scripts/smoke_inference.sh http://192.168.1.10:8002 http://192.168.1.10:8001

set -uo pipefail

# ---- endpoint 결정 ----
PRM_ENDPOINT="${1:-${PRM_ENDPOINT:-http://localhost:8002}}"
MU_ENDPOINT="${2:-${MU_ENDPOINT:-http://localhost:8001}}"

# 컬러 출력
if [[ -t 1 ]]; then
  G='\033[0;32m'; R='\033[0;31m'; Y='\033[0;33m'; B='\033[0;34m'; N='\033[0m'
else
  G=''; R=''; Y=''; B=''; N=''
fi

pass=0; fail=0
banner() { echo -e "\n${B}━━ $* ━━${N}"; }
ok()     { echo -e "  ${G}✓${N} $*"; pass=$((pass+1)); }
ko()     { echo -e "  ${R}✗${N} $*"; fail=$((fail+1)); }
info()   { echo -e "  ${Y}·${N} $*"; }

# httpx 또는 curl 둘 다 처리 — curl만 있으면 충분
req() {
  # req METHOD URL [JSON_PAYLOAD]  → stdout: 응답 body, exit code: HTTP status로 매핑 (0=2xx)
  local method="$1" url="$2" payload="${3:-}"
  local tmp; tmp=$(mktemp)
  local code
  if [[ -n "$payload" ]]; then
    code=$(curl -sS -m 30 -o "$tmp" -w "%{http_code}" \
      -X "$method" -H "Content-Type: application/json" -d "$payload" "$url") || code="000"
  else
    code=$(curl -sS -m 30 -o "$tmp" -w "%{http_code}" -X "$method" "$url") || code="000"
  fi
  cat "$tmp"
  rm -f "$tmp"
  [[ "$code" =~ ^2 ]]
}

# 간단 timing — `date +%s%N`로 ms 측정
ms_since() {
  local t0=$1
  echo $(( ($(date +%s%N) - t0) / 1000000 ))
}

# ============================================================================
echo -e "${B}분산 추론 smoke test${N}"
echo "  PRM_ENDPOINT = $PRM_ENDPOINT"
echo "  MU_ENDPOINT  = $MU_ENDPOINT"

# ----------------------------------------------------------------------------
banner "[1/6] PRM /health"
t0=$(date +%s%N)
if resp=$(req GET "$PRM_ENDPOINT/health"); then
  ok "응답 ($(ms_since $t0)ms): $resp"
else
  ko "health 실패: $resp"
  echo -e "\n${R}PRM 서버 응답 없음. 추론 PC에서 'docker compose -f docker-compose.inference.yml logs prm-server' 확인.${N}"
  exit 1
fi

# ----------------------------------------------------------------------------
banner "[2/6] μ vLLM /v1/models"
t0=$(date +%s%N)
if resp=$(req GET "$MU_ENDPOINT/v1/models"); then
  ok "응답 ($(ms_since $t0)ms)"
  echo "$resp" | head -c 300; echo "…"
else
  ko "models 실패: $resp"
  echo -e "\n${R}μ vLLM 서버 응답 없음. 'docker compose -f docker-compose.inference.yml logs mu-server' 확인.${N}"
  exit 1
fi

# ----------------------------------------------------------------------------
banner "[3/6] PRM /v1/score — 단일 (정답/오답 비교)"
TOY_PROBLEM="What is 7 + 5?"
PREFIX_CORRECT="Step 1: 7 + 5 = 12.\n"
PREFIX_WRONG="Step 1: 7 + 5 = 13.\n"

for label in correct wrong; do
  case $label in
    correct) prefix="$PREFIX_CORRECT" ;;
    wrong)   prefix="$PREFIX_WRONG" ;;
  esac
  payload=$(printf '{"problem": %s, "solution_prefix": %s}' \
    "\"$TOY_PROBLEM\"" "\"$prefix\"")
  t0=$(date +%s%N)
  if resp=$(req POST "$PRM_ENDPOINT/v1/score" "$payload"); then
    score=$(echo "$resp" | grep -oE '"score":[ ]*[0-9.]+' | grep -oE '[0-9.]+')
    ok "$label score=${score} ($(ms_since $t0)ms)"
  else
    ko "score 실패 ($label): $resp"
  fi
done

# ----------------------------------------------------------------------------
banner "[4/6] PRM /v1/score_batch — 4개 묶음"
t0=$(date +%s%N)
payload=$(cat <<'EOF'
{
  "problem": "What is 7 + 5?",
  "solution_prefixes": [
    "Step 1: 7 + 5 = 12.\n",
    "Step 1: Let me add these numbers.\n",
    "Step 1: 7 + 5 = 13.\n",
    "Step 1: Let me think carefully about this.\n"
  ]
}
EOF
)
if resp=$(req POST "$PRM_ENDPOINT/v1/score_batch" "$payload"); then
  ok "응답 ($(ms_since $t0)ms): $(echo "$resp" | head -c 200)"
else
  ko "score_batch 실패: $resp"
fi

# ----------------------------------------------------------------------------
banner "[5/6] μ /v1/completions — 단일 (n=1)"
t0=$(date +%s%N)
payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
  "prompt": "<|im_start|>system\nYou solve math step by step. Number each step on its own line.<|im_end|>\n<|im_start|>user\nWhat is 7 + 5?<|im_end|>\n<|im_start|>assistant\n",
  "n": 1,
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 64,
  "stop": ["\n\n"]
}
EOF
)
if resp=$(req POST "$MU_ENDPOINT/v1/completions" "$payload"); then
  text=$(echo "$resp" | python3 -c "import sys, json; d=json.load(sys.stdin); print(repr(d['choices'][0]['text']))" 2>/dev/null || echo "$resp" | head -c 300)
  ok "응답 ($(ms_since $t0)ms): $text"
else
  ko "completions(n=1) 실패: $resp"
fi

# ----------------------------------------------------------------------------
banner "[6/6] μ /v1/completions — K=16 alternatives (Phase 1 패턴)"
t0=$(date +%s%N)
payload=$(cat <<'EOF'
{
  "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
  "prompt": "<|im_start|>system\nYou solve math step by step. Number each step on its own line.<|im_end|>\n<|im_start|>user\nSolve x^2 = 9.<|im_end|>\n<|im_start|>assistant\n",
  "n": 16,
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 64,
  "stop": ["\n\n"]
}
EOF
)
if resp=$(req POST "$MU_ENDPOINT/v1/completions" "$payload"); then
  count=$(echo "$resp" | python3 -c "import sys, json; d=json.load(sys.stdin); print(len(d['choices']))" 2>/dev/null || echo "?")
  uniq=$(echo "$resp" | python3 -c "import sys, json; d=json.load(sys.stdin); texts=[c['text'].strip() for c in d['choices']]; print(len(set(texts)))" 2>/dev/null || echo "?")
  ok "K=16 응답 ($(ms_since $t0)ms): $count개 반환, $uniq개 unique (다양성 확인)"
else
  ko "completions(n=16) 실패: $resp"
fi

# ============================================================================
echo
echo -e "${B}━━ 결과 ━━${N}"
echo -e "  통과: ${G}${pass}${N}"
echo -e "  실패: ${R}${fail}${N}"
if [[ $fail -eq 0 ]]; then
  echo -e "\n${G}✓ 분산 추론 스택 정상.${N} 학습 PC에서 'bash run_train.sh --mode phase1' 진행 가능."
  exit 0
else
  echo -e "\n${R}✗ 일부 호출 실패 — 위 로그 확인.${N}"
  exit 1
fi
