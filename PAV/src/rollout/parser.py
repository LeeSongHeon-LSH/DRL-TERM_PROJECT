"""Step 경계 파싱 — 정책 출력을 step 단위로 분할.

정책(Qwen2.5-Math)는 보통 `\n\n` 또는 `Step k:` 헤더로 step을 구분.
Skywork PRM은 단일 `\n`을 step 경계로 사용 — score.py에서 자동 정규화하므로 여기서는
사람-친화적 분할 결과만 돌려주면 됨.

지원 경계:
    1) "Step k:" / "Step k." / "## Step k" 헤더
    2) blank-line (`\n\n+`)
    3) (fallback) 단일 newline
"""
from __future__ import annotations

import re

# "Step k:" / "Step k." / "## Step k" — 줄 시작
_STEP_HEADER_RE = re.compile(
    r"^\s*(?:#{1,3}\s*)?Step\s*\d+\s*[:\.]?", re.IGNORECASE | re.MULTILINE
)
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def split_steps(text: str) -> list[str]:
    """Solution을 step 리스트로 분할. 각 step은 trailing newline 제거."""
    if not text or not text.strip():
        return []

    if _STEP_HEADER_RE.search(text):
        positions = [m.start() for m in _STEP_HEADER_RE.finditer(text)]
        positions.append(len(text))
        steps = [
            text[positions[i] : positions[i + 1]].strip()
            for i in range(len(positions) - 1)
        ]
        return [s for s in steps if s]

    # blank-line 분할 우선
    parts = _BLANK_LINE_RE.split(text.strip())
    steps = [p.strip() for p in parts if p.strip()]
    if len(steps) > 1:
        return steps

    # fallback: 단일 newline (Skywork 형식)
    return [s.strip() for s in text.strip().split("\n") if s.strip()]


def normalize_step(step: str, with_trailing_sep: bool = True) -> str:
    """step 문자열 정규화 — PAVRewardFn에서 prefix 누적 시 사용. 기본은 \n 부착."""
    s = step.strip()
    if with_trailing_sep and not s.endswith("\n"):
        s = s + "\n"
    return s


def join_steps(steps: list[str]) -> str:
    """step 리스트 → 하나의 solution 문자열. 단일 \n 구분 (Skywork PRM 형식)."""
    return "\n".join(s.strip() for s in steps if s.strip())
