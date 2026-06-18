#!/usr/bin/env python3
"""
Extract and compare the actual solution texts for problems the few-shot model
solved but the zero-shot models failed on (AIME 2023-2025 combined).

For each differential problem it pulls:
  - a CORRECT few-shot solution,
  - a representative (most-common-prediction) attempt from each zero-shot model,
so the reasoning can be compared side by side.

Output: runs/AIME_fewshot_solution_comparison.md

Usage:
    python -m bench_passatk.extract_solutions
"""

import json
from collections import Counter
from pathlib import Path

YEARS = ["AIME2023", "AIME2024", "AIME2025"]
MODELS = {
    "few-shot": "runs/pav-fewshot-ckpt500-aime3",
    "PAV-distribution": "runs/pav-checkpoint-500",
    "PAV-scalar-c2": "runs/pav-scalar-c2-checkpoint-500",
    "Baseline": "runs/qwen-math-1.5b-baseline",
}
MAXLEN = 2200  # trim very long solutions


def load(d):
    out = {}
    for y in YEARS:
        for line in open(Path(d) / (y + ".jsonl"), encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_year"] = y
                out[r["problem_id"]] = r
    return out


def uses_code(t):
    return "```python" in t or "import " in t


def trim(t):
    t = t.strip()
    return t if len(t) <= MAXLEN else t[:MAXLEN] + "\n…(이하 생략)…"


def fence(t):
    # outer fence longer than any inner backtick run so nested ``` survive
    return "````text\n" + t + "\n````"


def pick_correct(samples):
    cor = [s for s in samples if s.get("is_correct")]
    if not cor:
        return None
    # prefer a code solution if present (representative of the winning strategy)
    code = [s for s in cor if uses_code(s["text"])]
    return (code or cor)[0]


def pick_representative_wrong(samples):
    wrong = [s for s in samples if not s.get("is_correct")]
    if not wrong:
        return None
    # most common wrong prediction = typical failure mode
    cnt = Counter(str(s.get("pred")) for s in wrong)
    top = cnt.most_common(1)[0][0]
    for s in wrong:
        if str(s.get("pred")) == top:
            return s
    return wrong[0]


def main():
    data = {n: load(d) for n, d in MODELS.items()}
    ids = list(data["few-shot"])

    def c(n, i):
        return data[n][i]["per_problem"]["c"]

    def solved(n, i):
        return data[n][i]["per_problem"]["oracle"]

    # few-shot solved AND PAV-distribution failed
    diff = [i for i in ids if solved("few-shot", i) and not solved("PAV-distribution", i)]
    # sort by how dramatic (few-shot correct count desc)
    diff.sort(key=lambda i: -c("few-shot", i))

    L = []
    L.append("# few-shot 정답 / 다른 모델 오답 — 실제 풀이 비교\n")
    L.append("> 생성일: 2026-06-12")
    L.append("> 대상: few-shot 모델이 맞추고(oracle 정답) PAV-distribution이 틀린 문제. "
             "각 모델에서 대표 풀이 1개씩 추출.")
    L.append("> few-shot=정답 샘플(가능하면 코드 풀이), zero-shot 3종=최빈 오답 풀이.\n")
    L.append(f"**차별 문제 수: {len(diff)}개** (few-shot 정답수 내림차순)\n")

    # summary table
    L.append("| 문제 | 연도 | 정답 | few-shot c | PAV-dist c | scalar c | Base c | few-shot 방식 |")
    L.append("|------|------|------|-----------|-----------|---------|--------|---------------|")
    for i in diff:
        fc = pick_correct(data["few-shot"][i]["samples"])
        method = "코드(TIR)" if (fc and uses_code(fc["text"])) else "자연어"
        L.append(f"| `{i}` | {data['few-shot'][i]['_year']} | {data['few-shot'][i]['gold']} "
                 f"| {c('few-shot',i)} | {c('PAV-distribution',i)} | {c('PAV-scalar-c2',i)} "
                 f"| {c('Baseline',i)} | {method} |")
    L.append("")

    for i in diff:
        prob = data["few-shot"][i]
        L.append("\n---\n")
        L.append(f"## `{i}`  ({prob['_year']}) — 정답: **{prob['gold']}**\n")
        L.append("**문제**\n")
        L.append(fence(trim(prob["problem"])))
        L.append("")

        # few-shot correct
        fc = pick_correct(prob["samples"])
        tag = " (코드/TIR)" if (fc and uses_code(fc["text"])) else " (자연어)"
        L.append(f"### ✅ few-shot 정답 풀이{tag}  — pred={fc.get('pred') if fc else 'N/A'}\n")
        L.append(fence(trim(fc["text"])) if fc else "_(정답 샘플 없음)_")
        L.append("")

        # wrong attempts from the three zero-shot models
        for n in ["PAV-distribution", "PAV-scalar-c2", "Baseline"]:
            w = pick_representative_wrong(data[n][i]["samples"])
            L.append(f"### ❌ {n} 대표 오답  — pred={w.get('pred') if w else 'N/A'}  (c={c(n,i)}/256)\n")
            L.append(fence(trim(w["text"])) if w else "_(샘플 없음)_")
            L.append("")

    out = Path("runs/AIME_fewshot_solution_comparison.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"{len(diff)} problems written to: {out}")


if __name__ == "__main__":
    main()
