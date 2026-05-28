"""
Dataset loaders and mixer for olympiad-level math evaluation.

Sources:
  - MathNet  : hendrycks/competition_math  (Level 4–5 filtered)
  - OlympiadBench: GAIR/OlympiadBench
"""

import random
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str:
    """Return the content of the last \\boxed{} in text, or empty string."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip() if depth == 0 else ""


# ---------------------------------------------------------------------------
# MathNet (hendrycks/competition_math)
# ---------------------------------------------------------------------------

def load_mathnet_olympiad(
    levels: list[int],
    types: list[str],
    seed: int = 42,
) -> list[dict]:
    """
    Load MATH benchmark filtered to the requested difficulty levels.
    answers are extracted from the \\boxed{} in each solution.
    """
    from datasets import load_dataset

    ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)

    level_strs = {f"Level {l}" for l in levels}
    type_set = set(types) if types else None

    result: list[dict] = []
    for i, ex in enumerate(ds):
        if ex["level"] not in level_strs:
            continue
        if type_set and ex["type"] not in type_set:
            continue

        answer = _extract_boxed(ex["solution"])
        if not answer:
            continue  # skip problems with no parseable answer

        result.append(
            {
                "id": f"mathnet_{i}",
                "source": "mathnet",
                "problem": ex["problem"],
                "answer": answer,
                "solution": ex["solution"],
                "level": ex["level"],
                "subject": ex["type"],
            }
        )

    rng = random.Random(seed)
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# OlympiadBench (GAIR/OlympiadBench)
# ---------------------------------------------------------------------------

_OLYMPIAD_CONFIGS = {
    "en": ["OE_TO_maths_en_COMP"],
    "zh": ["OE_TO_maths_zh_COMP"],
    "both": ["OE_TO_maths_en_COMP", "OE_TO_maths_zh_COMP"],
}


def load_olympiadbench(language: str = "en", seed: int = 42) -> list[dict]:
    """Load OlympiadBench math problems (open-ended, competition subset)."""
    from datasets import load_dataset

    configs = _OLYMPIAD_CONFIGS.get(language, _OLYMPIAD_CONFIGS["en"])
    result: list[dict] = []

    for cfg_name in configs:
        try:
            ds = load_dataset("GAIR/OlympiadBench", cfg_name, split="test", trust_remote_code=True)
        except Exception as exc:
            print(f"[WARN] Failed to load OlympiadBench config '{cfg_name}': {exc}")
            continue

        for i, ex in enumerate(ds):
            problem = ex.get("problem") or ex.get("question") or ""
            if not problem:
                continue

            # answer field varies; try common keys
            answer = (
                ex.get("final_answer")
                or ex.get("answer")
                or ex.get("solution")
                or ""
            )
            # Some entries have answer as a list
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer).strip()

            result.append(
                {
                    "id": f"olympiadbench_{cfg_name}_{i}",
                    "source": "olympiadbench",
                    "problem": problem,
                    "answer": answer,
                    "solution": ex.get("solution", ""),
                    "subject": ex.get("subject", ""),
                    "level": ex.get("difficulty", ""),
                    "language": "zh" if "zh" in cfg_name else "en",
                }
            )

    rng = random.Random(seed)
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------

def mix_datasets(
    mathnet: list[dict],
    olympiadbench: list[dict],
    total_samples: int,
    mathnet_ratio: float,
    seed: int = 42,
) -> list[dict]:
    """
    Sample `total_samples` problems according to `mathnet_ratio`.
    Caps each source to its available size.
    """
    n_math = min(int(total_samples * mathnet_ratio), len(mathnet))
    n_olym = min(total_samples - n_math, len(olympiadbench))

    mixed = mathnet[:n_math] + olympiadbench[:n_olym]

    rng = random.Random(seed)
    rng.shuffle(mixed)
    return mixed
