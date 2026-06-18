"""
AIME dataset loading with multiple fallback sources per year.

Normalised schema → AIMEProblem dataclass
  year            : int   (2023 / 2024 / 2025)
  competition     : str   ("I" or "II")
  problem_number  : int   (1-15)
  problem         : str   (problem text)
  answer          : int   (0-999)
  source          : str   (HuggingFace dataset name)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import datasets
from tqdm import tqdm


@dataclass
class AIMEProblem:
    year: int
    competition: str
    problem_number: int
    problem: str
    answer: int
    source: str = ""

    @property
    def problem_id(self) -> str:
        return f"AIME_{self.year}_{self.competition}_{self.problem_number:02d}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_aops_url(url: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """Extract (year, competition, problem_number) from an AoPS problem URL."""
    m = re.search(r"(\d{4})_AIME_([I]+)_Problems/Problem_(\d+)", url or "")
    if m:
        return int(m.group(1)), m.group(2), int(m.group(3))
    return None, None, None


def _competition_from_string(s: str) -> str:
    """Return 'II' if string contains 'II', else 'I'."""
    return "II" if "II" in str(s) else "I"


def _safe_int(val, default: int = -1) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def _load_aimo_validation_aime(years: List[int]) -> List[AIMEProblem]:
    """AI-MO/aimo-validation-aime — primary source for 2023 & 2024."""
    ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")
    problems: List[AIMEProblem] = []
    for row in ds:
        year, comp, num = _parse_aops_url(row.get("url", ""))
        if year is None or year not in years:
            continue
        ans = _safe_int(row.get("answer", -1))
        if ans < 0:
            continue
        problems.append(AIMEProblem(
            year=year,
            competition=comp or "I",
            problem_number=num or 0,
            problem=row["problem"],
            answer=ans,
            source="AI-MO/aimo-validation-aime",
        ))
    return problems


def _load_maxwell_aime(years: List[int]) -> List[AIMEProblem]:
    """Maxwell-Jia/AIME_1983_2024 — fallback for 2023 & 2024."""
    ds = datasets.load_dataset("Maxwell-Jia/AIME_1983_2024", split="train")
    problems: List[AIMEProblem] = []
    for row in ds:
        year = _safe_int(row.get("Year", row.get("year", -1)))
        if year not in years:
            continue
        id_str = str(row.get("ID", row.get("id", "")))
        comp = _competition_from_string(id_str)
        num = _safe_int(row.get("Problem Number", row.get("problem_number", 0)))
        ans = _safe_int(row.get("Answer", row.get("answer", -1)))
        if ans < 0:
            continue
        problems.append(AIMEProblem(
            year=year,
            competition=comp,
            problem_number=num,
            problem=str(row.get("Problem", row.get("problem", ""))),
            answer=ans,
            source="Maxwell-Jia/AIME_1983_2024",
        ))
    return problems


_SOURCES_2025 = [
    # (dataset_name, split)
    ("AI-MO/aimo-validation-aime-2025", "train"),
    ("openai/aime-2025",                "test"),
    ("HuggingFaceH4/aime_2025",         "train"),
    ("Maxwell-Jia/AIME_2025",           "train"),
    ("yentinglin/aime_2025",            "train"),
]


def _load_aime_2025() -> List[AIMEProblem]:
    """Try multiple known HuggingFace sources for AIME 2025."""
    for ds_name, split in _SOURCES_2025:
        try:
            ds = datasets.load_dataset(ds_name, split=split, trust_remote_code=True)
            problems: List[AIMEProblem] = []
            for i, row in enumerate(ds):
                problem_text = (
                    row.get("problem")
                    or row.get("Problem")
                    or row.get("question")
                    or ""
                )
                ans = _safe_int(
                    row.get("answer", row.get("Answer", row.get("solution", -1)))
                )
                comp_raw = row.get("competition", row.get("Competition", row.get("split", "")))
                comp = _competition_from_string(comp_raw)
                num = _safe_int(
                    row.get("problem_number", row.get("Problem Number", row.get("id", i + 1)))
                )
                if not problem_text or ans < 0:
                    continue
                problems.append(AIMEProblem(
                    year=2025,
                    competition=comp,
                    problem_number=num,
                    problem=problem_text,
                    answer=ans,
                    source=ds_name,
                ))
            if problems:
                print(f"  Loaded AIME 2025 from {ds_name}: {len(problems)} problems")
                return problems
        except Exception as exc:
            print(f"  {ds_name}: {exc}")

    print("  WARNING: Could not load AIME 2025 from any source. Skipping.")
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_aime_problems(years: List[int] = None) -> Dict[int, List[AIMEProblem]]:
    """
    Load AIME problems for each requested year.
    Returns {year: [AIMEProblem, ...]} sorted by (competition, problem_number).
    """
    if years is None:
        years = [2023, 2024, 2025]

    result: Dict[int, List[AIMEProblem]] = {y: [] for y in years}
    non_2025 = [y for y in years if y != 2025]

    # --- 2023 / 2024 ---
    if non_2025:
        print(f"Loading AIME data for years {non_2025}...")
        loaded = False

        for loader, name in [
            (_load_aimo_validation_aime, "AI-MO/aimo-validation-aime"),
            (_load_maxwell_aime,          "Maxwell-Jia/AIME_1983_2024"),
        ]:
            try:
                problems = loader(non_2025)
                for p in problems:
                    if p.year in result:
                        result[p.year].append(p)
                counts = {y: len(result[y]) for y in non_2025}
                print(f"  Loaded from {name}: {counts}")
                loaded = True
                # Stop if all years have ≥ 15 problems (one full competition)
                if all(len(result[y]) >= 15 for y in non_2025):
                    break
            except Exception as exc:
                print(f"  {name} failed: {exc}")

        if not loaded:
            print("  WARNING: Could not load 2023/2024 AIME data from any source.")

    # --- 2025 ---
    if 2025 in years:
        print("Loading AIME 2025...")
        result[2025] = _load_aime_2025()

    # Sort each year's problems
    for y in result:
        result[y].sort(key=lambda p: (p.competition, p.problem_number))

    # Summary
    print("\nDataset summary:")
    for y, probs in result.items():
        comps = {}
        for p in probs:
            comps.setdefault(p.competition, 0)
            comps[p.competition] += 1
        print(f"  AIME {y}: {len(probs)} problems — {comps}")

    return result
