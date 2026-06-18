"""AIME dataset loader."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


class AIMELoader:
    """Loader for AIME dataset with year filtering support.
    
    Supported years and their data sources:
    - 1983~2024: di-zhang-fdu/AIME_1983_2024 (HuggingFace)
    - 2024: HuggingFaceH4/aime_2024 (HuggingFace, preferred)
    - 2025: yentinglin/aime_2025 (HuggingFace)
    - 2026+: Custom JSON/JSONL file (local path or HuggingFace if available)
    
    For years not available on HuggingFace (e.g., 2026+), you can provide
    a local JSON/JSONL file via the AIME_CUSTOM_PATH environment variable or
    by placing a file at data/aime_{year}.json or data/aime_{year}.jsonl.
    The file should contain a list of objects with 'problem' and 'answer' fields,
    and optionally 'id' and 'url' fields.
    """
    
    # Valid years for AIME (updated to include 2025+)
    VALID_YEARS = [str(y) for y in range(1983, 2027)]
    
    # HuggingFace dataset sources per year
    HF_DATASETS = {
        "2024": ("HuggingFaceH4/aime_2024", "train"),
        "2025": ("yentinglin/aime_2025", "train"),
    }
    
    def __init__(self, year: Optional[str] = None, custom_path: Optional[str] = None):
        """
        Initialize AIME loader.
        
        Args:
            year: Year of AIME (e.g., '2024'). If None, loads all available.
            custom_path: Path to custom JSON/JSONL file for years not on HuggingFace.
                         Can also be set via AIME_CUSTOM_PATH environment variable.
        """
        self.year = year
        self.custom_path = custom_path or os.environ.get("AIME_CUSTOM_PATH", None)
    
    @property
    def name(self) -> str:
        if self.year:
            return f"AIME{self.year}"
        return "AIME"
    
    def _load_from_hf_2024(self) -> Optional[List[Dict]]:
        """Load AIME 2024 from HuggingFaceH4/aime_2024."""
        try:
            dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
            problems = []
            for item in dataset:
                problems.append({
                    "id": f"aime/{item['id']}",
                    "problem": item["problem"],
                    "gold": str(item["answer"]),
                    "level": None,
                    "year": 2024,
                })
            print(f"Loaded {len(problems)} AIME 2024 problems from HuggingFaceH4/aime_2024")
            return problems
        except Exception as e:
            print(f"Error loading HuggingFaceH4/aime_2024: {e}")
            return None
    
    def _load_from_hf_2025(self) -> Optional[List[Dict]]:
        """Load AIME 2025 from yentinglin/aime_2025."""
        try:
            dataset = load_dataset("yentinglin/aime_2025", split="train")
            problems = []
            for i, item in enumerate(dataset):
                problems.append({
                    "id": f"aime/{item.get('id', i)}",
                    "problem": item["problem"],
                    "gold": str(item["answer"]),
                    "level": None,
                    "year": 2025,
                })
            print(f"Loaded {len(problems)} AIME 2025 problems from yentinglin/aime_2025")
            return problems
        except Exception as e:
            print(f"Error loading yentinglin/aime_2025: {e}")
            return None
    
    def _load_from_hf_year(self, year: str) -> Optional[List[Dict]]:
        """Load AIME for a specific year from HuggingFace if available."""
        if year in self.HF_DATASETS:
            repo, split = self.HF_DATASETS[year]
            try:
                dataset = load_dataset(repo, split=split)
                problems = []
                for i, item in enumerate(dataset):
                    # Handle different field names across datasets
                    problem_text = item.get("problem", item.get("question", ""))
                    answer_text = str(item.get("answer", item.get("Answer", "")))
                    item_id = item.get("id", item.get("ID", i))
                    problems.append({
                        "id": f"aime/{item_id}",
                        "problem": problem_text,
                        "gold": answer_text,
                        "level": None,
                        "year": int(year),
                    })
                print(f"Loaded {len(problems)} AIME {year} problems from {repo}")
                return problems
            except Exception as e:
                print(f"Error loading {repo}: {e}")
                return None
        return None
    
    def _load_from_custom(self, year: str) -> Optional[List[Dict]]:
        """Load AIME from a custom JSON/JSONL file.
        
        Search order:
        1. Explicit custom_path
        2. AIME_CUSTOM_PATH environment variable
        3. data/aime_{year}.json
        4. data/aime_{year}.jsonl
        """
        search_paths = []
        
        if self.custom_path:
            search_paths.append(Path(self.custom_path))
        
        # Check common local paths
        for base in [Path("data"), Path("bench_passatk/data"), Path(".")]:
            search_paths.append(base / f"aime_{year}.json")
            search_paths.append(base / f"aime_{year}.jsonl")
        
        for path in search_paths:
            if path.exists():
                try:
                    problems = self._parse_custom_file(path, year)
                    if problems:
                        print(f"Loaded {len(problems)} AIME {year} problems from {path}")
                        return problems
                except Exception as e:
                    print(f"Error loading custom AIME file {path}: {e}")
        
        return None
    
    def _parse_custom_file(self, path: Path, year: str) -> List[Dict]:
        """Parse a custom JSON or JSONL file."""
        problems = []
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("problems", data.get("data", [data]))
            if not isinstance(data, list):
                data = [data]
        elif suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        for i, item in enumerate(data):
            problem_text = item.get("problem", item.get("question", ""))
            answer_text = str(item.get("answer", item.get("gold", "")))
            item_id = item.get("id", i)
            problems.append({
                "id": f"aime/{item_id}",
                "problem": problem_text,
                "gold": answer_text,
                "level": item.get("level", None),
                "year": int(year),
            })
        
        return problems
    
    def load(self) -> List[Dict]:
        """
        Load AIME problems.
        
        Returns:
            List of problems with 'id', 'problem', 'gold'.
        """
        problems = []
        
        # If a specific year is requested
        if self.year:
            year = self.year
            
            # Try HuggingFace datasets first (for 2024, 2025, etc.)
            if year in self.HF_DATASETS:
                result = self._load_from_hf_year(year)
                if result is not None:
                    return result
                # Fallback for 2024: try the dedicated loader
                if year == "2024":
                    result = self._load_from_hf_2024()
                    if result is not None:
                        return result
            
            # For years 2025+ not in HF_DATASETS, try custom file
            if int(year) >= 2025:
                result = self._load_from_custom(year)
                if result is not None:
                    return result
                print(f"Warning: AIME {year} not found on HuggingFace and no custom data file found.")
                print(f"  To use AIME {year}, create a file at data/aime_{year}.json or data/aime_{year}.jsonl")
                print(f"  Or set AIME_CUSTOM_PATH environment variable to the file path.")
                print(f"  The file should contain a list of objects with 'problem' and 'answer' fields.")
                return []
            
            # For years 1983-2024, use di-zhang-fdu/AIME_1983_2024
            try:
                dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
                
                for item in dataset:
                    item_year = str(item.get("Year", ""))
                    if item_year != year:
                        continue
                    
                    problem_id = item.get("ID", "")
                    question = item.get("Question", "")
                    answer = item.get("Answer", "")
                    
                    part = item.get("Part", None)
                    if part:
                        problem_id = f"{problem_id}-{part}"
                    
                    problems.append({
                        "id": f"aime/{problem_id}",
                        "problem": question,
                        "gold": str(answer),
                        "level": None,
                        "year": item.get("Year", None),
                    })
                
                print(f"Loaded {len(problems)} AIME {year} problems from di-zhang-fdu/AIME_1983_2024")
                return problems
            
            except Exception as e:
                print(f"Error loading AIME dataset: {e}")
                print("Warning: AIME dataset not found. Please provide custom data.")
                return []
        
        # If no year specified, load all available
        # Load 1983-2024 from di-zhang-fdu
        try:
            dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
            for item in dataset:
                problem_id = item.get("ID", "")
                question = item.get("Question", "")
                answer = item.get("Answer", "")
                part = item.get("Part", None)
                if part:
                    problem_id = f"{problem_id}-{part}"
                problems.append({
                    "id": f"aime/{problem_id}",
                    "problem": question,
                    "gold": str(answer),
                    "level": None,
                    "year": item.get("Year", None),
                })
            print(f"Loaded {len(problems)} AIME problems (1983-2024) from di-zhang-fdu/AIME_1983_2024")
        except Exception as e:
            print(f"Error loading di-zhang-fdu/AIME_1983_2024: {e}")
        
        # Load 2025 from HuggingFace
        result_2025 = self._load_from_hf_2025()
        if result_2025:
            problems.extend(result_2025)
        
        # Try loading 2026+ from custom files
        for y in range(2026, 2030):
            result = self._load_from_custom(str(y))
            if result:
                problems.extend(result)
        
        return problems
    
    def _extract_boxed(self, text: str) -> str:
        """Extract answer from \\boxed{} format."""
        pattern = r"\\boxed\s*\{([^{}]+)\}"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        start_idx = text.find(r"\boxed{")
        if start_idx == -1:
            return ""
        
        brace_start = text.find("{", start_idx)
        if brace_start == -1:
            return ""
        
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[brace_start + 1:i].strip()
        
        return ""