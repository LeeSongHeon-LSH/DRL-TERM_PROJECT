"""MATH dataset loader."""

import re
from typing import Dict, List, Optional

from datasets import load_dataset


class MATHLoader:
    """Loader for MATH test split with level information."""
    
    @property
    def name(self) -> str:
        return "MATH"
    
    def load(self) -> List[Dict]:
        """
        Load MATH test split.
        
        Returns:
            List of problems with 'id', 'problem', 'gold', 'level', 'type'.
        """
        # Use lighteval/MATH-hard as the original lighteval/MATH is no longer available
        dataset = load_dataset("lighteval/MATH-hard", split="test")
        
        problems = []
        for idx, item in enumerate(dataset):
            problem = item["problem"]
            solution = item.get("solution", "")
            
            # Extract gold answer from \boxed{}
            gold = self._extract_boxed(solution)
            
            # Get level (1-5)
            level = item.get("level", None)
            if level is not None:
                # Convert to int if string
                if isinstance(level, str):
                    level_match = re.search(r"Level\s*(\d+)", level)
                    if level_match:
                        level = int(level_match.group(1))
            
            # Get problem type
            problem_type = item.get("type", None)
            
            problems.append({
                "id": f"math/{idx}",
                "problem": problem,
                "gold": gold,
                "level": level,
                "type": problem_type,
            })
        
        return problems
    
    def _extract_boxed(self, text: str) -> str:
        """Extract answer from \\boxed{} format."""
        # Try simple pattern first
        pattern = r"\\boxed\s*\{([^{}]+)\}"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        # Try with nested braces
        start_idx = text.find(r"\boxed{")
        if start_idx == -1:
            start_idx = text.find(r"\boxed {")
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