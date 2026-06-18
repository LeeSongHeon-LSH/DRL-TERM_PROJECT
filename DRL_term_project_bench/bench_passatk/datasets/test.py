"""Test dataset loader for smoke tests."""

import json
from pathlib import Path
from typing import Dict, List


class TestLoader:
    """Loader for test dataset (used in smoke tests)."""
    
    @property
    def name(self) -> str:
        return "test"
    
    def load(self) -> List[Dict]:
        """
        Load test problems from a JSONL file.
        
        Looks for 'test.jsonl' in the output directory.
        
        Returns:
            List of problems with 'id', 'problem', 'gold'.
        """
        # Try to find test.jsonl in common locations
        possible_paths = [
            Path("runs/smoke_test/test.jsonl"),
            Path("test.jsonl"),
        ]
        
        for path in possible_paths:
            if path.exists():
                problems = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            problems.append(json.loads(line))
                return problems
        
        # Return default test problems if no file found
        return [
            {
                "id": "test/0",
                "problem": "What is 2 + 2?",
                "gold": "4",
                "level": None,
            },
            {
                "id": "test/1",
                "problem": "What is 3 * 4?",
                "gold": "12",
                "level": None,
            },
            {
                "id": "test/2",
                "problem": "If x = 5, what is x + 10?",
                "gold": "15",
                "level": None,
            },
            {
                "id": "test/3",
                "problem": "What is the square root of 16?",
                "gold": "4",
                "level": None,
            },
            {
                "id": "test/4",
                "problem": "What is 100 / 25?",
                "gold": "4",
                "level": None,
            },
        ]