"""OlympiadBench dataset loader."""

import re
from typing import Dict, List

from datasets import load_dataset


class OlympiadLoader:
    """Loader for OlympiadBench math-en subset."""
    
    @property
    def name(self) -> str:
        return "OlympiadBench"
    
    def load(self) -> List[Dict]:
        """
        Load OlympiadBench math-en subset.
        
        Uses OE_TO_maths_en_COMP config (Open-ended, Text-only, English, Competition).
        
        Returns:
            List of problems with 'id', 'problem', 'gold'.
        """
        problems = []
        
        try:
            # Load OE_TO_maths_en_COMP (Open-ended, Text-only, English, Competition)
            dataset = load_dataset(
                "Hothan/OlympiadBench",
                "OE_TO_maths_en_COMP",
                split="train"
            )
            
            for item in dataset:
                # Extract problem info
                problem_id = str(item.get("id", ""))
                question = item.get("question", "")
                final_answer = item.get("final_answer", "")
                
                # Handle list answers
                if isinstance(final_answer, list):
                    if len(final_answer) > 0:
                        gold = final_answer[0]
                    else:
                        gold = ""
                else:
                    gold = str(final_answer)
                
                # Clean up LaTeX
                gold = self._clean_latex(gold)
                
                problems.append({
                    "id": f"olympiad/{problem_id}",
                    "problem": question,
                    "gold": gold,
                    "level": item.get("difficulty", None),
                    "subject": item.get("subject", None),
                    "subfield": item.get("subfield", None),
                })
            
            print(f"Loaded {len(problems)} OlympiadBench problems from OE_TO_maths_en_COMP")
        
        except Exception as e:
            print(f"Error loading OlympiadBench: {e}")
            print("Warning: OlympiadBench dataset not found. Please provide custom data.")
            return []
        
        return problems
    
    def _clean_latex(self, text: str) -> str:
        """Clean up LaTeX formatting."""
        # Remove $...$ wrapper if present
        text = text.strip()
        if text.startswith("$") and text.endswith("$"):
            text = text[1:-1]
        
        # Remove \text{...} wrapper
        text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
        
        return text.strip()