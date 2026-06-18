"""I/O utilities for saving and loading benchmark results."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_results(
    output_path: Path,
    problem_id: str,
    problem: str,
    gold: str,
    samples: List[Dict[str, Any]],
    per_problem: Dict[str, Any],
) -> None:
    """
    Save results for a single problem to a JSONL file.
    
    Appends to the file if it exists, creates it otherwise.
    
    Args:
        output_path: Path to the output JSONL file.
        problem_id: Unique identifier for the problem.
        problem: The problem text.
        gold: The gold answer.
        samples: List of sample dictionaries with 'text', 'pred', 'is_correct'.
        per_problem: Dictionary of per-problem metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "problem_id": problem_id,
        "problem": problem,
        "gold": gold,
        "samples": samples,
        "per_problem": per_problem,
    }
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_results(output_path: Path) -> List[Dict[str, Any]]:
    """
    Load all results from a JSONL file.
    
    Args:
        output_path: Path to the JSONL file.
        
    Returns:
        List of result dictionaries.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return []
    
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_completed_ids(output_path: Path) -> set:
    """
    Get the set of problem IDs that have already been processed.
    
    Used for resume functionality - skip already completed problems.
    
    Args:
        output_path: Path to the JSONL file.
        
    Returns:
        Set of problem_id strings that have been completed.
    """
    results = load_results(output_path)
    return {r["problem_id"] for r in results}


def save_config(config: Dict[str, Any], output_dir: Path) -> None:
    """
    Save the configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        output_dir: Directory to save the config file.
    """
    import yaml
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load the configuration from a YAML file.
    
    Args:
        output_dir: Directory containing the config file.
        
    Returns:
        Configuration dictionary, or None if not found.
    """
    import yaml
    
    config_path = Path(output_dir) / "config.yaml"
    if not config_path.exists():
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)