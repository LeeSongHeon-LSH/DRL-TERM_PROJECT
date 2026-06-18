"""
Answer extraction and grading utilities.

Supports multiple answer formats:
- GSM8K: #### number format
- MATH/AIME/Olympiad: \\boxed{} format

Uses sympy for symbolic equivalence checking with fallback to string normalization.
"""

import re
from typing import Optional, Tuple

import sympy
from sympy import simplify, sympify


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} format.
    
    Handles nested braces and various LaTeX expressions.
    
    Args:
        text: The text containing the answer.
        
    Returns:
        The extracted answer string, or None if not found.
    """
    # Pattern to match \boxed{...} with proper brace matching
    # First try simple pattern
    pattern = r"\\boxed\s*\{([^{}]+)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # Try with nested braces
    pattern = r"\\boxed\s*\{((?:[^{}]|\\{|\\}|\\[{}\\])*)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # Try with multiple braces (for complex expressions)
    # Find all \boxed{ and match braces
    start_idx = text.find(r"\boxed{")
    if start_idx == -1:
        start_idx = text.find(r"\boxed {")
    if start_idx == -1:
        return None
    
    # Find the opening brace after \boxed
    brace_start = text.find("{", start_idx)
    if brace_start == -1:
        return None
    
    # Count braces to find matching close
    brace_count = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[brace_start + 1:i].strip()
    
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract answer from GSM8K format (#### number).
    
    Args:
        text: The text containing the answer.
        
    Returns:
        The extracted answer string, or None if not found.
    """
    # Look for #### followed by a number
    pattern = r"####\s*(-?[\d,]+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        # Remove commas from the number
        return match.group(1).replace(",", "")
    
    # Fallback: look for the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def extract_last_number(text: str) -> Optional[str]:
    """
    Extract the last number from text as a fallback.
    
    Args:
        text: The text to search.
        
    Returns:
        The last number found, or None.
    """
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_answer(text: str, dataset_type: str = "math") -> Optional[str]:
    """
    Extract answer from model output based on dataset type.
    
    Priority order:
    1. \\boxed{} format (if math-type dataset)
    2. #### format (if GSM8K)
    3. Last number in text
    
    Args:
        text: The model output text.
        dataset_type: Type of dataset ('gsm8k', 'math', 'aime', 'olympiad').
        
    Returns:
        The extracted answer string, or None if not found.
    """
    # Always try boxed first for math-type datasets
    if dataset_type in ("math", "aime", "olympiad"):
        boxed = extract_boxed_answer(text)
        if boxed is not None:
            return boxed
    
    # Try GSM8K format
    if dataset_type == "gsm8k":
        gsm8k = extract_gsm8k_answer(text)
        if gsm8k is not None:
            return gsm8k
    
    # Fallback to last number
    return extract_last_number(text)


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    
    Removes LaTeX formatting, extra whitespace, and standardizes notation.
    
    Args:
        answer: The answer string to normalize.
        
    Returns:
        Normalized answer string.
    """
    if answer is None:
        return ""
    
    # Convert to string and strip
    answer = str(answer).strip()
    
    # Remove common LaTeX commands
    answer = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\left\s*|\\right\s*", "", answer)
    
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", answer)
    
    # Normalize common symbols
    answer = answer.replace("\\pi", "pi")
    answer = answer.replace("\\infty", "inf")
    
    # Remove extra whitespace
    answer = " ".join(answer.split())
    
    # Remove outer parentheses if the whole expression is wrapped
    while answer.startswith("(") and answer.endswith(")"):
        inner = answer[1:-1]
        # Check if parentheses are balanced
        depth = 0
        balanced = True
        for c in inner:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth < 0:
                balanced = False
                break
        if balanced and depth == 0:
            answer = inner
        else:
            break
    
    return answer


def sympy_equal(a: str, b: str) -> bool:
    """
    Check if two answers are mathematically equivalent using sympy.
    
    Args:
        a: First answer string.
        b: Second answer string.
        
    Returns:
        True if the answers are mathematically equivalent.
    """
    try:
        # Normalize both answers
        a_norm = normalize_answer(a)
        b_norm = normalize_answer(b)
        
        # Try to parse as sympy expressions
        expr_a = sympify(a_norm)
        expr_b = sympify(b_norm)
        
        # Check if they simplify to the same thing
        diff = simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        # If sympy fails, fall back to string comparison
        return normalize_answer(a) == normalize_answer(b)


def try_math_verify(a: str, b: str) -> Optional[bool]:
    """
    Try to use math_verify library for answer comparison.
    
    Args:
        a: First answer string.
        b: Second answer string.
        
    Returns:
        True/False if math_verify succeeds, None if not available.
    """
    try:
        from math_verify import verify_answer
        
        result = verify_answer(a, b)
        return result
    except ImportError:
        return None
    except Exception:
        return None


def grade_answer(
    prediction: str,
    gold: str,
    dataset_type: str = "math",
) -> Tuple[bool, str, str]:
    """
    Grade a prediction against a gold answer.
    
    Uses the following comparison strategy:
    1. Try math_verify if available
    2. Try sympy symbolic equivalence
    3. Fall back to normalized string comparison
    
    Args:
        prediction: The model's predicted answer.
        gold: The gold answer.
        dataset_type: Type of dataset.
        
    Returns:
        Tuple of (is_correct, normalized_prediction, normalized_gold).
    """
    # Extract answers
    pred_answer = extract_answer(prediction, dataset_type)
    gold_answer = gold.strip() if gold else None
    
    if pred_answer is None or gold_answer is None:
        return False, str(pred_answer), str(gold_answer)
    
    # Normalize
    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)
    
    # Try math_verify first
    mv_result = try_math_verify(pred_norm, gold_norm)
    if mv_result is not None:
        return mv_result, pred_norm, gold_norm
    
    # Try sympy equivalence
    if sympy_equal(pred_norm, gold_norm):
        return True, pred_norm, gold_norm
    
    # Final fallback: string comparison
    return pred_norm == gold_norm, pred_norm, gold_norm