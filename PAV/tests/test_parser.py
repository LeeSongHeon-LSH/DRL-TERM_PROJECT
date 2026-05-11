"""parser.split_steps / normalize_step / join_steps."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rollout.parser import join_steps, normalize_step, split_steps


def test_split_step_headers():
    text = "Step 1: First.\n\nStep 2: Second.\n\nStep 3: Third."
    steps = split_steps(text)
    assert len(steps) == 3
    assert steps[0].startswith("Step 1")
    assert steps[2].startswith("Step 3")


def test_split_blank_line():
    text = "Para A.\n\nPara B.\n\nPara C."
    steps = split_steps(text)
    assert steps == ["Para A.", "Para B.", "Para C."]


def test_split_single_newline_fallback():
    text = "Line 1\nLine 2\nLine 3"
    steps = split_steps(text)
    assert steps == ["Line 1", "Line 2", "Line 3"]


def test_split_empty():
    assert split_steps("") == []
    assert split_steps("   \n\n   ") == []


def test_normalize_adds_newline():
    assert normalize_step("hello") == "hello\n"
    assert normalize_step("hello", with_trailing_sep=False) == "hello"


def test_join_steps():
    assert join_steps(["a", "b", "c"]) == "a\nb\nc"
    assert join_steps(["", "  ", "x"]) == "x"


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    n = 0
    for f in funcs:
        try:
            f()
            print(f"PASS  {f.__name__}")
            n += 1
        except Exception:
            print(f"FAIL  {f.__name__}")
            traceback.print_exc()
    print(f"\n{n}/{len(funcs)} passed")
