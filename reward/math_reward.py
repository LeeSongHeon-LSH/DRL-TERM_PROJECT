import re


def _extract_answer(text: str) -> str | None:
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def score(response: str, reference: str) -> float:
    pred = _extract_answer(response)
    if pred is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(reference)) < 1e-6 else 0.0
    except ValueError:
        return float(pred.strip() == reference.strip())


def batch_score(
    responses: list[list[str]], references: list[str]
) -> list[list[float]]:
    """responses: (B, G) list-of-lists; references: (B,) list."""
    return [
        [score(r, ref) for r in group]
        for group, ref in zip(responses, references)
    ]
