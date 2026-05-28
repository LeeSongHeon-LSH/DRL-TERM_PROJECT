"""
Olympiad math evaluation pipeline with Pass@k support.
Uses vLLM offline inference on Qwen2.5-Math-1.5B-Instruct.

Pass@k uses the unbiased estimator from the Codex paper (Chen et al. 2021):
  pass@k = E_problems[ 1 - C(n-c, k) / C(n, k) ]
where n = samples per problem, c = correct samples.
"""

import json
import re
import time
from pathlib import Path

import yaml

from data.olympiad_dataset import load_mathnet_olympiad, load_olympiadbench, mix_datasets


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert mathematician specializing in mathematical olympiad problems. "
    "Solve the problem step by step with rigorous reasoning. "
    "At the very end, state your final answer inside \\boxed{...}."
)


def build_prompt(problem: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Answer parsing & scoring
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Return content of the last \\boxed{} in model output, or None."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip() if depth == 0 else None


def _normalize(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r"\s+", "", ans)
    ans = re.sub(r"(\.\d*?)0+$", r"\1", ans)
    ans = ans.rstrip(".")
    return ans.lower()


def is_correct(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None or not ground_truth:
        return False
    return _normalize(predicted) == _normalize(ground_truth)


# ---------------------------------------------------------------------------
# Pass@k — unbiased estimator (numerically stable)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased Pass@k estimator.
    Uses iterative product to avoid combinatorial overflow for large n, k.

    Returns the probability that at least one of k randomly drawn samples
    (without replacement) from n total is correct, averaged to an expectation
    when used over many problems.
    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # Probability that ALL k drawn samples are wrong:
    # prod_{i=0}^{k-1} (n - c - i) / (n - i)
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
        if prob_all_wrong == 0.0:
            return 1.0
    return 1.0 - prob_all_wrong


def aggregate_pass_at_k(
    per_problem_c: list[int],
    n: int,
    k_values: list[int],
) -> dict[str, float]:
    """
    Average pass@k over a list of problems.
    per_problem_c[i] = number of correct samples for problem i (out of n).
    """
    results: dict[str, float] = {}
    for k in k_values:
        if k > n:
            print(f"[WARN] pass@{k} skipped — k={k} > n={n}")
            continue
        scores = [pass_at_k(n, c, k) for c in per_problem_c]
        results[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str = "config/olympiad_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_samples  = cfg["generation"]["n"]
    pass_k_cfg = cfg["eval"]["pass_k"]

    # ---- Load datasets ----
    print("[1/4] Loading datasets...")
    mathnet = load_mathnet_olympiad(
        levels=cfg["dataset"]["mathnet_levels"],
        types=cfg["dataset"].get("mathnet_types", []),
        seed=cfg["dataset"]["seed"],
    )
    olympiad = load_olympiadbench(
        language=cfg["dataset"]["olympiadbench_language"],
        seed=cfg["dataset"]["seed"],
    )
    print(f"      MathNet (Level {cfg['dataset']['mathnet_levels']}): {len(mathnet)} problems")
    print(f"      OlympiadBench: {len(olympiad)} problems")

    problems = mix_datasets(
        mathnet,
        olympiad,
        total_samples=cfg["dataset"]["total_samples"],
        mathnet_ratio=cfg["dataset"]["mathnet_ratio"],
        seed=cfg["dataset"]["seed"],
    )
    print(f"      Mixed total: {len(problems)} problems")

    # ---- Load vLLM ----
    print(f"[2/4] Loading {cfg['model']['name']} via vLLM...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model"]["name"],
        tensor_parallel_size=cfg["model"]["tensor_parallel_size"],
        max_model_len=cfg["model"]["max_model_len"],
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
        dtype=cfg["model"].get("dtype", "bfloat16"),
    )

    sampling_params = SamplingParams(
        temperature=cfg["generation"]["temperature"],
        max_tokens=cfg["generation"]["max_tokens"],
        top_p=cfg["generation"]["top_p"],
        n=n_samples,
        stop=cfg["generation"].get("stop", []),
    )

    # ---- Generate ----
    print(f"[3/4] Generating {n_samples} sample(s) per problem ({len(problems)} problems)...")
    prompts = [build_prompt(p["problem"]) for p in problems]
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    # ---- Score ----
    print("[4/4] Scoring responses...")

    # per_source[src] = {"c_list": [...], "total": int}
    per_source: dict[str, dict] = {}
    results: list[dict] = []

    for problem, output in zip(problems, outputs):
        # Score every sample
        c_samples = 0
        first_predicted = None
        first_output = ""

        for idx, completion in enumerate(output.outputs):
            predicted = extract_boxed(completion.text)
            correct   = is_correct(predicted, problem["answer"])
            if correct:
                c_samples += 1
            if idx == 0:
                first_output    = completion.text
                first_predicted = predicted

        # per_at_k for this problem
        problem_pass_k = {
            f"pass@{k}": pass_at_k(n_samples, c_samples, k)
            for k in pass_k_cfg
            if k <= n_samples
        }

        src = problem["source"]
        if src not in per_source:
            per_source[src] = {"c_list": [], "total": 0}
        per_source[src]["c_list"].append(c_samples)
        per_source[src]["total"] += 1

        results.append(
            {
                **problem,
                "n_samples": n_samples,
                "c_samples": c_samples,
                "pass_at_k": problem_pass_k,
                # First sample kept for display in dashboard
                "model_output": first_output,
                "predicted_answer": first_predicted,
                # correct = any sample passed (useful for dashboard color coding)
                "correct": c_samples > 0,
            }
        )

    # ---- Aggregate pass@k ----
    all_c = [r["c_samples"] for r in results]

    overall_pass_k = aggregate_pass_at_k(all_c, n_samples, pass_k_cfg)

    source_stats: dict[str, dict] = {}
    for src, d in per_source.items():
        src_pass_k = aggregate_pass_at_k(d["c_list"], n_samples, pass_k_cfg)
        source_stats[src] = {
            "total": d["total"],
            "pass_at_k": {k: round(v, 4) for k, v in src_pass_k.items()},
        }

    stats = {
        "model": cfg["model"]["name"],
        "n_samples_per_problem": n_samples,
        "total_problems": len(results),
        "pass_at_k": {k: round(v, 4) for k, v in overall_pass_k.items()},
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_problem": round(elapsed / len(results), 3) if results else 0,
        "by_source": source_stats,
        "config": cfg,
    }

    _print_summary(stats)
    _save_results(cfg, stats, results)

    return stats


def _print_summary(stats: dict) -> None:
    n = stats["n_samples_per_problem"]
    print("\n" + "=" * 56)
    print(f"  Model    : {stats['model']}")
    print(f"  Samples  : {n} per problem  ({stats['total_problems']} problems)")
    for key, val in stats["pass_at_k"].items():
        print(f"  {key:<10}: {val:.1%}")
    print("  ──────────────────────────────────────────────────")
    for src, d in stats["by_source"].items():
        pk_str = "  ".join(f"{k}={v:.1%}" for k, v in d["pass_at_k"].items())
        print(f"  {src:<20}: {pk_str}  (n={d['total']})")
    print(f"  Elapsed  : {stats['elapsed_seconds']:.1f}s ({stats['seconds_per_problem']:.2f}s/problem)")
    print("=" * 56)


def _save_results(cfg: dict, stats: dict, results: list[dict]) -> None:
    out_dir = Path(cfg["eval"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Full results (model_output can be long — omit from full dump if huge)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Dashboard payload — first 200 samples, strip long model_output
    dashboard_samples = [
        {k: v for k, v in r.items() if k != "model_output"}
        | {"model_output": r["model_output"][:600]}
        for r in results[:200]
    ]
    dashboard_path = Path(cfg["eval"]["dashboard_output"])
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(
            {"stats": stats, "samples": dashboard_samples},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\nResults   → {out_dir}/")
    print(f"Dashboard → {dashboard_path}")
