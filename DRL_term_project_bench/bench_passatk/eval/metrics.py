"""
Metrics computation for Pass@K benchmark.

Implements:
- Pass@k (unbiased estimator from Codex paper)
- Pass@1 (explicit accuracy of first sample)
- Majority@k (self-consistency)
- Best-of-N (BoN) - accuracy when selecting the best answer
- Oracle Pass@k
- Wilson confidence intervals
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def compute_pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Compute unbiased Pass@k estimator using the Codex formula.
    
    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    
    This is the unbiased estimator from the Codex paper (Chen et al., 2021).
    It avoids the bias of simply using c/k, especially when k is large.
    
    To avoid numerical overflow with large numbers, we compute in log space.
    
    Args:
        n: Total number of samples (should equal K, the total samples drawn).
        c: Number of correct samples.
        k: The k in Pass@k (number of samples to consider).
        
    Returns:
        The unbiased Pass@k estimate.
    """
    if n - c < k:
        # If we have fewer incorrect samples than k, pass@k = 1
        return 1.0
    
    if c >= n:
        # All samples correct
        return 1.0
    
    if k == 0:
        return 0.0
    
    # Use log space to avoid overflow
    # C(n-c, k) / C(n, k) = (n-c)! * (n-k)! / ((n-c-k)! * n!)
    # = product from i=0 to k-1 of (n-c-i) / (n-i)
    
    # Compute log of the ratio
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)
    
    # pass@k = 1 - exp(log_ratio)
    return 1.0 - math.exp(log_ratio)


def compute_pass_at_k_values(n: int, c: int, k_values: List[int] = None) -> Dict[int, float]:
    """
    Compute Pass@k for multiple k values from a single sampling run.
    
    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k_values: List of k values to compute. Default is [1, 2, 4, 8, 16, 32, 64, 128, 256].
        
    Returns:
        Dictionary mapping k to Pass@k value.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    result = {}
    for k in k_values:
        if k > n:
            # Can't compute pass@k for k > n
            result[k] = float('nan')
        else:
            result[k] = compute_pass_at_k_unbiased(n, c, k)
    
    return result


def compute_majority_at_k(samples: List[Dict], k: int, key: str = "pred") -> Tuple[bool, str]:
    """
    Compute Majority@k (self-consistency voting).
    
    Takes the most common answer among the first k samples and checks if it's correct.
    
    Args:
        samples: List of sample dictionaries with 'pred' and 'is_correct' keys.
        k: Number of samples to consider.
        key: Key to use for voting (default 'pred').
        
    Returns:
        Tuple of (is_correct, majority_answer).
    """
    from collections import Counter
    
    if not samples or k <= 0:
        return False, ""
    
    # Take first k samples
    samples_k = samples[:k]
    
    # Count predictions
    predictions = [s.get(key, "") for s in samples_k if s.get(key) is not None]
    if not predictions:
        return False, ""
    
    counter = Counter(predictions)
    majority_answer, _ = counter.most_common(1)[0]
    
    # Check if majority answer is correct
    # Find a sample with this answer and check its correctness
    for s in samples_k:
        if s.get(key) == majority_answer:
            return s.get("is_correct", False), majority_answer
    
    return False, majority_answer


def compute_majority_at_k_values(
    samples: List[Dict],
    k_values: List[int] = None,
) -> Dict[int, Tuple[bool, str]]:
    """
    Compute Majority@k for multiple k values.
    
    Args:
        samples: List of sample dictionaries.
        k_values: List of k values. Default is [1, 4, 16, 64, 256].
        
    Returns:
        Dictionary mapping k to (is_correct, majority_answer).
    """
    if k_values is None:
        k_values = [1, 4, 16, 64, 256]
    
    result = {}
    for k in k_values:
        if k <= len(samples):
            result[k] = compute_majority_at_k(samples, k)
        else:
            result[k] = (False, "")
    
    return result


def compute_oracle(samples: List[Dict]) -> bool:
    """
    Compute Oracle Pass@k (whether any sample is correct).
    
    This is the upper bound for Pass@k - if any sample is correct,
    an oracle could select it.
    
    Args:
        samples: List of sample dictionaries with 'is_correct' key.
        
    Returns:
        True if any sample is correct.
    """
    return any(s.get("is_correct", False) for s in samples)


def compute_pass_at_1(samples: List[Dict]) -> Tuple[bool, str]:
    """
    Compute Pass@1 (accuracy of the first sample).
    
    This is equivalent to the standard accuracy metric.
    
    Args:
        samples: List of sample dictionaries with 'pred' and 'is_correct' keys.
        
    Returns:
        Tuple of (is_correct, prediction).
    """
    if not samples:
        return False, ""
    
    first_sample = samples[0]
    return first_sample.get("is_correct", False), first_sample.get("pred", "")


def compute_best_of_n(samples: List[Dict], n: int = None) -> Tuple[bool, str, int]:
    """
    Compute Best-of-N (BoN) accuracy.
    
    BoN assumes we can select the best answer from N samples.
    This is equivalent to Oracle@N but returns the best answer.
    
    Args:
        samples: List of sample dictionaries with 'pred' and 'is_correct' keys.
        n: Number of samples to consider. If None, use all samples.
        
    Returns:
        Tuple of (is_correct, best_answer, index_of_best).
    """
    if not samples:
        return False, "", -1
    
    samples_to_consider = samples[:n] if n else samples
    
    # Find the first correct sample
    for idx, sample in enumerate(samples_to_consider):
        if sample.get("is_correct", False):
            return True, sample.get("pred", ""), idx
    
    # If no correct sample, return the first one
    return False, samples_to_consider[0].get("pred", ""), 0


def compute_best_of_n_values(
    samples: List[Dict],
    n_values: List[int] = None,
) -> Dict[int, Tuple[bool, str]]:
    """
    Compute Best-of-N for multiple N values.
    
    Args:
        samples: List of sample dictionaries.
        n_values: List of N values. Default is [1, 4, 16, 64, 256].
        
    Returns:
        Dictionary mapping N to (is_correct, best_answer).
    """
    if n_values is None:
        n_values = [1, 4, 16, 64, 256]
    
    result = {}
    for n in n_values:
        if n <= len(samples):
            is_correct, best_answer, _ = compute_best_of_n(samples, n)
            result[n] = (is_correct, best_answer)
        else:
            result[n] = (False, "")
    
    return result


def wilson_confidence_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson confidence interval for a proportion.
    
    More accurate than normal approximation for extreme proportions.
    
    Args:
        successes: Number of successes.
        trials: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if trials == 0:
        return 0.0, 1.0
    
    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return lower, upper


def compute_all_metrics(
    samples: List[Dict],
    k_values: List[int] = None,
    maj_k_values: List[int] = None,
    bon_values: List[int] = None,
) -> Dict:
    """
    Compute all metrics for a single problem.
    
    Args:
        samples: List of sample dictionaries with 'pred' and 'is_correct' keys.
        k_values: k values for Pass@k. Default [1, 2, 4, 8, 16, 32, 64, 128, 256].
        maj_k_values: k values for Majority@k. Default [1, 4, 16, 64, 256].
        bon_values: N values for Best-of-N. Default [1, 4, 16, 64, 256].
        
    Returns:
        Dictionary with:
        - n: total samples
        - c: correct samples
        - pass@k values
        - pass@1_explicit: explicit accuracy of first sample
        - maj@k values
        - bon@n values (Best-of-N)
        - oracle
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if maj_k_values is None:
        maj_k_values = [1, 4, 16, 64, 256]
    if bon_values is None:
        bon_values = [1, 4, 16, 64, 256]
    
    n = len(samples)
    c = sum(1 for s in samples if s.get("is_correct", False))
    
    # Pass@k values (unbiased estimator)
    pass_at_k = compute_pass_at_k_values(n, c, k_values)
    
    # Pass@1 explicit (first sample accuracy)
    pass_at_1_explicit, first_pred = compute_pass_at_1(samples)
    
    # Majority@k values
    maj_at_k = compute_majority_at_k_values(samples, maj_k_values)
    
    # Best-of-N values
    bon_at_n = compute_best_of_n_values(samples, bon_values)
    
    # Oracle
    oracle = compute_oracle(samples)
    
    return {
        "n": n,
        "c": c,
        **{f"pass@{k}": v for k, v in pass_at_k.items()},
        "pass@1_explicit": pass_at_1_explicit,
        "first_pred": first_pred,
        **{f"maj@{k}": correct for k, (correct, _) in maj_at_k.items()},
        **{f"bon@{n}": correct for n, (correct, _) in bon_at_n.items()},
        "oracle": oracle,
    }


def aggregate_metrics(
    results: List[Dict],
    k_values: List[int] = None,
    maj_k_values: List[int] = None,
    bon_values: List[int] = None,
    confidence: float = 0.95,
) -> Dict:
    """
    Aggregate metrics across all problems in a dataset.
    
    Args:
        results: List of result dictionaries (one per problem).
        k_values: k values for Pass@k.
        maj_k_values: k values for Majority@k.
        bon_values: N values for Best-of-N.
        confidence: Confidence level for Wilson CI.
        
    Returns:
        Dictionary with aggregated metrics and confidence intervals.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if maj_k_values is None:
        maj_k_values = [1, 4, 16, 64, 256]
    if bon_values is None:
        bon_values = [1, 4, 16, 64, 256]
    
    n_problems = len(results)
    
    # Aggregate Pass@k
    pass_at_k_agg = {}
    for k in k_values:
        pass_values = []
        for r in results:
            per_problem = r.get("per_problem", {})
            pass_k_key = f"pass@{k}"
            if pass_k_key in per_problem:
                pass_values.append(per_problem[pass_k_key])
        
        if pass_values:
            mean_pass = np.mean(pass_values)
            n = len(pass_values)
            std_err = np.std(pass_values, ddof=1) / np.sqrt(n)
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            ci_low = mean_pass - z * std_err
            ci_high = mean_pass + z * std_err
            
            pass_at_k_agg[k] = {
                "mean": mean_pass,
                "ci_low": max(0.0, ci_low),
                "ci_high": min(1.0, ci_high),
                "std": np.std(pass_values, ddof=1),
            }
    
    # Aggregate Pass@1 explicit (first sample accuracy)
    pass_at_1_correct = sum(
        1 for r in results 
        if r.get("per_problem", {}).get("pass@1_explicit", False)
    )
    ci_low, ci_high = wilson_confidence_interval(pass_at_1_correct, n_problems, confidence)
    pass_at_1_agg = {
        "accuracy": pass_at_1_correct / n_problems if n_problems > 0 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    
    # Aggregate Majority@k
    maj_at_k_agg = {}
    for k in maj_k_values:
        maj_correct = 0
        for r in results:
            per_problem = r.get("per_problem", {})
            maj_k_key = f"maj@{k}"
            if per_problem.get(maj_k_key, False):
                maj_correct += 1
        
        ci_low, ci_high = wilson_confidence_interval(maj_correct, n_problems, confidence)
        maj_at_k_agg[k] = {
            "accuracy": maj_correct / n_problems if n_problems > 0 else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    
    # Aggregate Best-of-N
    bon_at_n_agg = {}
    for n in bon_values:
        bon_correct = 0
        for r in results:
            per_problem = r.get("per_problem", {})
            bon_n_key = f"bon@{n}"
            if per_problem.get(bon_n_key, False):
                bon_correct += 1
        
        ci_low, ci_high = wilson_confidence_interval(bon_correct, n_problems, confidence)
        bon_at_n_agg[n] = {
            "accuracy": bon_correct / n_problems if n_problems > 0 else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    
    # Aggregate Oracle
    oracle_count = sum(1 for r in results if r.get("per_problem", {}).get("oracle", False))
    ci_low, ci_high = wilson_confidence_interval(oracle_count, n_problems, confidence)
    oracle_agg = {
        "accuracy": oracle_count / n_problems if n_problems > 0 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    
    return {
        "n_problems": n_problems,
        "pass@k": pass_at_k_agg,
        "pass@1_explicit": pass_at_1_agg,
        "maj@k": maj_at_k_agg,
        "bon@n": bon_at_n_agg,
        "oracle": oracle_agg,
    }