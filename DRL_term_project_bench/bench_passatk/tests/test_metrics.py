"""
Unit tests for Pass@K metrics.

Tests verify that the Pass@K implementation matches the Codex paper formula.
"""

import math
import pytest

from bench_passatk.eval.metrics import (
    compute_pass_at_k_unbiased,
    compute_pass_at_k_values,
    compute_majority_at_k,
    compute_oracle,
    compute_all_metrics,
    wilson_confidence_interval,
)


class TestPassAtKUnbiased:
    """Tests for the unbiased Pass@K estimator."""
    
    def test_codex_paper_example(self):
        """
        Test against the Codex paper example.
        
        From Chen et al. (2021), the unbiased estimator is:
        pass@k = 1 - C(n-c, k) / C(n, k)
        
        For n=200, c=100, k=10:
        pass@10 = 1 - C(100, 10) / C(200, 10)
        
        This should give approximately 0.993 (from the paper).
        """
        n = 200
        c = 100
        k = 10
        
        result = compute_pass_at_k_unbiased(n, c, k)
        
        # Expected value from Codex paper
        # pass@10 ≈ 0.993 for n=200, c=100, k=10
        expected = 1 - math.comb(n - c, k) / math.comb(n, k)
        
        assert abs(result - expected) < 1e-10
        assert result > 0.99  # Should be very high
    
    def test_all_correct(self):
        """Test when all samples are correct."""
        n = 100
        c = 100  # All correct
        k = 10
        
        result = compute_pass_at_k_unbiased(n, c, k)
        assert result == 1.0
    
    def test_none_correct(self):
        """Test when no samples are correct."""
        n = 100
        c = 0  # None correct
        k = 10
        
        result = compute_pass_at_k_unbiased(n, c, k)
        assert result == 0.0
    
    def test_k_greater_than_incorrect(self):
        """Test when k > (n - c), meaning pass@k = 1."""
        n = 100
        c = 95  # Only 5 incorrect
        k = 10
        
        result = compute_pass_at_k_unbiased(n, c, k)
        assert result == 1.0  # Must have at least one correct in k samples
    
    def test_small_sample(self):
        """Test with small sample sizes."""
        n = 10
        c = 5
        k = 2
        
        result = compute_pass_at_k_unbiased(n, c, k)
        
        # Manual calculation:
        # pass@2 = 1 - C(5, 2) / C(10, 2)
        #        = 1 - 10 / 45
        #        = 1 - 0.222...
        #        = 0.777...
        expected = 1 - math.comb(5, 2) / math.comb(10, 2)
        
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.7777) < 0.01
    
    def test_large_k(self):
        """Test with large k values."""
        n = 256
        c = 64
        k = 128
        
        result = compute_pass_at_k_unbiased(n, c, k)
        
        # Should not overflow
        assert 0 <= result <= 1
        
        # Manual calculation
        expected = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert abs(result - expected) < 1e-10
    
    def test_numerical_stability(self):
        """Test numerical stability with large numbers."""
        # This tests that we don't get overflow with large n
        n = 1000
        c = 500
        k = 100
        
        result = compute_pass_at_k_unbiased(n, c, k)
        
        # Should not raise overflow error
        assert 0 <= result <= 1
    
    def test_k_equals_n(self):
        """Test when k = n (all samples considered)."""
        n = 100
        c = 50
        k = 100
        
        result = compute_pass_at_k_unbiased(n, c, k)
        
        # When k = n, pass@k = 1 if c > 0, else 0
        # But our formula handles this correctly
        # C(50, 100) = 0, so we get 1 - 0 = 1
        assert result == 1.0


class TestPassAtKValues:
    """Tests for computing multiple Pass@K values."""
    
    def test_default_k_values(self):
        """Test with default k values."""
        n = 256
        c = 64
        
        result = compute_pass_at_k_values(n, c)
        
        # Should have all default k values
        expected_ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        assert set(result.keys()) == set(expected_ks)
        
        # Pass@k should be monotonically increasing with k
        for i in range(len(expected_ks) - 1):
            k1 = expected_ks[i]
            k2 = expected_ks[i + 1]
            assert result[k1] <= result[k2] + 1e-10  # Allow small numerical error
    
    def test_custom_k_values(self):
        """Test with custom k values."""
        n = 100
        c = 30
        k_values = [1, 5, 10, 50]
        
        result = compute_pass_at_k_values(n, c, k_values)
        
        assert set(result.keys()) == set(k_values)
        
        # Verify each value
        for k in k_values:
            expected = compute_pass_at_k_unbiased(n, c, k)
            assert abs(result[k] - expected) < 1e-10
    
    def test_k_greater_than_n(self):
        """Test when some k values are greater than n."""
        n = 50
        c = 10
        k_values = [1, 10, 50, 100]
        
        result = compute_pass_at_k_values(n, c, k_values)
        
        # k=100 > n=50 should return nan
        assert math.isnan(result[100])
        
        # Other values should be valid
        assert result[1] >= 0
        assert result[10] >= 0
        assert result[50] >= 0


class TestMajorityAtK:
    """Tests for Majority@K (self-consistency)."""
    
    def test_unanimous_answer(self):
        """Test when all samples have the same answer."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
        ]
        
        is_correct, majority = compute_majority_at_k(samples, 3)
        
        assert is_correct is True
        assert majority == "42"
    
    def test_majority_answer(self):
        """Test when majority answer is correct."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
            {"pred": "43", "is_correct": False},
        ]
        
        is_correct, majority = compute_majority_at_k(samples, 3)
        
        assert is_correct is True
        assert majority == "42"
    
    def test_minority_correct(self):
        """Test when minority answer is correct."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "43", "is_correct": False},
            {"pred": "43", "is_correct": False},
        ]
        
        is_correct, majority = compute_majority_at_k(samples, 3)
        
        assert is_correct is False
        assert majority == "43"
    
    def test_tie_breaking(self):
        """Test tie-breaking behavior."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "43", "is_correct": False},
        ]
        
        is_correct, majority = compute_majority_at_k(samples, 2)
        
        # First encountered answer wins in case of tie
        assert majority in ["42", "43"]
    
    def test_empty_samples(self):
        """Test with empty samples."""
        is_correct, majority = compute_majority_at_k([], 1)
        
        assert is_correct is False
        assert majority == ""
    
    def test_k_less_than_samples(self):
        """Test when k is less than total samples."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
            {"pred": "43", "is_correct": False},
            {"pred": "44", "is_correct": False},
        ]
        
        is_correct, majority = compute_majority_at_k(samples, 2)
        
        # Only consider first 2 samples
        assert is_correct is True
        assert majority == "42"


class TestOracle:
    """Tests for Oracle Pass@K."""
    
    def test_any_correct(self):
        """Test when at least one sample is correct."""
        samples = [
            {"pred": "42", "is_correct": False},
            {"pred": "43", "is_correct": False},
            {"pred": "44", "is_correct": True},
        ]
        
        result = compute_oracle(samples)
        assert result is True
    
    def test_none_correct(self):
        """Test when no samples are correct."""
        samples = [
            {"pred": "42", "is_correct": False},
            {"pred": "43", "is_correct": False},
            {"pred": "44", "is_correct": False},
        ]
        
        result = compute_oracle(samples)
        assert result is False
    
    def test_all_correct(self):
        """Test when all samples are correct."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
        ]
        
        result = compute_oracle(samples)
        assert result is True
    
    def test_empty_samples(self):
        """Test with empty samples."""
        result = compute_oracle([])
        assert result is False


class TestWilsonConfidenceInterval:
    """Tests for Wilson confidence interval."""
    
    def test_perfect_accuracy(self):
        """Test Wilson CI with 100% accuracy."""
        lower, upper = wilson_confidence_interval(100, 100, 0.95)
        
        # Lower bound should be high but not 1
        assert lower > 0.9
        assert upper == 1.0
    
    def test_zero_accuracy(self):
        """Test Wilson CI with 0% accuracy."""
        lower, upper = wilson_confidence_interval(0, 100, 0.95)
        
        assert lower == 0.0
        assert upper < 0.1
    
    def test_fifty_percent(self):
        """Test Wilson CI with 50% accuracy."""
        lower, upper = wilson_confidence_interval(50, 100, 0.95)
        
        # Should be roughly [0.40, 0.60]
        assert 0.35 < lower < 0.45
        assert 0.55 < upper < 0.65
    
    def test_small_sample(self):
        """Test Wilson CI with small sample."""
        lower, upper = wilson_confidence_interval(1, 2, 0.95)
        
        # Wide interval for small sample
        assert lower < 0.5
        assert upper > 0.5
    
    def test_zero_trials(self):
        """Test Wilson CI with zero trials."""
        lower, upper = wilson_confidence_interval(0, 0, 0.95)
        
        assert lower == 0.0
        assert upper == 1.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""
    
    def test_basic_metrics(self):
        """Test basic metrics computation."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
            {"pred": "43", "is_correct": False},
            {"pred": "44", "is_correct": False},
        ]
        
        metrics = compute_all_metrics(samples)
        
        assert metrics["n"] == 4
        assert metrics["c"] == 2
        assert "pass@1" in metrics
        assert "pass@2" in metrics
        assert "maj@1" in metrics
        assert "oracle" in metrics
    
    def test_empty_samples(self):
        """Test with empty samples."""
        metrics = compute_all_metrics([])
        
        assert metrics["n"] == 0
        assert metrics["c"] == 0
        assert metrics["oracle"] is False
    
    def test_all_correct(self):
        """Test with all correct samples."""
        samples = [
            {"pred": "42", "is_correct": True},
            {"pred": "42", "is_correct": True},
        ]
        
        metrics = compute_all_metrics(samples)
        
        assert metrics["c"] == 2
        assert metrics["pass@1"] == 1.0
        assert metrics["pass@2"] == 1.0
        assert metrics["oracle"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])