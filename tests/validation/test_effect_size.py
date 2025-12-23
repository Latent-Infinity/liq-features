"""Tests for Cohen's d effect size calculation.

Tests cover:
- Small, medium, large effect sizes
- Equal and unequal group sizes
- Equal and unequal variances
- Known analytical solutions
- Confidence interval calculation
- Edge cases (zero variance, identical means)
"""

from __future__ import annotations

import numpy as np
import pytest

from liq.features.validation.effect_size import (
    cohens_d,
    cohens_d_ci,
    pooled_std,
)
from liq.features.validation.results import EffectSizeResult


class TestPooledStd:
    """Tests for pooled standard deviation calculation."""

    def test_equal_variance_equal_size(self) -> None:
        """Pooled std with equal variance and equal sizes."""
        # Two groups with known variance
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)
        group2 = rng.normal(0, 1, 100)

        pstd = pooled_std(group1, group2)
        # Should be close to 1.0 (the true std)
        assert 0.8 < pstd < 1.2

    def test_equal_variance_unequal_size(self) -> None:
        """Pooled std with equal variance but unequal sizes."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 2, 200)  # std = 2
        group2 = rng.normal(0, 2, 50)  # std = 2

        pstd = pooled_std(group1, group2)
        # Should be close to 2.0
        assert 1.6 < pstd < 2.4

    def test_analytical_example(self) -> None:
        """Verify with known analytical result."""
        # Group 1: n=3, values [1, 2, 3], var = 1.0, n-1=2, ss=2
        # Group 2: n=3, values [4, 5, 6], var = 1.0, n-1=2, ss=2
        # Pooled var = (2 + 2) / (2 + 2) = 1.0
        # Pooled std = 1.0
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0])

        pstd = pooled_std(group1, group2)
        assert abs(pstd - 1.0) < 0.01


class TestCohensD:
    """Tests for Cohen's d calculation."""

    def test_small_effect(self) -> None:
        """Test small effect size (d ≈ 0.2)."""
        rng = np.random.default_rng(42)
        # Groups with means 0.2 std apart
        group1 = rng.normal(0, 1, 500)
        group2 = rng.normal(0.2, 1, 500)

        d = cohens_d(group1, group2)
        assert -0.4 < d < 0.0  # Should be around -0.2

    def test_medium_effect(self) -> None:
        """Test medium effect size (d ≈ 0.5)."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 500)
        group2 = rng.normal(0.5, 1, 500)

        d = cohens_d(group1, group2)
        assert -0.7 < d < -0.3  # Should be around -0.5

    def test_large_effect(self) -> None:
        """Test large effect size (d ≈ 0.8)."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 500)
        group2 = rng.normal(0.8, 1, 500)

        d = cohens_d(group1, group2)
        assert -1.0 < d < -0.6  # Should be around -0.8

    def test_no_effect(self) -> None:
        """Test no effect (d ≈ 0)."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 500)
        group2 = rng.normal(0, 1, 500)

        d = cohens_d(group1, group2)
        assert -0.2 < d < 0.2  # Should be close to 0

    def test_sign_convention(self) -> None:
        """Test that d is negative when group1 < group2."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0])

        d = cohens_d(group1, group2)
        assert d < 0  # group1 mean < group2 mean, so d is negative

    def test_symmetry(self) -> None:
        """Test that swapping groups flips the sign."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)
        group2 = rng.normal(0.5, 1, 100)

        d1 = cohens_d(group1, group2)
        d2 = cohens_d(group2, group1)

        assert abs(d1 + d2) < 0.01  # Should be opposite signs

    def test_unequal_variance(self) -> None:
        """Test with unequal variances (uses Welch's approach)."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)  # std = 1
        group2 = rng.normal(0.5, 2, 100)  # std = 2

        # With unequal variances, pooled std is larger
        d = cohens_d(group1, group2)
        # Effect should be smaller due to larger pooled std
        assert -0.5 < d < 0.0

    def test_unequal_size(self) -> None:
        """Test with unequal group sizes."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 200)
        group2 = rng.normal(0.5, 1, 50)

        d = cohens_d(group1, group2)
        assert -0.7 < d < -0.3  # Should still be around -0.5

    def test_analytical_known_value(self) -> None:
        """Test with analytically known result."""
        # Group 1: mean=2, std=1
        # Group 2: mean=3, std=1
        # d = (2-3)/1 = -1.0
        group1 = np.array([1.0, 2.0, 3.0])  # mean=2
        group2 = np.array([2.0, 3.0, 4.0])  # mean=3

        d = cohens_d(group1, group2)
        assert abs(d - (-1.0)) < 0.01


class TestCohensD_CI:
    """Tests for Cohen's d confidence interval calculation."""

    def test_ci_contains_point_estimate(self) -> None:
        """CI should contain the point estimate."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)
        group2 = rng.normal(0.5, 1, 100)

        result = cohens_d_ci(group1, group2, n_bootstrap=500)

        assert result.ci_lower <= result.cohens_d <= result.ci_upper

    def test_ci_interpretation(self) -> None:
        """Result should include correct interpretation."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)
        group2 = rng.normal(0.5, 1, 100)

        result = cohens_d_ci(group1, group2, n_bootstrap=500)

        # Medium effect expected
        assert result.interpretation in ["small", "medium"]

    def test_ci_width_scales_with_n(self) -> None:
        """CI should be narrower with larger samples."""
        rng = np.random.default_rng(42)

        # Small sample
        g1_small = rng.normal(0, 1, 30)
        g2_small = rng.normal(0.5, 1, 30)
        result_small = cohens_d_ci(g1_small, g2_small, n_bootstrap=500)
        width_small = result_small.ci_upper - result_small.ci_lower

        # Large sample
        g1_large = rng.normal(0, 1, 300)
        g2_large = rng.normal(0.5, 1, 300)
        result_large = cohens_d_ci(g1_large, g2_large, n_bootstrap=500)
        width_large = result_large.ci_upper - result_large.ci_lower

        # Larger sample should have narrower CI
        assert width_large < width_small

    def test_ci_confidence_level(self) -> None:
        """CI width should scale with confidence level."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)
        group2 = rng.normal(0.5, 1, 100)

        result_90 = cohens_d_ci(group1, group2, n_bootstrap=500, confidence_level=0.90)
        result_99 = cohens_d_ci(group1, group2, n_bootstrap=500, confidence_level=0.99)

        width_90 = result_90.ci_upper - result_90.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower

        # 99% CI should be wider than 90% CI
        assert width_99 > width_90

    def test_result_structure(self) -> None:
        """Result should have all required fields."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 50)
        group2 = rng.normal(0.5, 1, 50)

        result = cohens_d_ci(group1, group2, n_bootstrap=100)

        assert isinstance(result, EffectSizeResult)
        assert result.n_group1 == 50
        assert result.n_group2 == 50
        assert result.n_bootstrap == 100
        assert result.confidence_level == 0.95
        assert result.pooled_std > 0
        assert result.mean_diff != 0

    def test_reproducibility_with_seed(self) -> None:
        """Results should be reproducible with same random_state."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 50)
        group2 = rng.normal(0.5, 1, 50)

        result1 = cohens_d_ci(group1, group2, n_bootstrap=100, random_state=123)
        result2 = cohens_d_ci(group1, group2, n_bootstrap=100, random_state=123)

        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_identical_groups(self) -> None:
        """Identical groups should give d ≈ 0."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        d = cohens_d(group, group.copy())
        assert abs(d) < 0.01

    def test_very_small_effect(self) -> None:
        """Very small differences should give d close to 0."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([1.01, 2.01, 3.01])

        d = cohens_d(group1, group2)
        assert abs(d) < 0.05

    def test_minimum_sample_size(self) -> None:
        """Should work with minimum sample size (n=2 per group)."""
        group1 = np.array([1.0, 2.0])
        group2 = np.array([3.0, 4.0])

        d = cohens_d(group1, group2)
        assert d < 0  # group1 < group2

    def test_single_value_per_group_raises(self) -> None:
        """Single value per group should raise (can't compute std)."""
        group1 = np.array([1.0])
        group2 = np.array([2.0])

        with pytest.raises(ValueError, match="sample size"):
            cohens_d(group1, group2)

    def test_zero_variance_raises(self) -> None:
        """Zero variance in both groups should raise."""
        group1 = np.array([1.0, 1.0, 1.0])
        group2 = np.array([2.0, 2.0, 2.0])

        with pytest.raises(ValueError, match="variance"):
            cohens_d(group1, group2)
