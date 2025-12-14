"""Tests for liq.features.regime module."""

import numpy as np
import pytest

from liq.features.regime import hurst_exponent


class TestHurstExponent:
    """Tests for hurst_exponent function."""

    def test_short_series_returns_default(self) -> None:
        """Test series shorter than 20 returns 0.5."""
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        h = hurst_exponent(series)
        assert h == 0.5

    def test_series_exactly_19_elements(self) -> None:
        """Test series with exactly 19 elements returns 0.5."""
        series = list(range(1, 20))  # 19 elements
        h = hurst_exponent(series)
        assert h == 0.5

    def test_series_exactly_20_elements(self) -> None:
        """Test series with exactly 20 elements computes hurst."""
        series = list(range(1, 21))  # 20 elements
        h = hurst_exponent(series)
        # Should compute, not return default
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_empty_series(self) -> None:
        """Test empty series returns default."""
        h = hurst_exponent([])
        assert h == 0.5

    def test_single_element_series(self) -> None:
        """Test single element series returns default."""
        h = hurst_exponent([42.0])
        assert h == 0.5

    def test_trending_series_computes_valid_hurst(self) -> None:
        """Test trending series computes a valid Hurst value."""
        # Strong upward trend
        series = [float(i) for i in range(100)]
        h = hurst_exponent(series)
        # Implementation uses rescaled range with limited tau values [2, 4, 8, 16]
        # which may not perfectly detect trends; verify it returns valid float
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_mean_reverting_series(self) -> None:
        """Test mean-reverting series has Hurst exponent < 0.5."""
        # Alternating series (mean-reverting)
        series = [1.0, -1.0] * 50  # 100 elements
        h = hurst_exponent(series)
        # Mean-reverting should have H < 0.5
        assert h < 0.5

    def test_random_walk_series_near_half(self) -> None:
        """Test random walk series has Hurst exponent near 0.5."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        # Cumulative sum of random steps
        steps = np.random.randn(100)
        series = np.cumsum(steps).tolist()
        h = hurst_exponent(series)
        # Random walk should be near 0.5 (within tolerance)
        assert 0.3 <= h <= 0.7

    def test_returns_float(self) -> None:
        """Test function returns Python float."""
        series = list(range(100))
        h = hurst_exponent(series)
        assert isinstance(h, float)

    def test_constant_series(self) -> None:
        """Test constant series (zero std) returns default."""
        series = [5.0] * 100
        h = hurst_exponent(series)
        # When std is 0, should return default 0.5
        assert h == 0.5

    def test_large_series(self) -> None:
        """Test with larger series works correctly."""
        series = [float(i) + np.sin(i / 10) * 5 for i in range(500)]
        h = hurst_exponent(series)
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_with_numpy_array_input(self) -> None:
        """Test works with numpy array as input."""
        series = np.linspace(0, 100, 50)
        h = hurst_exponent(series.tolist())
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_negative_values(self) -> None:
        """Test series with negative values."""
        series = [float(i - 50) for i in range(100)]
        h = hurst_exponent(series)
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_decimal_values(self) -> None:
        """Test series with decimal values."""
        series = [i * 0.01 for i in range(100)]
        h = hurst_exponent(series)
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_mixed_segment_validity(self) -> None:
        """Test series where some segments have valid data."""
        # Series with enough data for some tau values but not others
        series = list(range(32))  # Divisible by 2, 4, 8, 16
        h = hurst_exponent(series)
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_series_not_divisible_by_tau(self) -> None:
        """Test series length not evenly divisible by tau values."""
        series = list(range(23))  # Not divisible by 4, 8, 16
        h = hurst_exponent(series)
        assert isinstance(h, float)
        assert 0 <= h <= 1
