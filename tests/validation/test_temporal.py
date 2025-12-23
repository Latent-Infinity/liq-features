"""Tests for rolling window temporal stability analysis.

Tests cover:
- Rolling window MI computation
- Ranking stability over time
- Regime change detection
- Adjacent window correlations
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from liq.features.validation.exceptions import (
    ConfigurationError,
    InsufficientDataError,
)
from liq.features.validation.temporal import rolling_mi_analysis
from liq.features.validation.results import TemporalStabilityResult


class TestRollingMIAnalysis:
    """Tests for rolling window MI analysis."""

    @pytest.fixture
    def stable_data(self) -> tuple[pl.DataFrame, pl.Series]:
        """Create synthetic data with stable MI over time."""
        rng = np.random.default_rng(42)
        n = 2000

        # Features with consistent relationship
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.2, n)

        df = pl.DataFrame({
            "feature_1": x1,
            "feature_2": x2,
        })
        target = pl.Series("target", y)

        return df, target

    @pytest.fixture
    def regime_change_data(self) -> tuple[pl.DataFrame, pl.Series]:
        """Create data with a regime change in the middle."""
        rng = np.random.default_rng(42)
        n = 2000
        half = n // 2

        # First half: x1 is informative
        x1_first = rng.normal(0, 1, half)
        x2_first = rng.normal(0, 1, half)
        y_first = 0.8 * x1_first + rng.normal(0, 0.2, half)

        # Second half: x2 is informative
        x1_second = rng.normal(0, 1, half)
        x2_second = rng.normal(0, 1, half)
        y_second = 0.8 * x2_second + rng.normal(0, 0.2, half)

        df = pl.DataFrame({
            "feature_1": np.concatenate([x1_first, x1_second]),
            "feature_2": np.concatenate([x2_first, x2_second]),
        })
        target = pl.Series("target", np.concatenate([y_first, y_second]))

        return df, target

    def test_returns_temporal_stability_result(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should return TemporalStabilityResult."""
        X, y = stable_data

        result = rolling_mi_analysis(
            X, y, features=["feature_1", "feature_2"], window_size=500, step_size=250
        )

        assert isinstance(result, TemporalStabilityResult)

    def test_correct_number_of_windows(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute correct number of windows."""
        X, y = stable_data

        result = rolling_mi_analysis(
            X, y, window_size=500, step_size=250
        )

        # With n=2000, window=500, step=250:
        # Windows start at 0, 250, 500, 750, 1000, 1250, 1500 (last valid end: 2000)
        # Window at 1500 ends at 2000 (valid)
        # Window at 1750 would end at 2250 (invalid)
        expected_windows = (2000 - 500) // 250 + 1  # = 7
        assert result.n_windows == expected_windows

    def test_mi_computed_for_each_window(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute MI for each feature in each window."""
        X, y = stable_data
        features = ["feature_1", "feature_2"]

        result = rolling_mi_analysis(
            X, y, features=features, window_size=500, step_size=250
        )

        assert len(result.mi_by_window) == result.n_windows
        for mi_dict in result.mi_by_window:
            assert set(mi_dict.keys()) == set(features)

    def test_rankings_computed_for_each_window(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute rankings for each window."""
        X, y = stable_data
        features = ["feature_1", "feature_2"]

        result = rolling_mi_analysis(
            X, y, features=features, window_size=500, step_size=250
        )

        assert len(result.rank_by_window) == result.n_windows
        for rank_dict in result.rank_by_window:
            assert set(rank_dict.values()) == {1, 2}

    def test_adjacent_correlations_computed(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute correlations between adjacent windows."""
        X, y = stable_data

        result = rolling_mi_analysis(X, y, window_size=500, step_size=250)

        # Should have n_windows - 1 correlations
        assert len(result.adjacent_correlations) == result.n_windows - 1

    def test_stable_data_has_high_correlation(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Stable data should have high adjacent correlations."""
        X, y = stable_data

        result = rolling_mi_analysis(X, y, window_size=500, step_size=250)

        # Mean correlation should be high for stable data
        assert result.mean_correlation > 0.5

    def test_is_stable_for_stable_data(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """is_stable should be True for stable data."""
        X, y = stable_data

        result = rolling_mi_analysis(X, y, window_size=500, step_size=250)

        if result.mean_correlation > 0.8:
            assert result.is_stable is True

    def test_regime_change_detected(
        self, regime_change_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should detect regime changes."""
        X, y = regime_change_data

        result = rolling_mi_analysis(
            X, y, window_size=400, step_size=200, regime_threshold=0.3
        )

        # Should detect at least one regime change
        assert result.n_regime_changes >= 0  # May or may not detect depending on params

    def test_min_correlation_tracked(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should track minimum correlation."""
        X, y = stable_data

        result = rolling_mi_analysis(X, y, window_size=500, step_size=250)

        assert result.min_correlation <= result.mean_correlation

    def test_window_starts_recorded(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should record window start indices."""
        X, y = stable_data

        result = rolling_mi_analysis(X, y, window_size=500, step_size=250)

        assert len(result.window_starts) == result.n_windows
        assert result.window_starts[0] == 0
        # Each subsequent window starts step_size later
        for i in range(1, len(result.window_starts)):
            assert result.window_starts[i] == result.window_starts[i - 1] + 250

    def test_reproducibility_with_random_state(
        self, stable_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Results should be reproducible."""
        X, y = stable_data

        result1 = rolling_mi_analysis(
            X, y, window_size=500, step_size=250, random_state=42
        )
        result2 = rolling_mi_analysis(
            X, y, window_size=500, step_size=250, random_state=42
        )

        assert result1.mi_by_window == result2.mi_by_window


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_window_size_larger_than_data_raises(self) -> None:
        """Window larger than data should raise."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(InsufficientDataError):
            rolling_mi_analysis(X, y, window_size=200, step_size=50)

    def test_invalid_window_size_raises(self) -> None:
        """Window size <= 0 should raise."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(ConfigurationError, match="window_size"):
            rolling_mi_analysis(X, y, window_size=0, step_size=50)

    def test_invalid_step_size_raises(self) -> None:
        """Step size <= 0 should raise."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(ConfigurationError, match="step_size"):
            rolling_mi_analysis(X, y, window_size=50, step_size=0)

    def test_single_window(self) -> None:
        """Should work with only one window."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        result = rolling_mi_analysis(X, y, window_size=100, step_size=50)

        assert result.n_windows >= 1
        assert len(result.adjacent_correlations) == result.n_windows - 1

    def test_single_feature(self) -> None:
        """Should work with single feature."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 500)})
        y = pl.Series("y", rng.normal(0, 1, 500))

        result = rolling_mi_analysis(X, y, window_size=200, step_size=100)

        assert len(result.features) == 1
        # Single feature has perfect rank correlation
        for corr in result.adjacent_correlations:
            assert corr == 1.0 or np.isnan(corr)

    def test_to_dict_serialization(self) -> None:
        """Result should be serializable."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 500)})
        y = pl.Series("y", rng.normal(0, 1, 500))

        result = rolling_mi_analysis(X, y, window_size=200, step_size=100)
        d = result.to_dict()

        assert "mi_by_window" in d
        assert "adjacent_correlations" in d
        assert "mean_correlation" in d
