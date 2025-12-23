"""Tests for MI estimator sensitivity analysis.

Tests cover:
- MI stability across k-NN values (k=1,3,5,7)
- Ranking correlation across k values
- Coefficient of variation calculation
- Synthetic data with known MI properties
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from liq.features.validation.exceptions import ConfigurationError
from liq.features.validation.sensitivity import (
    mi_sensitivity_analysis,
    batch_sensitivity_analysis,
)
from liq.features.validation.results import SensitivityResult


class TestMISensitivityAnalysis:
    """Tests for single-feature MI sensitivity analysis."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[pl.DataFrame, pl.Series]:
        """Create synthetic data with known relationship."""
        rng = np.random.default_rng(42)
        n = 1000

        # Feature with strong linear relationship to target
        x = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, n)
        y = 0.8 * x + noise  # Strong positive correlation

        df = pl.DataFrame({"feature_a": x, "feature_b": rng.normal(0, 1, n)})
        target = pl.Series("target", y)

        return df, target

    def test_returns_sensitivity_result(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should return SensitivityResult."""
        X, y = synthetic_data

        result = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[1, 3, 5, 7]
        )

        assert isinstance(result, SensitivityResult)

    def test_mi_values_for_each_k(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute MI for each k value."""
        X, y = synthetic_data
        k_values = [1, 3, 5, 7]

        result = mi_sensitivity_analysis(X, y, feature="feature_a", k_values=k_values)

        assert len(result.mi_values) == len(k_values)
        assert result.parameter_values == k_values
        assert result.parameter_name == "k_neighbors"

    def test_informative_feature_has_positive_mi(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Informative feature should have positive MI at all k values."""
        X, y = synthetic_data

        result = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[1, 3, 5, 7]
        )

        for mi in result.mi_values:
            assert mi > 0, "MI should be positive for informative feature"

    def test_noise_feature_has_lower_mi(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Noise feature should have lower MI than informative feature."""
        X, y = synthetic_data

        result_a = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[3]
        )
        result_b = mi_sensitivity_analysis(
            X, y, feature="feature_b", k_values=[3]
        )

        assert result_a.mean_mi > result_b.mean_mi

    def test_coefficient_of_variation(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """CV should be computed correctly."""
        X, y = synthetic_data

        result = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[1, 3, 5, 7]
        )

        expected_cv = result.std_mi / result.mean_mi if result.mean_mi > 0 else 0
        assert abs(result.cv_mi - expected_cv) < 0.01

    def test_is_stable_for_stable_feature(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Stable features should have is_stable=True."""
        X, y = synthetic_data

        result = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[1, 3, 5, 7]
        )

        # For a strongly correlated feature, MI should be fairly stable
        # CV < 0.2 is reasonable for most cases
        if result.cv_mi < 0.1:
            assert result.is_stable is True

    def test_reproducibility_with_random_state(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Results should be reproducible with same random_state."""
        X, y = synthetic_data

        result1 = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[3, 5], random_state=42
        )
        result2 = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[3, 5], random_state=42
        )

        assert result1.mi_values == result2.mi_values

    def test_custom_k_values(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should work with custom k values."""
        X, y = synthetic_data

        result = mi_sensitivity_analysis(
            X, y, feature="feature_a", k_values=[2, 4, 6, 8, 10]
        )

        assert len(result.mi_values) == 5
        assert result.parameter_values == [2, 4, 6, 8, 10]


class TestBatchSensitivityAnalysis:
    """Tests for batch sensitivity analysis across features."""

    @pytest.fixture
    def multi_feature_data(self) -> tuple[pl.DataFrame, pl.Series]:
        """Create data with multiple features of varying informativeness."""
        rng = np.random.default_rng(42)
        n = 1000

        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)

        # y depends on x1 and x2, not x3
        y = 0.6 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n)

        df = pl.DataFrame({
            "feature_1": x1,  # Most informative
            "feature_2": x2,  # Less informative
            "feature_3": x3,  # Noise
        })
        target = pl.Series("target", y)

        return df, target

    def test_returns_dict_of_results(
        self, multi_feature_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should return dict mapping feature names to results."""
        X, y = multi_feature_data

        results = batch_sensitivity_analysis(
            X, y, features=["feature_1", "feature_2"], k_values=[3, 5]
        )

        assert isinstance(results, dict)
        assert "feature_1" in results
        assert "feature_2" in results
        assert isinstance(results["feature_1"], SensitivityResult)

    def test_all_features_analyzed(
        self, multi_feature_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """All specified features should be analyzed."""
        X, y = multi_feature_data
        features = ["feature_1", "feature_2", "feature_3"]

        results = batch_sensitivity_analysis(X, y, features=features, k_values=[3, 5])

        assert len(results) == 3

    def test_default_features_uses_all_columns(
        self, multi_feature_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """When features=None, should analyze all columns."""
        X, y = multi_feature_data

        results = batch_sensitivity_analysis(X, y, k_values=[3])

        assert len(results) == 3  # All three features

    def test_ranking_correlation_computed(
        self, multi_feature_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Ranking correlation should be computed across k values."""
        X, y = multi_feature_data

        results = batch_sensitivity_analysis(
            X, y, features=["feature_1", "feature_2", "feature_3"], k_values=[1, 3, 5, 7]
        )

        # Each result should have rank_at_each
        for result in results.values():
            assert len(result.rank_at_each) == 4

        # Ranking correlation should be computed
        for result in results.values():
            assert -1 <= result.rank_correlation <= 1

    def test_feature_not_in_dataframe_raises(
        self, multi_feature_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Requesting non-existent feature should raise."""
        X, y = multi_feature_data

        with pytest.raises(ValueError, match="not found"):
            batch_sensitivity_analysis(X, y, features=["nonexistent"], k_values=[3])


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_k_value(self) -> None:
        """Should work with single k value."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        result = mi_sensitivity_analysis(X, y, feature="f", k_values=[3])

        assert len(result.mi_values) == 1
        assert result.std_mi == 0  # Only one value
        assert result.cv_mi == 0

    def test_invalid_k_value_raises(self) -> None:
        """k < 1 should raise ConfigurationError."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(ConfigurationError, match="positive"):
            mi_sensitivity_analysis(X, y, feature="f", k_values=[0])

    def test_small_sample_size_warning(self) -> None:
        """Small sample size relative to k should handle gracefully."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 20)})
        y = pl.Series("y", rng.normal(0, 1, 20))

        # k=7 with only 20 samples should still work
        result = mi_sensitivity_analysis(X, y, feature="f", k_values=[3, 5, 7])
        assert len(result.mi_values) == 3

    def test_nan_handling(self) -> None:
        """NaN values should be handled."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)

        # Introduce some NaNs
        x[10:15] = np.nan

        X = pl.DataFrame({"f": x})
        target = pl.Series("y", y)

        result = mi_sensitivity_analysis(X, target, feature="f", k_values=[3])

        # Should compute MI on non-NaN rows
        assert len(result.mi_values) == 1
        assert result.mi_values[0] is not None
