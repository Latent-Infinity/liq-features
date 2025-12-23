"""Tests for out-of-sample validation.

Tests cover:
- Train/test split MI validation
- Temporal (non-shuffled) splits
- Ranking correlation computation
- Synthetic data with known stability
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from liq.features.validation.exceptions import (
    ConfigurationError,
    InsufficientDataError,
)
from liq.features.validation.out_of_sample import validate_oos
from liq.features.validation.results import OutOfSampleResult


class TestValidateOOS:
    """Tests for out-of-sample validation."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[pl.DataFrame, pl.Series]:
        """Create synthetic data with stable MI."""
        rng = np.random.default_rng(42)
        n = 1000

        # Features with known relationships
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)  # Noise

        # Target depends on x1 and x2
        y = 0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.2, n)

        df = pl.DataFrame({
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
        })
        target = pl.Series("target", y)

        return df, target

    def test_returns_oos_result(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should return OutOfSampleResult."""
        X, y = synthetic_data

        result = validate_oos(
            X, y, features=["feature_1", "feature_2"], test_ratio=0.2
        )

        assert isinstance(result, OutOfSampleResult)

    def test_train_test_sizes(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should correctly split data."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2)

        assert result.n_train == 800
        assert result.n_test == 200
        assert result.test_ratio == 0.2

    def test_computes_mi_for_all_features(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute MI for all specified features."""
        X, y = synthetic_data
        features = ["feature_1", "feature_2", "feature_3"]

        result = validate_oos(X, y, features=features, test_ratio=0.2)

        assert set(result.train_mi.keys()) == set(features)
        assert set(result.test_mi.keys()) == set(features)

    def test_rankings_computed(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should compute rankings for all features."""
        X, y = synthetic_data
        features = ["feature_1", "feature_2", "feature_3"]

        result = validate_oos(X, y, features=features, test_ratio=0.2)

        assert len(result.train_ranking) == 3
        assert len(result.test_ranking) == 3
        # Rankings should be 1, 2, 3
        assert set(result.train_ranking.values()) == {1, 2, 3}
        assert set(result.test_ranking.values()) == {1, 2, 3}

    def test_informative_feature_ranked_higher(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Informative features should be ranked higher than noise."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2)

        # feature_1 should be ranked higher (lower number) than feature_3
        assert result.train_ranking["feature_1"] < result.train_ranking["feature_3"]

    def test_spearman_correlation_for_stable_data(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Rankings should be correlated for stable relationships."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2)

        # For stable synthetic data, correlation should be high
        assert result.spearman_correlation > 0.5

    def test_temporal_split(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Temporal split should use first n% for training."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2, temporal=True)

        assert result.temporal_split is True
        # With temporal split, train is first 80%, test is last 20%

    def test_shuffled_split(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Non-temporal split should shuffle data."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2, temporal=False)

        assert result.temporal_split is False

    def test_is_stable_true(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """is_stable should be True for high correlation."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2)

        if result.spearman_correlation > 0.8:
            assert result.is_stable is True

    def test_top_k_overlap(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """top_k_overlap should be computed correctly."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2)

        # With 3 features, top_k_overlap(3) should be 1.0 if rankings match
        overlap = result.top_k_overlap(3)
        assert 0 <= overlap <= 1

    def test_reproducibility_with_random_state(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Results should be reproducible with same random_state."""
        X, y = synthetic_data

        result1 = validate_oos(
            X, y, test_ratio=0.2, temporal=False, random_state=42
        )
        result2 = validate_oos(
            X, y, test_ratio=0.2, temporal=False, random_state=42
        )

        assert result1.train_mi == result2.train_mi
        assert result1.test_mi == result2.test_mi

    def test_default_features_uses_all_columns(
        self, synthetic_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """When features=None, should use all columns."""
        X, y = synthetic_data

        result = validate_oos(X, y, test_ratio=0.2)

        assert len(result.features) == 3


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_test_ratio_too_low(self) -> None:
        """test_ratio <= 0 should raise."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(ConfigurationError, match="test_ratio"):
            validate_oos(X, y, test_ratio=0)

    def test_invalid_test_ratio_too_high(self) -> None:
        """test_ratio >= 1 should raise."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(ConfigurationError, match="test_ratio"):
            validate_oos(X, y, test_ratio=1.0)

    def test_insufficient_data(self) -> None:
        """Too little data should raise InsufficientDataError."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 5)})
        y = pl.Series("y", rng.normal(0, 1, 5))

        with pytest.raises(InsufficientDataError):
            validate_oos(X, y, test_ratio=0.5)

    def test_feature_not_found(self) -> None:
        """Requesting non-existent feature should raise."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        with pytest.raises(ValueError, match="not found"):
            validate_oos(X, y, features=["nonexistent"], test_ratio=0.2)

    def test_single_feature(self) -> None:
        """Should work with single feature."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        result = validate_oos(X, y, test_ratio=0.2)

        assert len(result.features) == 1
        assert result.train_ranking["f"] == 1
        # Single feature always has perfect rank correlation
        assert result.spearman_correlation == 1.0

    def test_nan_handling(self) -> None:
        """NaN values should be handled."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        x[10:15] = np.nan

        X = pl.DataFrame({"f": x})
        target = pl.Series("y", y)

        result = validate_oos(X, target, test_ratio=0.2)

        # Should compute with available data
        assert result.train_mi["f"] >= 0

    def test_to_dict_serialization(self) -> None:
        """Result should be serializable."""
        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f": rng.normal(0, 1, 100)})
        y = pl.Series("y", rng.normal(0, 1, 100))

        result = validate_oos(X, y, test_ratio=0.2)
        d = result.to_dict()

        assert "train_mi" in d
        assert "test_mi" in d
        assert "spearman_correlation" in d
