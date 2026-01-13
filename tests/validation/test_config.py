"""Tests for configuration validation.

Tests verify:
- Invalid parameter values are rejected
- Missing required parameters are detected
- Clear, actionable error messages are provided
- Configuration errors are raised at function entry
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from liq.features.validation.exceptions import ConfigurationError, InsufficientDataError


@pytest.fixture
def sample_data() -> tuple[pl.DataFrame, pl.Series]:
    """Create sample data for testing."""
    rng = np.random.default_rng(42)
    n = 100

    X = pl.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
    })
    y = pl.Series("target", rng.normal(0, 1, n))

    return X, y


class TestSensitivityConfigValidation:
    """Tests for sensitivity analysis configuration validation."""

    def test_negative_k_value_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject negative k values."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            mi_sensitivity_analysis(X, y, "feature_a", k_values=[-1])

        assert "k" in str(exc_info.value).lower()
        assert exc_info.value.parameter == "k_neighbors"

    def test_zero_k_value_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject zero k value."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            mi_sensitivity_analysis(X, y, "feature_a", k_values=[0])

        assert exc_info.value.value == 0

    def test_mixed_valid_invalid_k_values_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject if any k value is invalid."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError):
            mi_sensitivity_analysis(X, y, "feature_a", k_values=[3, -1, 5])

    def test_feature_not_found_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should raise if feature not in DataFrame."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        with pytest.raises(ValueError) as exc_info:
            mi_sensitivity_analysis(X, y, "nonexistent_feature")

        assert "nonexistent_feature" in str(exc_info.value)


class TestOOSConfigValidation:
    """Tests for OOS validation configuration validation."""

    def test_test_ratio_zero_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject test_ratio of 0."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            validate_oos(X, y, features=["feature_a"], test_ratio=0)

        assert exc_info.value.parameter == "test_ratio"
        assert "0 < test_ratio < 1" in exc_info.value.valid_range

    def test_test_ratio_one_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject test_ratio of 1."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            validate_oos(X, y, features=["feature_a"], test_ratio=1)

        assert exc_info.value.value == 1

    def test_test_ratio_negative_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject negative test_ratio."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        with pytest.raises(ConfigurationError):
            validate_oos(X, y, features=["feature_a"], test_ratio=-0.2)

    def test_test_ratio_greater_than_one_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject test_ratio > 1."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        with pytest.raises(ConfigurationError):
            validate_oos(X, y, features=["feature_a"], test_ratio=1.5)

    def test_insufficient_samples_raises(self) -> None:
        """Should raise if not enough samples for split."""
        from liq.features.validation import validate_oos

        rng = np.random.default_rng(42)
        n = 5  # Too few samples

        X = pl.DataFrame({"feature": rng.normal(0, 1, n)})
        y = pl.Series("target", rng.normal(0, 1, n))

        with pytest.raises(InsufficientDataError) as exc_info:
            validate_oos(X, y, features=["feature"], test_ratio=0.5)

        assert exc_info.value.required > exc_info.value.actual


class TestTemporalConfigValidation:
    """Tests for temporal analysis configuration validation."""

    def test_negative_window_size_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject negative window_size."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            rolling_mi_analysis(
                X, y, features=["feature_a"],
                window_size=-10, step_size=5,
            )

        assert exc_info.value.parameter == "window_size"

    def test_zero_window_size_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject zero window_size."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            rolling_mi_analysis(
                X, y, features=["feature_a"],
                window_size=0, step_size=5,
            )

        assert exc_info.value.value == 0

    def test_negative_step_size_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject negative step_size."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            rolling_mi_analysis(
                X, y, features=["feature_a"],
                window_size=30, step_size=-5,
            )

        assert exc_info.value.parameter == "step_size"

    def test_zero_step_size_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject zero step_size."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            rolling_mi_analysis(
                X, y, features=["feature_a"],
                window_size=30, step_size=0,
            )

        assert exc_info.value.value == 0

    def test_window_size_too_small_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should reject window_size below minimum."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        with pytest.raises(InsufficientDataError) as exc_info:
            rolling_mi_analysis(
                X, y, features=["feature_a"],
                window_size=5, step_size=2,  # Too small
            )

        # Minimum is 20 samples per window
        assert exc_info.value.required >= 20

    def test_window_size_exceeds_data_raises(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Should raise if window_size > n_samples."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        with pytest.raises(InsufficientDataError):
            rolling_mi_analysis(
                X, y, features=["feature_a"],
                window_size=1000, step_size=100,  # Exceeds data
            )


class TestEffectSizeConfigValidation:
    """Tests for effect size configuration validation."""

    def test_group_too_small_raises(self) -> None:
        """Should reject groups with < 2 samples."""
        from liq.features.validation import cohens_d

        group1 = np.array([1.0])  # Only 1 sample
        group2 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError) as exc_info:
            cohens_d(group1, group2)

        assert "at least 2 samples" in str(exc_info.value)

    def test_empty_group_raises(self) -> None:
        """Should reject empty groups."""
        from liq.features.validation import cohens_d

        group1 = np.array([])
        group2 = np.array([1.0, 2.0])

        with pytest.raises(ValueError):
            cohens_d(group1, group2)

    def test_zero_variance_raises(self) -> None:
        """Should reject groups with zero pooled variance."""
        from liq.features.validation import cohens_d

        group1 = np.array([5.0, 5.0, 5.0])  # Zero variance
        group2 = np.array([5.0, 5.0, 5.0])  # Zero variance

        with pytest.raises(ValueError) as exc_info:
            cohens_d(group1, group2)

        assert "zero" in str(exc_info.value).lower()


class TestErrorMessageClarity:
    """Tests for clear, actionable error messages."""

    def test_config_error_includes_parameter_name(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """ConfigurationError should include parameter name."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            mi_sensitivity_analysis(X, y, "feature_a", k_values=[-1])

        assert exc_info.value.parameter is not None

    def test_config_error_includes_invalid_value(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """ConfigurationError should include the invalid value."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            mi_sensitivity_analysis(X, y, "feature_a", k_values=[-5])

        assert exc_info.value.value == -5

    def test_config_error_includes_valid_range(
        self, sample_data: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """ConfigurationError should include valid range info."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        with pytest.raises(ConfigurationError) as exc_info:
            validate_oos(X, y, features=["feature_a"], test_ratio=0)

        assert exc_info.value.valid_range is not None
        assert "0 < test_ratio < 1" in exc_info.value.valid_range

    def test_insufficient_data_includes_required_actual(self) -> None:
        """InsufficientDataError should include required and actual counts."""
        from liq.features.validation import validate_oos

        rng = np.random.default_rng(42)
        n = 5

        X = pl.DataFrame({"feature": rng.normal(0, 1, n)})
        y = pl.Series("target", rng.normal(0, 1, n))

        with pytest.raises(InsufficientDataError) as exc_info:
            validate_oos(X, y, features=["feature"], test_ratio=0.5)

        assert exc_info.value.required is not None
        assert exc_info.value.actual is not None
        assert exc_info.value.actual < exc_info.value.required
