"""Tests for validation exception hierarchy.

Verifies that exceptions are properly formatted and contain
contextual information.
"""

from __future__ import annotations

import pytest

from liq.features.validation.exceptions import (
    ConfigurationError,
    ConvergenceError,
    InsufficientDataError,
    ValidationError,
)


class TestValidationError:
    """Tests for ValidationError base class."""

    def test_message_only(self) -> None:
        """ValidationError with message only."""
        err = ValidationError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.context == {}

    def test_message_with_context(self) -> None:
        """ValidationError with context dictionary."""
        err = ValidationError("Failed", {"feature": "natr", "iteration": 5})
        assert "feature=natr" in str(err)
        assert "iteration=5" in str(err)
        assert err.context["feature"] == "natr"

    def test_inheritance(self) -> None:
        """ValidationError should inherit from Exception."""
        err = ValidationError("test")
        assert isinstance(err, Exception)


class TestInsufficientDataError:
    """Tests for InsufficientDataError."""

    def test_basic_usage(self) -> None:
        """InsufficientDataError with required and actual values."""
        err = InsufficientDataError(
            "Not enough samples for bootstrap",
            required=1000,
            actual=500,
        )
        assert err.required == 1000
        assert err.actual == 500
        assert "required=1000" in str(err)
        assert "actual=500" in str(err)

    def test_with_additional_context(self) -> None:
        """InsufficientDataError with additional context."""
        err = InsufficientDataError(
            "Not enough samples",
            required=100,
            actual=50,
            context={"split": "test", "feature": "div"},
        )
        assert "split=test" in str(err)
        assert "feature=div" in str(err)

    def test_inheritance(self) -> None:
        """InsufficientDataError should inherit from ValidationError."""
        err = InsufficientDataError("test", 100, 50)
        assert isinstance(err, ValidationError)
        assert isinstance(err, Exception)


class TestConvergenceError:
    """Tests for ConvergenceError."""

    def test_basic_usage(self) -> None:
        """ConvergenceError with feature and iteration."""
        err = ConvergenceError(
            "MI estimation failed",
            feature="natr_timeperiod5",
            iteration=42,
        )
        assert err.feature == "natr_timeperiod5"
        assert err.iteration == 42
        assert "feature=natr_timeperiod5" in str(err)
        assert "iteration=42" in str(err)

    def test_feature_only(self) -> None:
        """ConvergenceError with feature only."""
        err = ConvergenceError("Zero variance", feature="constant_feature")
        assert err.feature == "constant_feature"
        assert err.iteration is None
        assert "feature=constant_feature" in str(err)

    def test_no_specifics(self) -> None:
        """ConvergenceError without feature or iteration."""
        err = ConvergenceError("Numerical instability")
        assert err.feature is None
        assert err.iteration is None
        assert str(err) == "Numerical instability"

    def test_inheritance(self) -> None:
        """ConvergenceError should inherit from ValidationError."""
        err = ConvergenceError("test")
        assert isinstance(err, ValidationError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_usage(self) -> None:
        """ConfigurationError with parameter and value."""
        err = ConfigurationError(
            "Invalid k value",
            parameter="k_neighbors",
            value=-1,
            valid_range="k >= 1",
        )
        assert err.parameter == "k_neighbors"
        assert err.value == -1
        assert err.valid_range == "k >= 1"
        assert "parameter=k_neighbors" in str(err)
        assert "value=-1" in str(err)
        assert "valid_range=k >= 1" in str(err)

    def test_without_valid_range(self) -> None:
        """ConfigurationError without valid_range."""
        err = ConfigurationError(
            "Missing required parameter",
            parameter="output_dir",
            value=None,
        )
        assert err.valid_range is None
        assert "parameter=output_dir" in str(err)

    def test_inheritance(self) -> None:
        """ConfigurationError should inherit from ValidationError."""
        err = ConfigurationError("test", "param", 0)
        assert isinstance(err, ValidationError)


class TestExceptionCatching:
    """Tests for catching exceptions in hierarchy."""

    def test_catch_all_validation_errors(self) -> None:
        """All validation exceptions should be catchable as ValidationError."""
        exceptions = [
            ValidationError("base"),
            InsufficientDataError("data", 100, 50),
            ConvergenceError("converge"),
            ConfigurationError("config", "param", 0),
        ]

        for exc in exceptions:
            with pytest.raises(ValidationError):
                raise exc

    def test_catch_specific_errors(self) -> None:
        """Specific exceptions should be catchable by type."""
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("test", 100, 50)

        with pytest.raises(ConvergenceError):
            raise ConvergenceError("test")

        with pytest.raises(ConfigurationError):
            raise ConfigurationError("test", "param", 0)
