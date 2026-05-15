"""Tests for validation module logging.

Tests verify:
- Log messages include relevant context
- Appropriate log levels for different events
- No sensitive data in logs
- Structured logging format
"""

from __future__ import annotations

import logging
from io import StringIO

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_data() -> tuple[pl.DataFrame, pl.Series]:
    """Create sample data for testing."""
    rng = np.random.default_rng(42)
    n = 100

    X = pl.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
        }
    )
    y = pl.Series("target", rng.normal(0, 1, n))

    return X, y


@pytest.fixture
def log_capture() -> StringIO:
    """Capture log output for testing."""
    return StringIO()


@pytest.fixture(autouse=True)
def configure_logging(log_capture: StringIO) -> logging.Logger:
    """Configure logging to capture output."""
    logger = logging.getLogger("liq.features.validation")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Add stream handler to capture output
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class TestLoggingContext:
    """Tests for logging context information."""

    def test_oos_validation_logs_parameters(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """OOS validation should log key parameters."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        validate_oos(X, y, features=["feature_a"], test_ratio=0.2)

        log_output = log_capture.getvalue()

        # Should log test ratio
        assert "0.2" in log_output or "test_ratio" in log_output.lower()

    def test_sensitivity_logs_k_values(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Sensitivity analysis should log k values being tested."""
        from liq.features.validation import mi_sensitivity_analysis

        X, y = sample_data

        mi_sensitivity_analysis(X, y, "feature_a", k_values=[3, 5])

        log_output = log_capture.getvalue()

        # Should log k values
        assert "k" in log_output.lower() or "3" in log_output

    def test_temporal_logs_window_parameters(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Temporal analysis should log window parameters."""
        from liq.features.validation import rolling_mi_analysis

        X, y = sample_data

        rolling_mi_analysis(
            X,
            y,
            features=["feature_a"],
            window_size=30,
            step_size=10,
        )

        log_output = log_capture.getvalue()

        # Should log window size
        assert "30" in log_output or "window" in log_output.lower()


class TestLoggingLevels:
    """Tests for appropriate log levels."""

    def test_debug_level_for_entry_exit(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Function entry/exit should be at DEBUG level."""
        from liq.features.validation import validate_oos

        X, y = sample_data
        validate_oos(X, y, features=["feature_a"])

        log_output = log_capture.getvalue()

        # DEBUG messages should be present
        assert "DEBUG" in log_output

    def test_info_level_for_results(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Key results should be logged at INFO level."""
        from liq.features.validation import validate_oos

        X, y = sample_data
        validate_oos(X, y, features=["feature_a"])

        log_output = log_capture.getvalue()

        # INFO messages should be present for results
        assert "INFO" in log_output

    def test_warning_level_for_small_samples(
        self,
        log_capture: StringIO,
    ) -> None:
        """Small sample sizes should trigger WARNING."""
        from liq.features.validation import mi_sensitivity_analysis

        rng = np.random.default_rng(42)
        n = 25  # Small sample

        X = pl.DataFrame({"feature": rng.normal(0, 1, n)})
        y = pl.Series("target", rng.normal(0, 1, n))

        mi_sensitivity_analysis(X, y, "feature", k_values=[3])

        log_output = log_capture.getvalue()

        # Should warn about small sample
        assert "WARNING" in log_output or "small" in log_output.lower()


class TestNoSensitiveData:
    """Tests to ensure no sensitive data in logs."""

    def test_no_raw_data_in_logs(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Raw data values should not appear in logs."""
        from liq.features.validation import validate_oos

        X, y = sample_data

        # Get first value to check it's not logged
        first_value = str(X["feature_a"][0])

        validate_oos(X, y, features=["feature_a"])

        log_output = log_capture.getvalue()

        # Raw data values should not be in logs
        assert first_value not in log_output

    def test_no_full_arrays_in_logs(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Full arrays should not be dumped to logs."""
        from liq.features.validation import validate_oos

        X, y = sample_data
        validate_oos(X, y, features=["feature_a"])

        log_output = log_capture.getvalue()

        # Should not contain array representations
        assert "[array" not in log_output.lower()
        assert "array([" not in log_output


class TestStructuredLogging:
    """Tests for structured log format."""

    def test_logs_include_module_name(
        self,
        sample_data: tuple[pl.DataFrame, pl.Series],
        log_capture: StringIO,
    ) -> None:
        """Logs should include the module name."""
        from liq.features.validation import validate_oos

        X, y = sample_data
        validate_oos(X, y, features=["feature_a"])

        log_output = log_capture.getvalue()

        # Should include module path
        assert "liq.features.validation" in log_output

    def test_error_logs_include_context(
        self,
        log_capture: StringIO,
    ) -> None:
        """Error logs should include context information."""
        from liq.features.validation import rolling_mi_analysis
        from liq.features.validation.exceptions import ConfigurationError

        rng = np.random.default_rng(42)
        X = pl.DataFrame({"feature": rng.normal(0, 1, 50)})
        y = pl.Series("target", rng.normal(0, 1, 50))

        with pytest.raises(ConfigurationError):
            rolling_mi_analysis(
                X,
                y,
                features=["feature"],
                window_size=-1,  # Invalid
                step_size=10,
            )

        log_output = log_capture.getvalue()

        # Should log the error with context
        assert "ERROR" in log_output or "window" in log_output.lower()


class TestEffectSizeLogging:
    """Tests for effect size logging."""

    def test_cohens_d_logs_computation(
        self,
        log_capture: StringIO,
    ) -> None:
        """Cohen's d should log computation details."""
        from liq.features.validation import cohens_d

        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 50)
        group2 = rng.normal(0.5, 1, 50)

        cohens_d(group1, group2)

        log_output = log_capture.getvalue()

        # Should log that effect size is being computed
        assert "effect" in log_output.lower() or "cohen" in log_output.lower()

    def test_bootstrap_logs_iterations(
        self,
        log_capture: StringIO,
    ) -> None:
        """Bootstrap CI should log iteration count."""
        from liq.features.validation import cohens_d_ci

        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 50)
        group2 = rng.normal(0.5, 1, 50)

        cohens_d_ci(group1, group2, n_bootstrap=100)

        log_output = log_capture.getvalue()

        # Should log bootstrap iterations
        assert "100" in log_output or "bootstrap" in log_output.lower()


class TestLoggingHelpers:
    """Direct tests for logging helper behavior."""

    def test_entry_sanitizes_large_and_structured_values(
        self,
        log_capture: StringIO,
    ) -> None:
        """Entry logs summarize large values without dumping raw contents."""
        from liq.features.validation.logging_config import log_function_entry

        logger = logging.getLogger("liq.features.validation.helper_test")

        log_function_entry(
            logger,
            "sanitize",
            values=np.array([1.0, 2.0, 3.0]),
            small_list=[1, 2, 3],
            large_list=list(range(20)),
            mapping={"secret": "not_logged"},
        )

        log_output = log_capture.getvalue()
        assert "values=<ndarray len=3>" in log_output
        assert "small_list=<list len=3>" in log_output
        assert "large_list=<list len=20>" in log_output
        assert "mapping=<dict>" in log_output
        assert "not_logged" not in log_output

    def test_exit_result_warning_and_error_branches(
        self,
        log_capture: StringIO,
    ) -> None:
        """Helper functions include context only when supplied."""
        from liq.features.validation.logging_config import (
            log_error,
            log_function_exit,
            log_result,
            log_warning,
        )

        logger = logging.getLogger("liq.features.validation.helper_test")
        error = ValueError("bad window")

        log_function_exit(logger, "without_summary")
        log_result(logger, "plain result")
        log_warning(logger, "plain warning")
        log_warning(logger, "context warning", feature="midrange")
        log_error(logger, "context error", error, window_size=-1)

        log_output = log_capture.getvalue()
        assert "Exiting without_summary" in log_output
        assert "plain result" in log_output
        assert "plain warning" in log_output
        assert "context warning (feature=midrange)" in log_output
        assert "context error (window_size=-1)" in log_output
        assert "ValueError: bad window" in log_output
