"""Protocol interfaces for validation and analysis components.

Defines the contracts that validators, metric calculators, and analyzers
must implement to ensure consistent APIs across the validation framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

    from liq.features.validation.results import (
        EffectSizeResult,
        OutOfSampleResult,
        SensitivityResult,
        TemporalStabilityResult,
    )


@runtime_checkable
class Validator(Protocol):
    """Protocol for validation implementations.

    Validators compute out-of-sample metrics to assess MI stability
    and feature selection robustness.

    Example:
        >>> class OOSValidator:
        ...     def validate(
        ...         self, X: pl.DataFrame, y: pl.Series, features: list[str]
        ...     ) -> OutOfSampleResult:
        ...         # Implementation
        ...         ...
    """

    def validate(
        self,
        X: "pl.DataFrame",
        y: "pl.Series",
        features: list[str],
    ) -> "OutOfSampleResult":
        """Validate feature selection on given data.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            features: List of feature names to validate.

        Returns:
            OutOfSampleResult with train/test MI and ranking correlation.
        """
        ...


@runtime_checkable
class MetricCalculator(Protocol):
    """Protocol for information-theoretic metric implementations.

    Metric calculators compute dependency measures between features
    and targets, returning a single numeric score.

    Example:
        >>> class MICalculator:
        ...     def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        ...         return mutual_info_regression(x.reshape(-1, 1), y)[0]
    """

    def calculate(
        self,
        x: "pl.Series",
        y: "pl.Series",
    ) -> float:
        """Calculate dependency metric between x and y.

        Args:
            x: Feature values.
            y: Target values.

        Returns:
            Numeric metric value (higher = stronger dependency).
        """
        ...


@runtime_checkable
class TemporalAnalyzer(Protocol):
    """Protocol for temporal stability analysis implementations.

    Temporal analyzers compute rolling-window metrics to assess
    how stable feature rankings are over time.

    Example:
        >>> class RollingMIAnalyzer:
        ...     def analyze(
        ...         self,
        ...         df: pl.DataFrame,
        ...         X: pl.DataFrame,
        ...         y: pl.Series,
        ...         features: list[str],
        ...         window_size: int,
        ...     ) -> TemporalStabilityResult:
        ...         # Implementation
        ...         ...
    """

    def analyze(
        self,
        df: "pl.DataFrame",
        X: "pl.DataFrame",
        y: "pl.Series",
        features: list[str],
        window_size: int,
    ) -> "TemporalStabilityResult":
        """Analyze temporal stability of feature rankings.

        Args:
            df: Full DataFrame with timestamps.
            X: Feature DataFrame.
            y: Target Series.
            features: List of feature names to analyze.
            window_size: Size of rolling window in rows.

        Returns:
            TemporalStabilityResult with time series of MI values
            and ranking stability metrics.
        """
        ...


@runtime_checkable
class EffectSizeCalculator(Protocol):
    """Protocol for effect size calculation implementations.

    Effect size calculators quantify the magnitude of differences
    between groups, independent of sample size.

    Example:
        >>> class CohensD:
        ...     def calculate(
        ...         self, group1: np.ndarray, group2: np.ndarray
        ...     ) -> EffectSizeResult:
        ...         # Implementation
        ...         ...
    """

    def calculate(
        self,
        group1: "pl.Series",
        group2: "pl.Series",
    ) -> "EffectSizeResult":
        """Calculate effect size between two groups.

        Args:
            group1: First group values.
            group2: Second group values.

        Returns:
            EffectSizeResult with effect size and interpretation.
        """
        ...


@runtime_checkable
class SensitivityAnalyzer(Protocol):
    """Protocol for sensitivity analysis implementations.

    Sensitivity analyzers test robustness to parameter choices,
    such as k-NN neighbors in MI estimation.

    Example:
        >>> class KnnSensitivityAnalyzer:
        ...     def analyze(
        ...         self,
        ...         X: pl.DataFrame,
        ...         y: pl.Series,
        ...         features: list[str],
        ...         k_values: list[int],
        ...     ) -> SensitivityResult:
        ...         # Implementation
        ...         ...
    """

    def analyze(
        self,
        X: "pl.DataFrame",
        y: "pl.Series",
        features: list[str],
        parameter_values: list,
    ) -> "SensitivityResult":
        """Analyze sensitivity to a parameter.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            features: List of feature names to analyze.
            parameter_values: List of parameter values to test.

        Returns:
            SensitivityResult with metrics across parameter values
            and ranking stability assessment.
        """
        ...
