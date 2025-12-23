"""Tests for validation protocol compliance.

Verifies that protocol interfaces are correctly defined and that
implementations can be checked for compliance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import polars as pl
import pytest

from liq.features.validation.protocols import (
    EffectSizeCalculator,
    MetricCalculator,
    SensitivityAnalyzer,
    TemporalAnalyzer,
    Validator,
)
from liq.features.validation.results import (
    EffectSizeResult,
    OutOfSampleResult,
    SensitivityResult,
    TemporalStabilityResult,
)


class TestValidatorProtocol:
    """Tests for Validator protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Validator protocol should be runtime checkable."""
        assert hasattr(Validator, "__protocol_attrs__") or hasattr(
            Validator, "__subclasshook__"
        )

    def test_valid_implementation_satisfies_protocol(self) -> None:
        """A class with validate method should satisfy Validator."""

        class MockValidator:
            def validate(
                self,
                X: pl.DataFrame,
                y: pl.Series,
                features: list[str],
            ) -> OutOfSampleResult:
                return OutOfSampleResult(
                    features=features,
                    train_mi={f: 0.1 for f in features},
                    test_mi={f: 0.1 for f in features},
                    train_ranking={f: i for i, f in enumerate(features)},
                    test_ranking={f: i for i, f in enumerate(features)},
                    spearman_correlation=0.9,
                    n_train=100,
                    n_test=20,
                    test_ratio=0.2,
                    temporal_split=True,
                )

        validator = MockValidator()
        assert isinstance(validator, Validator)

    def test_missing_validate_method_fails_protocol(self) -> None:
        """A class without validate method should not satisfy Validator."""

        class InvalidValidator:
            def some_other_method(self) -> None:
                pass

        invalid = InvalidValidator()
        assert not isinstance(invalid, Validator)


class TestMetricCalculatorProtocol:
    """Tests for MetricCalculator protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """MetricCalculator protocol should be runtime checkable."""
        assert hasattr(MetricCalculator, "__protocol_attrs__") or hasattr(
            MetricCalculator, "__subclasshook__"
        )

    def test_valid_implementation_satisfies_protocol(self) -> None:
        """A class with calculate method should satisfy MetricCalculator."""

        class MockMetricCalculator:
            def calculate(self, x: pl.Series, y: pl.Series) -> float:
                return 0.5

        calculator = MockMetricCalculator()
        assert isinstance(calculator, MetricCalculator)

    def test_missing_calculate_method_fails_protocol(self) -> None:
        """A class without calculate method should not satisfy MetricCalculator."""

        class InvalidCalculator:
            def compute(self, x: pl.Series, y: pl.Series) -> float:
                return 0.5

        invalid = InvalidCalculator()
        assert not isinstance(invalid, MetricCalculator)


class TestTemporalAnalyzerProtocol:
    """Tests for TemporalAnalyzer protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """TemporalAnalyzer protocol should be runtime checkable."""
        assert hasattr(TemporalAnalyzer, "__protocol_attrs__") or hasattr(
            TemporalAnalyzer, "__subclasshook__"
        )

    def test_valid_implementation_satisfies_protocol(self) -> None:
        """A class with analyze method should satisfy TemporalAnalyzer."""

        class MockTemporalAnalyzer:
            def analyze(
                self,
                df: pl.DataFrame,
                X: pl.DataFrame,
                y: pl.Series,
                features: list[str],
                window_size: int,
            ) -> TemporalStabilityResult:
                return TemporalStabilityResult(
                    features=features,
                    window_size=window_size,
                    step_size=window_size // 2,
                    n_windows=5,
                    window_starts=[0, 100, 200, 300, 400],
                    mi_by_window=[{f: 0.1 for f in features} for _ in range(5)],
                    rank_by_window=[
                        {f: i for i, f in enumerate(features)} for _ in range(5)
                    ],
                    adjacent_correlations=[0.9, 0.85, 0.88, 0.92],
                    mean_correlation=0.89,
                    min_correlation=0.85,
                )

        analyzer = MockTemporalAnalyzer()
        assert isinstance(analyzer, TemporalAnalyzer)

    def test_missing_analyze_method_fails_protocol(self) -> None:
        """A class without analyze method should not satisfy TemporalAnalyzer."""

        class InvalidAnalyzer:
            def run_analysis(self) -> None:
                pass

        invalid = InvalidAnalyzer()
        assert not isinstance(invalid, TemporalAnalyzer)


class TestEffectSizeCalculatorProtocol:
    """Tests for EffectSizeCalculator protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """EffectSizeCalculator protocol should be runtime checkable."""
        assert hasattr(EffectSizeCalculator, "__protocol_attrs__") or hasattr(
            EffectSizeCalculator, "__subclasshook__"
        )

    def test_valid_implementation_satisfies_protocol(self) -> None:
        """A class with calculate method should satisfy EffectSizeCalculator."""

        class MockEffectSizeCalculator:
            def calculate(
                self,
                group1: pl.Series,
                group2: pl.Series,
            ) -> EffectSizeResult:
                return EffectSizeResult(
                    cohens_d=0.5,
                    ci_lower=0.3,
                    ci_upper=0.7,
                    n_group1=100,
                    n_group2=100,
                    pooled_std=1.0,
                    mean_diff=0.5,
                    interpretation="medium",
                )

        calculator = MockEffectSizeCalculator()
        assert isinstance(calculator, EffectSizeCalculator)


class TestSensitivityAnalyzerProtocol:
    """Tests for SensitivityAnalyzer protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """SensitivityAnalyzer protocol should be runtime checkable."""
        assert hasattr(SensitivityAnalyzer, "__protocol_attrs__") or hasattr(
            SensitivityAnalyzer, "__subclasshook__"
        )

    def test_valid_implementation_satisfies_protocol(self) -> None:
        """A class with analyze method should satisfy SensitivityAnalyzer."""

        class MockSensitivityAnalyzer:
            def analyze(
                self,
                X: pl.DataFrame,
                y: pl.Series,
                features: list[str],
                parameter_values: list,
            ) -> SensitivityResult:
                return SensitivityResult(
                    feature=features[0],
                    parameter_name="k_neighbors",
                    parameter_values=parameter_values,
                    mi_values=[0.1, 0.11, 0.09, 0.1],
                    rank_at_each=[1, 1, 2, 1],
                    mean_mi=0.1,
                    std_mi=0.01,
                    cv_mi=0.1,
                    rank_correlation=0.95,
                )

        analyzer = MockSensitivityAnalyzer()
        assert isinstance(analyzer, SensitivityAnalyzer)
