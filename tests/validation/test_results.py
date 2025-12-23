"""Tests for validation result dataclasses.

Verifies serialization, deserialization, and computed properties.
"""

from __future__ import annotations

import pytest

from liq.features.validation.results import (
    EffectSizeResult,
    ModelComparisonResult,
    OutOfSampleResult,
    SensitivityResult,
    TemporalStabilityResult,
)


class TestOutOfSampleResult:
    """Tests for OutOfSampleResult dataclass."""

    @pytest.fixture
    def oos_result(self) -> OutOfSampleResult:
        """Create a sample OOS result."""
        return OutOfSampleResult(
            features=["feature_a", "feature_b", "feature_c"],
            train_mi={"feature_a": 0.3, "feature_b": 0.2, "feature_c": 0.1},
            test_mi={"feature_a": 0.28, "feature_b": 0.22, "feature_c": 0.12},
            train_ranking={"feature_a": 1, "feature_b": 2, "feature_c": 3},
            test_ranking={"feature_a": 1, "feature_b": 2, "feature_c": 3},
            spearman_correlation=0.95,
            n_train=800,
            n_test=200,
            test_ratio=0.2,
            temporal_split=True,
        )

    def test_is_stable_true(self, oos_result: OutOfSampleResult) -> None:
        """is_stable should be True for high correlation."""
        assert oos_result.is_stable is True

    def test_is_stable_false(self) -> None:
        """is_stable should be False for low correlation."""
        result = OutOfSampleResult(
            features=["a"],
            train_mi={"a": 0.1},
            test_mi={"a": 0.1},
            train_ranking={"a": 1},
            test_ranking={"a": 1},
            spearman_correlation=0.5,
            n_train=100,
            n_test=20,
            test_ratio=0.2,
            temporal_split=True,
        )
        assert result.is_stable is False

    def test_top_k_overlap_full(self, oos_result: OutOfSampleResult) -> None:
        """top_k_overlap should be 1.0 when rankings match."""
        assert oos_result.top_k_overlap(3) == 1.0

    def test_top_k_overlap_partial(self) -> None:
        """top_k_overlap should be partial when rankings differ."""
        result = OutOfSampleResult(
            features=["a", "b", "c", "d"],
            train_mi={"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1},
            test_mi={"a": 0.3, "b": 0.4, "c": 0.1, "d": 0.2},
            train_ranking={"a": 1, "b": 2, "c": 3, "d": 4},
            test_ranking={"b": 1, "a": 2, "d": 3, "c": 4},
            spearman_correlation=0.6,
            n_train=100,
            n_test=20,
            test_ratio=0.2,
            temporal_split=True,
        )
        # Top 2: train=[a, b], test=[b, a] -> overlap = 2/2 = 1.0
        assert result.top_k_overlap(2) == 1.0
        # Top 3: train=[a, b, c], test=[b, a, d] -> overlap = 2/3
        assert abs(result.top_k_overlap(3) - 2 / 3) < 0.01

    def test_to_dict_from_dict_roundtrip(self, oos_result: OutOfSampleResult) -> None:
        """to_dict and from_dict should roundtrip."""
        d = oos_result.to_dict()
        restored = OutOfSampleResult.from_dict(d)

        assert restored.features == oos_result.features
        assert restored.train_mi == oos_result.train_mi
        assert restored.test_mi == oos_result.test_mi
        assert restored.spearman_correlation == oos_result.spearman_correlation

    def test_to_dict_includes_computed_properties(
        self, oos_result: OutOfSampleResult
    ) -> None:
        """to_dict should include computed properties."""
        d = oos_result.to_dict()
        assert "is_stable" in d
        assert "top_10_overlap" in d


class TestModelComparisonResult:
    """Tests for ModelComparisonResult dataclass."""

    @pytest.fixture
    def model_result(self) -> ModelComparisonResult:
        """Create a sample model comparison result."""
        return ModelComparisonResult(
            close_mae=0.02,
            close_rmse=0.025,
            close_directional_accuracy=0.52,
            close_sharpe=0.5,
            midrange_mae=0.015,
            midrange_rmse=0.02,
            midrange_directional_accuracy=0.58,
            midrange_sharpe=0.8,
            n_samples=1000,
            n_features=20,
            features=["f1", "f2", "f3"],
            cv_folds=5,
        )

    def test_mae_improvement(self, model_result: ModelComparisonResult) -> None:
        """mae_improvement should be positive when midrange is better."""
        # (0.02 - 0.015) / 0.02 = 0.25
        assert abs(model_result.mae_improvement - 0.25) < 0.01

    def test_mae_improvement_zero_close(self) -> None:
        """mae_improvement should be 0 when close_mae is 0."""
        result = ModelComparisonResult(
            close_mae=0,
            close_rmse=0,
            close_directional_accuracy=0.5,
            close_sharpe=0,
            midrange_mae=0.01,
            midrange_rmse=0.01,
            midrange_directional_accuracy=0.55,
            midrange_sharpe=0.3,
            n_samples=100,
            n_features=5,
            features=["a"],
        )
        assert result.mae_improvement == 0.0

    def test_rmse_improvement(self, model_result: ModelComparisonResult) -> None:
        """rmse_improvement should be positive when midrange is better."""
        # (0.025 - 0.02) / 0.025 = 0.2
        assert abs(model_result.rmse_improvement - 0.2) < 0.01

    def test_directional_improvement(self, model_result: ModelComparisonResult) -> None:
        """directional_improvement should be 0.06."""
        assert abs(model_result.directional_improvement - 0.06) < 0.01

    def test_sharpe_improvement(self, model_result: ModelComparisonResult) -> None:
        """sharpe_improvement should be 0.3."""
        assert abs(model_result.sharpe_improvement - 0.3) < 0.01

    def test_to_dict_from_dict_roundtrip(
        self, model_result: ModelComparisonResult
    ) -> None:
        """to_dict and from_dict should roundtrip."""
        d = model_result.to_dict()
        restored = ModelComparisonResult.from_dict(d)

        assert restored.close_mae == model_result.close_mae
        assert restored.midrange_sharpe == model_result.midrange_sharpe
        assert restored.features == model_result.features


class TestEffectSizeResult:
    """Tests for EffectSizeResult dataclass."""

    def test_interpret_cohens_d_negligible(self) -> None:
        """Values < 0.2 should be negligible."""
        assert EffectSizeResult.interpret_cohens_d(0.1) == "negligible"
        assert EffectSizeResult.interpret_cohens_d(-0.1) == "negligible"

    def test_interpret_cohens_d_small(self) -> None:
        """Values [0.2, 0.5) should be small."""
        assert EffectSizeResult.interpret_cohens_d(0.3) == "small"
        assert EffectSizeResult.interpret_cohens_d(-0.3) == "small"

    def test_interpret_cohens_d_medium(self) -> None:
        """Values [0.5, 0.8) should be medium."""
        assert EffectSizeResult.interpret_cohens_d(0.6) == "medium"
        assert EffectSizeResult.interpret_cohens_d(-0.6) == "medium"

    def test_interpret_cohens_d_large(self) -> None:
        """Values >= 0.8 should be large."""
        assert EffectSizeResult.interpret_cohens_d(1.0) == "large"
        assert EffectSizeResult.interpret_cohens_d(-1.5) == "large"

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """to_dict and from_dict should roundtrip."""
        result = EffectSizeResult(
            cohens_d=0.75,
            ci_lower=0.5,
            ci_upper=1.0,
            n_group1=50,
            n_group2=50,
            pooled_std=1.0,
            mean_diff=0.75,
            interpretation="medium",
            n_bootstrap=1000,
            confidence_level=0.95,
        )
        d = result.to_dict()
        restored = EffectSizeResult.from_dict(d)

        assert restored.cohens_d == result.cohens_d
        assert restored.interpretation == result.interpretation


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_is_stable_true(self) -> None:
        """is_stable should be True when CV < 0.1."""
        result = SensitivityResult(
            feature="test",
            parameter_name="k_neighbors",
            parameter_values=[1, 3, 5, 7],
            mi_values=[0.10, 0.10, 0.10, 0.10],
            rank_at_each=[1, 1, 1, 1],
            mean_mi=0.10,
            std_mi=0.005,
            cv_mi=0.05,
            rank_correlation=0.99,
        )
        assert result.is_stable is True

    def test_is_stable_false(self) -> None:
        """is_stable should be False when CV >= 0.1."""
        result = SensitivityResult(
            feature="test",
            parameter_name="k_neighbors",
            parameter_values=[1, 3, 5, 7],
            mi_values=[0.05, 0.10, 0.15, 0.20],
            rank_at_each=[3, 2, 1, 1],
            mean_mi=0.125,
            std_mi=0.055,
            cv_mi=0.44,
            rank_correlation=0.60,
        )
        assert result.is_stable is False

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """to_dict and from_dict should roundtrip."""
        result = SensitivityResult(
            feature="natr",
            parameter_name="k_neighbors",
            parameter_values=[1, 3, 5, 7],
            mi_values=[0.1, 0.11, 0.09, 0.1],
            rank_at_each=[1, 1, 2, 1],
            mean_mi=0.1,
            std_mi=0.01,
            cv_mi=0.1,
            rank_correlation=0.95,
        )
        d = result.to_dict()
        restored = SensitivityResult.from_dict(d)

        assert restored.feature == result.feature
        assert restored.parameter_values == result.parameter_values
        assert restored.mi_values == result.mi_values


class TestTemporalStabilityResult:
    """Tests for TemporalStabilityResult dataclass."""

    @pytest.fixture
    def temporal_result(self) -> TemporalStabilityResult:
        """Create a sample temporal stability result."""
        return TemporalStabilityResult(
            features=["f1", "f2", "f3"],
            window_size=1000,
            step_size=500,
            n_windows=5,
            window_starts=[0, 500, 1000, 1500, 2000],
            mi_by_window=[
                {"f1": 0.1, "f2": 0.2, "f3": 0.15},
                {"f1": 0.11, "f2": 0.19, "f3": 0.16},
                {"f1": 0.09, "f2": 0.21, "f3": 0.14},
                {"f1": 0.12, "f2": 0.18, "f3": 0.17},
                {"f1": 0.10, "f2": 0.20, "f3": 0.15},
            ],
            rank_by_window=[
                {"f2": 1, "f3": 2, "f1": 3},
                {"f2": 1, "f3": 2, "f1": 3},
                {"f2": 1, "f3": 2, "f1": 3},
                {"f2": 1, "f3": 2, "f1": 3},
                {"f2": 1, "f3": 2, "f1": 3},
            ],
            adjacent_correlations=[0.9, 0.85, 0.88, 0.92],
            mean_correlation=0.8875,
            min_correlation=0.85,
            regime_changes=[],
        )

    def test_is_stable_true(self, temporal_result: TemporalStabilityResult) -> None:
        """is_stable should be True when mean correlation > 0.8."""
        assert temporal_result.is_stable is True

    def test_is_stable_false(self) -> None:
        """is_stable should be False when mean correlation <= 0.8."""
        result = TemporalStabilityResult(
            features=["a"],
            window_size=100,
            step_size=50,
            n_windows=3,
            window_starts=[0, 50, 100],
            mi_by_window=[{"a": 0.1}, {"a": 0.2}, {"a": 0.05}],
            rank_by_window=[{"a": 1}, {"a": 1}, {"a": 1}],
            adjacent_correlations=[0.5, 0.6],
            mean_correlation=0.55,
            min_correlation=0.5,
        )
        assert result.is_stable is False

    def test_n_regime_changes(self, temporal_result: TemporalStabilityResult) -> None:
        """n_regime_changes should count regime changes."""
        assert temporal_result.n_regime_changes == 0

        result_with_changes = TemporalStabilityResult(
            features=["a"],
            window_size=100,
            step_size=50,
            n_windows=5,
            window_starts=[0, 50, 100, 150, 200],
            mi_by_window=[{"a": 0.1}] * 5,
            rank_by_window=[{"a": 1}] * 5,
            adjacent_correlations=[0.9, 0.3, 0.85, 0.2],
            mean_correlation=0.56,
            min_correlation=0.2,
            regime_changes=[1, 3],
        )
        assert result_with_changes.n_regime_changes == 2

    def test_to_dict_from_dict_roundtrip(
        self, temporal_result: TemporalStabilityResult
    ) -> None:
        """to_dict and from_dict should roundtrip."""
        d = temporal_result.to_dict()
        restored = TemporalStabilityResult.from_dict(d)

        assert restored.features == temporal_result.features
        assert restored.window_size == temporal_result.window_size
        assert restored.mean_correlation == temporal_result.mean_correlation
