"""Tests for statistical significance testing functions."""

import numpy as np
import polars as pl
import pytest

from liq.features.selection.significance import (
    apply_fdr_correction,
    batch_bootstrap_mi,
    batch_paired_difference,
    batch_permutation_test,
    bootstrap_mi,
    paired_bootstrap_difference,
    permutation_test_mi,
    run_significance_analysis,
)
from liq.features.selection.significance_results import (
    BootstrapResult,
    PairedDifferenceResult,
    PermutationResult,
    SignificanceReport,
)


class TestBootstrapMI:
    """Tests for bootstrap_mi function."""

    def test_basic_bootstrap(self) -> None:
        """Test basic bootstrap CI calculation."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)  # Strong relationship

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = bootstrap_mi(
            features,
            target,
            "x",
            n_bootstrap=100,  # Small for speed
            random_state=42,
        )

        # Check result type and structure
        assert isinstance(result, BootstrapResult)
        assert result.feature == "x"
        assert result.point_estimate > 0
        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.std_error > 0
        assert result.n_bootstrap == 100
        assert result.confidence_level == 0.95
        assert result.n_samples == n

    def test_confidence_interval_contains_estimate(self) -> None:
        """Test that CI contains the point estimate."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = x + 0.3 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = bootstrap_mi(
            features,
            target,
            "x",
            n_bootstrap=200,
            random_state=42,
        )

        assert result.ci_lower <= result.point_estimate
        assert result.point_estimate <= result.ci_upper

    def test_reproducibility_with_seed(self) -> None:
        """Test reproducibility with fixed random state."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result1 = bootstrap_mi(features, target, "x", n_bootstrap=50, random_state=123)
        result2 = bootstrap_mi(features, target, "x", n_bootstrap=50, random_state=123)

        assert result1.point_estimate == result2.point_estimate
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper

    def test_handles_nan_values(self) -> None:
        """Test NaN values are handled correctly."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        x[:20] = np.nan  # 10% NaN
        y = np.random.randn(n)
        y[180:] = np.nan  # Some NaN in target too

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = bootstrap_mi(features, target, "x", n_bootstrap=50, random_state=42)

        # Should have fewer valid samples
        assert result.n_samples < n
        assert result.n_samples > 100  # Still enough to compute

    def test_raises_on_insufficient_samples(self) -> None:
        """Test raises error when too few valid samples."""
        features = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        target = pl.Series("y", [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Insufficient valid samples"):
            bootstrap_mi(features, target, "x")

    def test_raises_on_missing_feature(self) -> None:
        """Test raises error for non-existent feature."""
        features = pl.DataFrame({"x": [1.0] * 200})
        target = pl.Series("y", [1.0] * 200)

        with pytest.raises(ValueError, match="Feature 'missing' not found"):
            bootstrap_mi(features, target, "missing")

    def test_confidence_level_affects_ci(self) -> None:
        """Test that higher confidence level gives wider CI."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result_90 = bootstrap_mi(
            features,
            target,
            "x",
            n_bootstrap=200,
            confidence_level=0.90,
            random_state=42,
        )
        result_99 = bootstrap_mi(
            features,
            target,
            "x",
            n_bootstrap=200,
            confidence_level=0.99,
            random_state=42,
        )

        width_90 = result_90.ci_upper - result_90.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower

        assert width_99 > width_90


class TestPermutationTestMI:
    """Tests for permutation_test_mi function."""

    def test_basic_permutation_test(self) -> None:
        """Test basic permutation test returns expected format."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = permutation_test_mi(
            features,
            target,
            "x",
            n_permutations=100,
            random_state=42,
        )

        assert isinstance(result, PermutationResult)
        assert result.feature == "x"
        assert result.observed_mi > 0
        assert 0 <= result.p_value <= 1
        assert result.null_mean >= 0
        assert result.null_std >= 0
        assert result.n_permutations == 100
        assert result.n_samples == n

    def test_related_variables_low_pvalue(self) -> None:
        """Test strongly related variables give low p-value."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = x + 0.1 * np.random.randn(n)  # Very strong relationship

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = permutation_test_mi(
            features,
            target,
            "x",
            n_permutations=200,
            random_state=42,
        )

        # Should be significant
        assert result.p_value < 0.05
        assert result.is_significant()

    def test_independent_variables_high_pvalue(self) -> None:
        """Test independent variables give high p-value."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = np.random.randn(n)  # Independent

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = permutation_test_mi(
            features,
            target,
            "x",
            n_permutations=200,
            random_state=42,
        )

        # Should not be significant (p > 0.05 typically)
        # With random data, p-value should be uniformly distributed
        assert result.p_value > 0.01  # Very unlikely to be significant

    def test_null_distribution_near_zero(self) -> None:
        """Test null distribution has mean near zero."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result = permutation_test_mi(
            features,
            target,
            "x",
            n_permutations=200,
            random_state=42,
        )

        # Null mean should be low (MI under null is ~0)
        assert result.null_mean < 0.1
        # Observed should be much higher than null
        assert result.observed_mi > result.null_mean + 2 * result.null_std

    def test_reproducibility_with_seed(self) -> None:
        """Test reproducibility with fixed random state."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        result1 = permutation_test_mi(features, target, "x", n_permutations=50, random_state=123)
        result2 = permutation_test_mi(features, target, "x", n_permutations=50, random_state=123)

        assert result1.observed_mi == result2.observed_mi
        assert result1.p_value == result2.p_value


class TestPairedBootstrapDifference:
    """Tests for paired_bootstrap_difference function."""

    def test_basic_paired_test(self) -> None:
        """Test basic paired difference test returns expected format."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y_close = x + 0.5 * np.random.randn(n)
        y_midrange = x + 0.3 * np.random.randn(n)  # Stronger relationship

        features = pl.DataFrame({"x": x})
        target_close = pl.Series("y_close", y_close)
        target_midrange = pl.Series("y_midrange", y_midrange)

        result = paired_bootstrap_difference(
            features,
            target_close,
            target_midrange,
            "x",
            n_bootstrap=100,
            random_state=42,
        )

        assert isinstance(result, PairedDifferenceResult)
        assert result.feature == "x"
        assert result.close_mi > 0
        assert result.midrange_mi > 0
        assert result.difference == pytest.approx(result.midrange_mi - result.close_mi)
        assert 0 <= result.p_value <= 1
        assert result.n_bootstrap == 100
        assert result.n_samples == n

    def test_positive_difference_low_pvalue(self) -> None:
        """Test midrange > close gives low p-value."""
        np.random.seed(42)
        n = 400
        x = np.random.randn(n)
        y_close = x + np.random.randn(n)  # Weaker
        y_midrange = x + 0.1 * np.random.randn(n)  # Much stronger

        features = pl.DataFrame({"x": x})
        target_close = pl.Series("y_close", y_close)
        target_midrange = pl.Series("y_midrange", y_midrange)

        result = paired_bootstrap_difference(
            features,
            target_close,
            target_midrange,
            "x",
            n_bootstrap=200,
            random_state=42,
        )

        # Midrange should have higher MI
        assert result.difference > 0
        assert result.midrange_mi > result.close_mi
        # P-value should be low (H0: diff <= 0 rejected)
        assert result.p_value < 0.10
        assert result.is_significant(alpha=0.10)

    def test_ci_contains_difference(self) -> None:
        """Test that CI contains the point estimate."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y_close = x + 0.5 * np.random.randn(n)
        y_midrange = x + 0.3 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target_close = pl.Series("y_close", y_close)
        target_midrange = pl.Series("y_midrange", y_midrange)

        result = paired_bootstrap_difference(
            features,
            target_close,
            target_midrange,
            "x",
            n_bootstrap=200,
            random_state=42,
        )

        # CI should contain point estimate
        assert result.ci_lower <= result.difference <= result.ci_upper

    def test_relative_improvement(self) -> None:
        """Test relative improvement calculation."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y_close = x + 0.5 * np.random.randn(n)
        y_midrange = x + 0.3 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target_close = pl.Series("y_close", y_close)
        target_midrange = pl.Series("y_midrange", y_midrange)

        result = paired_bootstrap_difference(
            features,
            target_close,
            target_midrange,
            "x",
            n_bootstrap=100,
            random_state=42,
        )

        expected_improvement = result.difference / result.close_mi
        assert result.relative_improvement == pytest.approx(expected_improvement)


class TestBatchFunctions:
    """Tests for batch processing functions."""

    def test_batch_bootstrap_mi(self) -> None:
        """Test batch bootstrap returns results for all features."""
        np.random.seed(42)
        n = 200
        features = pl.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "x3": np.random.randn(n),
            }
        )
        target = pl.Series("y", features["x1"].to_numpy() + 0.5 * np.random.randn(n))

        results = batch_bootstrap_mi(
            features,
            target,
            features=["x1", "x2", "x3"],
            n_bootstrap=50,
            random_state=42,
            n_jobs=2,
        )

        assert len(results) == 3
        feature_names = {r.feature for r in results}
        assert feature_names == {"x1", "x2", "x3"}

    def test_batch_permutation_test(self) -> None:
        """Test batch permutation returns results for all features."""
        np.random.seed(42)
        n = 200
        features = pl.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
            }
        )
        target = pl.Series("y", features["x1"].to_numpy() + 0.5 * np.random.randn(n))

        results = batch_permutation_test(
            features,
            target,
            features=["x1", "x2"],
            n_permutations=50,
            random_state=42,
            n_jobs=2,
        )

        assert len(results) == 2
        feature_names = {r.feature for r in results}
        assert feature_names == {"x1", "x2"}

    def test_batch_paired_difference(self) -> None:
        """Test batch paired difference returns results for all features."""
        np.random.seed(42)
        n = 200
        features = pl.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
            }
        )
        y_close = pl.Series("y_close", features["x1"].to_numpy() + 0.5 * np.random.randn(n))
        y_midrange = pl.Series("y_mid", features["x1"].to_numpy() + 0.3 * np.random.randn(n))

        results = batch_paired_difference(
            features,
            y_close,
            y_midrange,
            features=["x1", "x2"],
            n_bootstrap=50,
            random_state=42,
            n_jobs=2,
        )

        assert len(results) == 2
        feature_names = {r.feature for r in results}
        assert feature_names == {"x1", "x2"}

    def test_batch_with_progress_callback(self) -> None:
        """Test progress callback is called."""
        np.random.seed(42)
        n = 200
        features = pl.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
            }
        )
        target = pl.Series("y", np.random.randn(n))

        progress_calls = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        batch_bootstrap_mi(
            features,
            target,
            features=["x1", "x2"],
            n_bootstrap=20,
            random_state=42,
            n_jobs=1,
            progress_callback=callback,
        )

        # Should have been called for each feature
        assert len(progress_calls) == 2
        assert all(total == 2 for _, total in progress_calls)


class TestFDRCorrection:
    """Tests for apply_fdr_correction function."""

    def test_basic_fdr_correction(self) -> None:
        """Test basic FDR correction."""
        p_values = [0.001, 0.01, 0.03, 0.05, 0.10]

        significant, adjusted = apply_fdr_correction(p_values, alpha=0.05)

        assert len(significant) == 5
        assert len(adjusted) == 5
        assert all(isinstance(s, bool) for s in significant)
        assert all(isinstance(a, float) for a in adjusted)

    def test_adjusted_pvalues_monotonic(self) -> None:
        """Test adjusted p-values maintain monotonicity."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        _, adjusted = apply_fdr_correction(p_values)

        # Adjusted values should be in same order as original
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1]

    def test_adjusted_pvalues_capped_at_one(self) -> None:
        """Test adjusted p-values don't exceed 1."""
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9]

        _, adjusted = apply_fdr_correction(p_values)

        assert all(a <= 1.0 for a in adjusted)

    def test_handles_nan_pvalues(self) -> None:
        """Test NaN p-values are handled."""
        p_values = [0.01, np.nan, 0.05, np.nan, 0.10]

        significant, adjusted = apply_fdr_correction(p_values)

        assert len(significant) == 5
        # NaN p-values should result in NaN adjusted and False significant
        assert not significant[1]
        assert not significant[3]
        assert np.isnan(adjusted[1])
        assert np.isnan(adjusted[3])

    def test_empty_list(self) -> None:
        """Test empty p-value list."""
        significant, adjusted = apply_fdr_correction([])

        assert significant == []
        assert adjusted == []

    def test_all_significant(self) -> None:
        """Test case where all p-values are significant."""
        p_values = [0.001, 0.002, 0.003]

        significant, _ = apply_fdr_correction(p_values, alpha=0.05)

        assert all(significant)

    def test_none_significant(self) -> None:
        """Test case where no p-values are significant."""
        p_values = [0.5, 0.6, 0.7]

        significant, _ = apply_fdr_correction(p_values, alpha=0.05)

        assert not any(significant)


class TestResultDataclasses:
    """Tests for result dataclass serialization."""

    def test_bootstrap_result_to_dict(self) -> None:
        """Test BootstrapResult serialization."""
        result = BootstrapResult(
            feature="x1",
            point_estimate=0.5,
            ci_lower=0.4,
            ci_upper=0.6,
            std_error=0.05,
            n_bootstrap=1000,
            confidence_level=0.95,
            n_samples=500,
        )

        d = result.to_dict()

        assert d["feature"] == "x1"
        assert d["point_estimate"] == 0.5
        assert d["ci_lower"] == 0.4
        assert d["ci_upper"] == 0.6

    def test_bootstrap_result_from_dict(self) -> None:
        """Test BootstrapResult deserialization."""
        d = {
            "feature": "x1",
            "point_estimate": 0.5,
            "ci_lower": 0.4,
            "ci_upper": 0.6,
            "std_error": 0.05,
            "n_bootstrap": 1000,
            "confidence_level": 0.95,
            "n_samples": 500,
        }

        result = BootstrapResult.from_dict(d)

        assert result.feature == "x1"
        assert result.point_estimate == 0.5
        assert result.ci_lower == 0.4

    def test_permutation_result_round_trip(self) -> None:
        """Test PermutationResult serialization round-trip."""
        original = PermutationResult(
            feature="x1",
            observed_mi=0.5,
            p_value=0.001,
            null_mean=0.02,
            null_std=0.01,
            n_permutations=1000,
            n_samples=500,
        )

        d = original.to_dict()
        restored = PermutationResult.from_dict(d)

        assert restored.feature == original.feature
        assert restored.observed_mi == original.observed_mi
        assert restored.p_value == original.p_value
        assert restored.is_significant() == original.is_significant()

    def test_paired_difference_result_round_trip(self) -> None:
        """Test PairedDifferenceResult serialization round-trip."""
        original = PairedDifferenceResult(
            feature="x1",
            close_mi=0.3,
            midrange_mi=0.5,
            difference=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
            p_value=0.01,
            n_bootstrap=1000,
            n_samples=500,
        )

        d = original.to_dict()
        restored = PairedDifferenceResult.from_dict(d)

        assert restored.feature == original.feature
        assert restored.difference == original.difference
        assert restored.relative_improvement == pytest.approx(original.relative_improvement)

    def test_significance_report_round_trip(self) -> None:
        """Test SignificanceReport serialization round-trip."""
        bootstrap = BootstrapResult(
            feature="x1",
            point_estimate=0.5,
            ci_lower=0.4,
            ci_upper=0.6,
            std_error=0.05,
            n_bootstrap=100,
            confidence_level=0.95,
            n_samples=500,
        )
        permutation = PermutationResult(
            feature="x1",
            observed_mi=0.5,
            p_value=0.001,
            null_mean=0.02,
            null_std=0.01,
            n_permutations=100,
            n_samples=500,
        )
        paired = PairedDifferenceResult(
            feature="x1",
            close_mi=0.3,
            midrange_mi=0.5,
            difference=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
            p_value=0.01,
            n_bootstrap=100,
            n_samples=500,
        )

        original = SignificanceReport(
            timeframe="1h",
            close_bootstrap=[bootstrap],
            midrange_bootstrap=[bootstrap],
            close_permutation=[permutation],
            midrange_permutation=[permutation],
            paired_results=[paired],
            n_features=1,
            n_bootstrap=100,
            n_permutations=100,
            confidence_level=0.95,
            fdr_adjusted_p_values=[0.01],
            fdr_significant=[True],
        )

        d = original.to_dict()
        restored = SignificanceReport.from_dict(d)

        assert restored.timeframe == original.timeframe
        assert restored.n_features == original.n_features
        assert len(restored.close_bootstrap) == 1
        assert len(restored.paired_results) == 1
        assert restored.n_significant_paired_fdr == 1

    def test_significance_report_summary_counts(self) -> None:
        """Test SignificanceReport property calculations."""
        perm_sig = PermutationResult(
            feature="x1",
            observed_mi=0.5,
            p_value=0.01,
            null_mean=0.02,
            null_std=0.01,
            n_permutations=100,
            n_samples=500,
        )
        perm_not_sig = PermutationResult(
            feature="x2",
            observed_mi=0.1,
            p_value=0.10,
            null_mean=0.08,
            null_std=0.05,
            n_permutations=100,
            n_samples=500,
        )
        paired_sig = PairedDifferenceResult(
            feature="x1",
            close_mi=0.3,
            midrange_mi=0.5,
            difference=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
            p_value=0.01,
            n_bootstrap=100,
            n_samples=500,
        )
        paired_not_sig = PairedDifferenceResult(
            feature="x2",
            close_mi=0.3,
            midrange_mi=0.31,
            difference=0.01,
            ci_lower=-0.1,
            ci_upper=0.1,
            p_value=0.40,
            n_bootstrap=100,
            n_samples=500,
        )

        report = SignificanceReport(
            timeframe="1h",
            close_permutation=[perm_sig, perm_not_sig],
            midrange_permutation=[perm_sig],
            paired_results=[paired_sig, paired_not_sig],
            fdr_significant=[True, False],
            n_features=2,
        )

        assert report.n_significant_permutation_close == 1
        assert report.n_significant_permutation_midrange == 1
        assert report.n_significant_paired == 1
        assert report.n_significant_paired_fdr == 1


class TestRunSignificanceAnalysis:
    """Tests for the full significance analysis orchestration."""

    def test_run_significance_analysis_populates_all_stages_and_progress(self) -> None:
        """Full report includes bootstrap, permutation, paired, and FDR results."""
        rng = np.random.default_rng(42)
        n = 140
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        features = pl.DataFrame({"x1": x1, "x2": x2})
        close = pl.Series("close", x1 + 0.6 * rng.normal(size=n))
        midrange = pl.Series("midrange", x1 + 0.2 * rng.normal(size=n))
        progress_calls: list[tuple[str, int, int]] = []

        report = run_significance_analysis(
            features,
            close,
            midrange,
            ["x1", "x2"],
            timeframe="1h",
            n_bootstrap=5,
            n_permutations=5,
            random_state=42,
            n_jobs=1,
            progress_callback=lambda stage, current, total: progress_calls.append(
                (stage, current, total)
            ),
        )

        assert report.timeframe == "1h"
        assert report.n_features == 2
        assert report.n_bootstrap == 5
        assert report.n_permutations == 5
        assert len(report.close_bootstrap) == 2
        assert len(report.midrange_bootstrap) == 2
        assert len(report.close_permutation) == 2
        assert len(report.midrange_permutation) == 2
        assert len(report.paired_results) == 2
        assert len(report.fdr_significant) == 2
        assert len(report.fdr_adjusted_p_values) == 2
        assert {call[0] for call in progress_calls} == {
            "Bootstrap (close)",
            "Bootstrap (midrange)",
            "Permutation (close)",
            "Permutation (midrange)",
            "Paired difference",
        }
        assert all(total == 2 for _, _, total in progress_calls)

    def test_run_significance_analysis_without_paired_results_skips_fdr(self) -> None:
        """Invalid features produce an empty paired set and leave FDR fields empty."""
        features = pl.DataFrame({"x1": [1.0, 2.0, 3.0]})
        close = pl.Series("close", [1.0, 2.0, 3.0])
        midrange = pl.Series("midrange", [1.0, 2.0, 3.0])

        report = run_significance_analysis(
            features,
            close,
            midrange,
            ["missing"],
            n_bootstrap=1,
            n_permutations=1,
            n_jobs=1,
        )

        assert report.paired_results == []
        assert report.fdr_significant == []
        assert report.fdr_adjusted_p_values == []
