"""Tests for Mutual Information calculation wrapper."""

from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from liq.features.selection.mutual_info import (
    _add_jitter,
    mutual_info_matrix,
    mutual_info_scores,
    mutual_info_scores_per_feature,
)


class TestMutualInfoScores:
    """Tests for mutual_info_scores function."""

    def test_basic_calculation(self) -> None:
        """Test basic MI calculation returns expected format."""
        np.random.seed(42)

        # Create features with known relationship to target
        n = 1000
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = x1 + 0.1 * np.random.randn(n)  # y is strongly correlated with x1

        features = pl.DataFrame({"x1": x1, "x2": x2})
        target = pl.Series("y", y)

        result = mutual_info_scores(features, target, random_state=42)

        # Check output format
        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "mi_score" in result.columns
        assert len(result) == 2

        # x1 should have higher MI than x2
        x1_score = result.filter(pl.col("feature") == "x1")["mi_score"].item()
        x2_score = result.filter(pl.col("feature") == "x2")["mi_score"].item()
        assert x1_score > x2_score

    def test_sorted_descending(self) -> None:
        """Test results are sorted by MI score descending."""
        np.random.seed(42)
        n = 500

        features = pl.DataFrame(
            {
                "low": np.random.randn(n),
                "medium": np.random.randn(n),
                "high": np.random.randn(n),
            }
        )

        # Target correlated with 'high'
        target = pl.Series("y", features["high"].to_numpy() + 0.1 * np.random.randn(n))

        result = mutual_info_scores(features, target, random_state=42)

        # Should be sorted descending
        scores = result["mi_score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_normalize_option(self) -> None:
        """Test normalization scales to [0, 1]."""
        np.random.seed(42)
        n = 500

        x1 = np.random.randn(n)
        y = x1 + 0.1 * np.random.randn(n)

        features = pl.DataFrame({"x1": x1, "x2": np.random.randn(n)})
        target = pl.Series("y", y)

        result = mutual_info_scores(features, target, normalize=True, random_state=42)

        # Max score should be 1.0 after normalization
        max_raw = result["mi_score"].max()
        min_raw = result["mi_score"].min()
        assert isinstance(max_raw, (int, float, Decimal))
        assert isinstance(min_raw, (int, float, Decimal))
        max_score = float(max_raw)
        min_score = float(min_raw)
        assert max_score == pytest.approx(1.0, rel=1e-6)
        assert min_score >= 0.0

    def test_handles_nan_values(self) -> None:
        """Test NaN values are handled correctly with drop_na=True."""
        np.random.seed(42)
        n = 100

        x1 = np.random.randn(n)
        x1[0:10] = np.nan  # Add some NaN values
        y = np.random.randn(n)

        features = pl.DataFrame({"x1": x1, "x2": np.random.randn(n)})
        target = pl.Series("y", y)

        # Should not raise with drop_na=True (default)
        result = mutual_info_scores(features, target, random_state=42)
        assert len(result) == 2

    def test_raises_on_all_nan(self) -> None:
        """Test raises error when all values are NaN after dropping."""
        features = pl.DataFrame({"x1": [np.nan, np.nan, np.nan]})
        target = pl.Series("y", [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="No valid samples"):
            mutual_info_scores(features, target)

    def test_independent_variables_low_mi(self) -> None:
        """Test truly independent variables have low MI."""
        np.random.seed(42)
        n = 1000

        # Truly independent variables
        features = pl.DataFrame({"x": np.random.randn(n)})
        target = pl.Series("y", np.random.randn(n))

        result = mutual_info_scores(features, target, random_state=42)

        # MI should be close to 0 for independent variables
        # (not exactly 0 due to estimation noise)
        mi_score = result["mi_score"].item()
        assert mi_score < 0.1  # Should be very low

    def test_n_neighbors_parameter(self) -> None:
        """Test n_neighbors parameter is passed correctly."""
        np.random.seed(42)
        n = 500

        x = np.random.randn(n)
        y = x + 0.1 * np.random.randn(n)

        features = pl.DataFrame({"x": x})
        target = pl.Series("y", y)

        # Different n_neighbors should give different (but similar) results
        result_3 = mutual_info_scores(features, target, n_neighbors=3, random_state=42)
        result_5 = mutual_info_scores(features, target, n_neighbors=5, random_state=42)

        # Both should detect the relationship
        assert result_3["mi_score"].item() > 0.1
        assert result_5["mi_score"].item() > 0.1


class TestMutualInfoMatrix:
    """Tests for mutual_info_matrix function."""

    def test_basic_matrix(self) -> None:
        """Test pairwise MI matrix calculation."""
        np.random.seed(42)
        n = 500

        x1 = np.random.randn(n)
        x2 = x1 + 0.1 * np.random.randn(n)  # Correlated with x1
        x3 = np.random.randn(n)  # Independent

        features = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        result = mutual_info_matrix(features, random_state=42)

        # Check format
        assert "feature" in result.columns
        assert "x1" in result.columns
        assert "x2" in result.columns
        assert "x3" in result.columns
        assert len(result) == 3

        # Diagonal should be high (self-information)
        x1_x1 = result.filter(pl.col("feature") == "x1")["x1"].item()
        assert x1_x1 > 0.5

        # x1-x2 should have higher MI than x1-x3
        x1_x2 = result.filter(pl.col("feature") == "x1")["x2"].item()
        x1_x3 = result.filter(pl.col("feature") == "x1")["x3"].item()
        assert x1_x2 > x1_x3

    def test_symmetric_matrix(self) -> None:
        """Test matrix is symmetric."""
        np.random.seed(42)
        n = 200

        features = pl.DataFrame(
            {
                "a": np.random.randn(n),
                "b": np.random.randn(n),
            }
        )

        result = mutual_info_matrix(features, random_state=42)

        # MI(a, b) should equal MI(b, a)
        a_b = result.filter(pl.col("feature") == "a")["b"].item()
        b_a = result.filter(pl.col("feature") == "b")["a"].item()
        assert a_b == pytest.approx(b_a, rel=0.01)


class TestMutualInfoPerFeature:
    """Tests for per-feature MI with independent validity masks."""

    def test_per_feature_scores_handle_warmup_nans_and_progress(self) -> None:
        rng = np.random.default_rng(42)
        n = 120
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x2[:20] = np.nan
        y = x1 + rng.normal(scale=0.05, size=n)
        progress: list[tuple[int, int]] = []

        result = mutual_info_scores_per_feature(
            pl.DataFrame({"x1": x1, "x2": x2}),
            pl.Series("y", y),
            min_samples=30,
            n_jobs=1,
            normalize=True,
            progress_callback=lambda current, total: progress.append((current, total)),
        )

        assert result.columns == ["feature", "mi_score", "n_samples"]
        assert result.filter(pl.col("feature") == "x1")["n_samples"].item() == n
        assert result.filter(pl.col("feature") == "x2")["n_samples"].item() == n - 20
        assert result["mi_score"].max() == pytest.approx(1.0)
        assert progress[-1] == (2, 2)

    def test_per_feature_scores_mark_small_samples_as_null_sorted_last(self) -> None:
        result = mutual_info_scores_per_feature(
            pl.DataFrame({"few": [1.0, np.nan, np.nan], "ok": [1.0, 2.0, 3.0]}),
            pl.Series("y", [1.0, 2.0, 3.0]),
            min_samples=3,
            n_jobs=1,
        )

        assert np.isnan(result.filter(pl.col("feature") == "few")["mi_score"].item())
        assert result.filter(pl.col("feature") == "few")["n_samples"].item() == 1
        assert set(result["feature"].to_list()) == {"few", "ok"}

    def test_per_feature_parallel_path_reports_results(self) -> None:
        rng = np.random.default_rng(7)
        n = 80
        data = {f"x{i}": rng.normal(size=n) for i in range(10)}
        target = pl.Series("y", data["x0"] + rng.normal(scale=0.1, size=n))

        result = mutual_info_scores_per_feature(
            pl.DataFrame(data), target, min_samples=20, n_jobs=2
        )

        assert result.height == 10
        assert set(result.columns) == {"feature", "mi_score", "n_samples"}

    def test_add_jitter_changes_tied_values_deterministically(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        first = _add_jitter(values, np.random.default_rng(1))
        second = _add_jitter(values, np.random.default_rng(1))

        assert first.tolist() == second.tolist()
        assert first.tolist() != values.tolist()

    def test_mutual_info_scores_without_dropping_nan_raises_from_sklearn(self) -> None:
        features = pl.DataFrame({"x": [1.0, np.nan, 3.0]})
        target = pl.Series("y", [1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            mutual_info_scores(features, target, drop_na=False)

    def test_mutual_info_matrix_raises_when_all_rows_drop(self) -> None:
        features = pl.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]})

        with pytest.raises(ValueError, match="No valid samples"):
            mutual_info_matrix(features)
