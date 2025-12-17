"""Tests for correlation utilities."""

import numpy as np
import polars as pl
import pytest

from liq.features.selection.correlation import (
    cluster_features,
    highly_correlated_pairs,
    pearson_matrix,
    spearman_matrix,
)


class TestSpearmanMatrix:
    """Tests for Spearman correlation matrix."""

    def test_basic_calculation(self) -> None:
        """Test basic Spearman correlation matrix."""
        features = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfectly correlated with a
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfectly anti-correlated with a
        })

        result = spearman_matrix(features)

        # Check format
        assert "feature" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert len(result) == 3

        # a-b should be 1.0 (perfect correlation)
        a_b = result.filter(pl.col("feature") == "a")["b"].item()
        assert a_b == pytest.approx(1.0, rel=1e-6)

        # a-c should be -1.0 (perfect anti-correlation)
        a_c = result.filter(pl.col("feature") == "a")["c"].item()
        assert a_c == pytest.approx(-1.0, rel=1e-6)

    def test_diagonal_is_one(self) -> None:
        """Test diagonal values are 1.0."""
        np.random.seed(42)
        features = pl.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
        })

        result = spearman_matrix(features)

        # Diagonal should be 1.0
        x_x = result.filter(pl.col("feature") == "x")["x"].item()
        y_y = result.filter(pl.col("feature") == "y")["y"].item()
        assert x_x == pytest.approx(1.0, rel=1e-6)
        assert y_y == pytest.approx(1.0, rel=1e-6)

    def test_symmetric_matrix(self) -> None:
        """Test matrix is symmetric."""
        np.random.seed(42)
        features = pl.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
        })

        result = spearman_matrix(features)

        # Check symmetry
        a_b = result.filter(pl.col("feature") == "a")["b"].item()
        b_a = result.filter(pl.col("feature") == "b")["a"].item()
        assert a_b == pytest.approx(b_a, rel=1e-6)

        a_c = result.filter(pl.col("feature") == "a")["c"].item()
        c_a = result.filter(pl.col("feature") == "c")["a"].item()
        assert a_c == pytest.approx(c_a, rel=1e-6)

    def test_handles_nan_pairwise(self) -> None:
        """Test NaN handling with pairwise deletion."""
        features = pl.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = spearman_matrix(features, drop_na=True)

        # Should still produce valid correlations
        a_b = result.filter(pl.col("feature") == "a")["b"].item()
        assert not np.isnan(a_b)
        assert a_b == pytest.approx(1.0, rel=1e-6)

    def test_independent_variables_low_correlation(self) -> None:
        """Test independent variables have correlation near 0."""
        np.random.seed(42)
        n = 1000

        features = pl.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n),
        })

        result = spearman_matrix(features)

        x_y = result.filter(pl.col("feature") == "x")["y"].item()
        assert abs(x_y) < 0.1  # Should be close to 0


class TestPearsonMatrix:
    """Tests for Pearson correlation matrix."""

    def test_basic_calculation(self) -> None:
        """Test basic Pearson correlation."""
        features = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # Linear relationship
        })

        result = pearson_matrix(features)

        a_b = result.filter(pl.col("feature") == "a")["b"].item()
        assert a_b == pytest.approx(1.0, rel=1e-6)


class TestHighlyCorrelatedPairs:
    """Tests for highly_correlated_pairs function."""

    def test_finds_correlated_pairs(self) -> None:
        """Test finding highly correlated pairs."""
        # Create correlation matrix
        corr = pl.DataFrame({
            "feature": ["a", "b", "c"],
            "a": [1.0, 0.95, 0.1],
            "b": [0.95, 1.0, 0.2],
            "c": [0.1, 0.2, 1.0],
        })

        result = highly_correlated_pairs(corr, threshold=0.8)

        # Should find a-b pair
        assert len(result) == 1
        assert result["feature_1"].item() == "a"
        assert result["feature_2"].item() == "b"
        assert result["correlation"].item() == pytest.approx(0.95, rel=1e-6)

    def test_no_pairs_below_threshold(self) -> None:
        """Test returns empty when no pairs above threshold."""
        corr = pl.DataFrame({
            "feature": ["a", "b"],
            "a": [1.0, 0.3],
            "b": [0.3, 1.0],
        })

        result = highly_correlated_pairs(corr, threshold=0.8)
        assert len(result) == 0

    def test_negative_correlation(self) -> None:
        """Test finds negatively correlated pairs."""
        corr = pl.DataFrame({
            "feature": ["a", "b"],
            "a": [1.0, -0.9],
            "b": [-0.9, 1.0],
        })

        result = highly_correlated_pairs(corr, threshold=0.8)
        assert len(result) == 1
        assert result["correlation"].item() == pytest.approx(-0.9, rel=1e-6)


class TestClusterFeatures:
    """Tests for hierarchical clustering of features."""

    def test_basic_clustering(self) -> None:
        """Test basic feature clustering."""
        # Create correlation matrix with clear clusters
        corr = pl.DataFrame({
            "feature": ["a1", "a2", "b1", "b2"],
            "a1": [1.0, 0.9, 0.1, 0.1],
            "a2": [0.9, 1.0, 0.1, 0.1],
            "b1": [0.1, 0.1, 1.0, 0.9],
            "b2": [0.1, 0.1, 0.9, 1.0],
        })

        result = cluster_features(corr)

        # Should have 4 features
        assert len(result) == 4
        assert "feature" in result.columns
        assert "cluster" in result.columns

        # a1 and a2 should be in same cluster
        a1_cluster = result.filter(pl.col("feature") == "a1")["cluster"].item()
        a2_cluster = result.filter(pl.col("feature") == "a2")["cluster"].item()
        assert a1_cluster == a2_cluster

        # b1 and b2 should be in same cluster
        b1_cluster = result.filter(pl.col("feature") == "b1")["cluster"].item()
        b2_cluster = result.filter(pl.col("feature") == "b2")["cluster"].item()
        assert b1_cluster == b2_cluster

        # a and b groups should be in different clusters
        assert a1_cluster != b1_cluster
