"""Tests for shared regime cluster-quality metrics."""

from __future__ import annotations

import math

import polars as pl
import pytest

from liq.features.regime import cluster_quality, cross_seed_stability, terminal_coverage


def _well_separated_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [-2.0, -2.1, -1.9, 0.0, 0.1, -0.1, 2.0, 2.1, 1.9],
            "y": [0.0, 0.1, -0.1, 2.0, 2.1, 1.9, 0.0, 0.1, -0.1],
        }
    )


def test_cluster_quality_scores_well_separated_clusters() -> None:
    features = _well_separated_features()
    labels = pl.Series("regime", [0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = cluster_quality(labels, features)

    assert result.silhouette is not None
    assert result.silhouette > 0.8
    assert result.davies_bouldin is not None
    assert result.davies_bouldin < 0.2
    assert result.calinski_harabasz is not None
    assert result.calinski_harabasz > 100
    assert result.min_cluster_fraction == pytest.approx(1 / 3)
    assert result.n_singletons == 0


def test_cluster_quality_flags_degenerate_singletons_without_invalid_metric_errors() -> None:
    features = _well_separated_features()
    labels = pl.Series("regime", [0, 0, 0, 0, 0, 0, 0, 1, 2])

    result = cluster_quality(labels, features)

    assert result.min_cluster_fraction == pytest.approx(1 / features.height)
    assert result.n_singletons == 2
    assert result.silhouette is not None
    assert result.davies_bouldin is not None
    assert result.calinski_harabasz is not None


def test_cluster_quality_returns_none_for_internal_metrics_when_one_label() -> None:
    features = _well_separated_features()
    labels = pl.Series("regime", [0] * features.height)

    result = cluster_quality(labels, features)

    assert result.silhouette is None
    assert result.davies_bouldin is None
    assert result.calinski_harabasz is None
    assert result.min_cluster_fraction == pytest.approx(1.0)
    assert result.n_singletons == 0


def test_cluster_quality_supports_sampled_silhouette_for_large_sweeps() -> None:
    features = _well_separated_features()
    labels = pl.Series("regime", [0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = cluster_quality(labels, features, silhouette_sample_size=6, random_state=7)

    assert result.silhouette is not None
    assert result.silhouette > 0.7
    assert result.davies_bouldin is not None
    assert result.calinski_harabasz is not None


def test_cluster_quality_validates_sampled_silhouette_arguments() -> None:
    features = _well_separated_features()
    labels = pl.Series("regime", [0, 0, 0, 1, 1, 1, 2, 2, 2])

    with pytest.raises(ValueError, match="silhouette_sample_size"):
        cluster_quality(labels, features, silhouette_sample_size=1, random_state=7)


def test_cluster_quality_validates_label_length() -> None:
    with pytest.raises(ValueError, match="row count"):
        cluster_quality(pl.Series("regime", [0, 1]), _well_separated_features())


def test_cross_seed_stability_returns_mean_pairwise_adjusted_rand_index() -> None:
    stability = cross_seed_stability(
        [
            pl.Series("a", [0, 0, 1, 1]),
            pl.Series("b", [1, 1, 0, 0]),
            pl.Series("c", [0, 1, 0, 1]),
        ]
    )

    assert -1.0 <= stability <= 1.0
    assert stability == pytest.approx((1.0 - 0.5 - 0.5) / 3)


def test_cross_seed_stability_requires_comparable_label_sets() -> None:
    assert math.isnan(cross_seed_stability([]))

    with pytest.raises(ValueError, match="same length"):
        cross_seed_stability([pl.Series("a", [0, 1]), pl.Series("b", [0])])


def test_terminal_coverage_reports_fraction_per_label() -> None:
    coverage = terminal_coverage(
        pl.Series("regime", ["trend", "range", "trend", "neutral"]),
        expected_labels=("trend", "range", "neutral", "volatile"),
    )

    assert coverage == {
        "trend": 0.5,
        "range": 0.25,
        "neutral": 0.25,
        "volatile": 0.0,
    }
