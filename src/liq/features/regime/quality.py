"""Shared cluster-quality metrics for regime bootstrap comparisons."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class ClusterQuality:
    """Intrinsic quality summary for one cluster labeling."""

    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    min_cluster_fraction: float
    n_singletons: int


def cluster_quality(
    labels: pl.Series,
    features: pl.DataFrame,
    *,
    silhouette_sample_size: int | None = None,
    random_state: int | None = None,
) -> ClusterQuality:
    """Score cluster separation and balance for one feature matrix."""
    if labels.len() != features.height:
        raise ValueError("labels and features row count must match")
    if silhouette_sample_size is not None and silhouette_sample_size < 2:
        raise ValueError("silhouette_sample_size must be at least 2")
    label_values = np.asarray(labels.to_list())
    counts = _cluster_counts(label_values)
    n_samples = len(label_values)
    n_labels = len(counts)
    metrics_are_valid = 2 <= n_labels <= n_samples - 1
    values = features.to_numpy().astype(np.float64, copy=False)
    sample_size = (
        min(silhouette_sample_size, n_samples) if silhouette_sample_size is not None else None
    )

    return ClusterQuality(
        silhouette=float(
            silhouette_score(
                values,
                label_values,
                sample_size=sample_size,
                random_state=random_state,
            )
        )
        if metrics_are_valid
        else None,
        davies_bouldin=float(davies_bouldin_score(values, label_values))
        if metrics_are_valid
        else None,
        calinski_harabasz=float(calinski_harabasz_score(values, label_values))
        if metrics_are_valid
        else None,
        min_cluster_fraction=min(counts.values()) / n_samples if n_samples else math.nan,
        n_singletons=sum(1 for count in counts.values() if count == 1),
    )


def cross_seed_stability(label_sets: Sequence[pl.Series]) -> float:
    """Return mean pairwise Adjusted Rand Index across seeded label sets."""
    if len(label_sets) < 2:
        return math.nan
    expected_length = label_sets[0].len()
    if any(labels.len() != expected_length for labels in label_sets):
        raise ValueError("all label sets must have the same length")

    scores: list[float] = []
    for left_index, left in enumerate(label_sets[:-1]):
        for right in label_sets[left_index + 1 :]:
            scores.append(float(adjusted_rand_score(left.to_list(), right.to_list())))
    return float(np.mean(scores))


def terminal_coverage(labels: pl.Series, *, expected_labels: Iterable[str]) -> dict[str, float]:
    """Return observed row fraction for each expected semantic regime label."""
    total = labels.len()
    counts = _cluster_counts(np.asarray(labels.to_list()))
    return {label: counts.get(label, 0) / total if total else 0.0 for label in expected_labels}


def _cluster_counts(values: np.ndarray) -> dict[object, int]:
    unique, counts = np.unique(values, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist(), strict=True))
