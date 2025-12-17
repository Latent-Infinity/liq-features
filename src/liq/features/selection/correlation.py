"""Correlation utilities for feature diversity analysis.

Provides Spearman rank correlation for measuring redundancy between features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


def spearman_matrix(
    X: pl.DataFrame,
    *,
    drop_na: bool = True,
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Calculate Spearman rank correlation matrix between features.

    Spearman correlation measures monotonic relationships and is more robust
    to outliers than Pearson correlation. Useful for assessing feature redundancy.

    Args:
        X: DataFrame of features (each column is a feature).
        drop_na: If True, use pairwise complete observations.
        min_periods: Minimum number of observations required for each pair.
            If None, requires all observations to be non-null.

    Returns:
        Square DataFrame with Spearman correlation between each pair of features.
        Diagonal values are 1.0.

    Example:
        >>> features = pl.DataFrame({
        ...     "rsi_14": [30, 40, 50, 60, 70],
        ...     "rsi_21": [32, 42, 48, 58, 68],
        ...     "atr": [0.5, 0.6, 0.4, 0.7, 0.5],
        ... })
        >>> corr = spearman_matrix(features)
        >>> print(corr)
        shape: (3, 4)
        ┌─────────┬────────┬────────┬──────┐
        │ feature ┆ rsi_14 ┆ rsi_21 ┆ atr  │
        ├─────────┼────────┼────────┼──────┤
        │ rsi_14  ┆ 1.0    ┆ 0.98   ┆ 0.1  │
        │ rsi_21  ┆ 0.98   ┆ 1.0    ┆ 0.15 │
        │ atr     ┆ 0.1    ┆ 0.15   ┆ 1.0  │
        └─────────┴────────┴────────┴──────┘
    """
    from scipy.stats import spearmanr

    feature_names = X.columns
    n_features = len(feature_names)

    X_np: NDArray[np.floating] = X.to_numpy()

    if drop_na:
        # Use pairwise complete observations
        corr_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i, n_features):
                x_i = X_np[:, i]
                x_j = X_np[:, j]

                # Mask for pairwise complete
                mask = ~(np.isnan(x_i) | np.isnan(x_j))

                if min_periods is not None and mask.sum() < min_periods:
                    corr_matrix[i, j] = np.nan
                    corr_matrix[j, i] = np.nan
                elif mask.sum() < 2:
                    corr_matrix[i, j] = np.nan
                    corr_matrix[j, i] = np.nan
                else:
                    corr, _ = spearmanr(x_i[mask], x_j[mask])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
    else:
        # Require all observations
        mask = ~np.isnan(X_np).any(axis=1)
        if mask.sum() < 2:
            raise ValueError("Insufficient non-null observations for correlation")
        X_clean = X_np[mask]
        corr_matrix, _ = spearmanr(X_clean)

        # spearmanr returns scalar for 2 features
        if n_features == 2:
            corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])

    # Create DataFrame with feature names
    result = pl.DataFrame(
        corr_matrix,
        schema=list(feature_names),
    ).with_columns(pl.Series("feature", feature_names)).select(["feature", *feature_names])

    return result


def pearson_matrix(
    X: pl.DataFrame,
    *,
    drop_na: bool = True,
) -> pl.DataFrame:
    """Calculate Pearson correlation matrix between features.

    Args:
        X: DataFrame of features.
        drop_na: If True, drop rows with any NaN values.

    Returns:
        Square DataFrame with Pearson correlation between features.
    """
    feature_names = X.columns

    X_np: NDArray[np.floating] = X.to_numpy()

    if drop_na:
        mask = ~np.isnan(X_np).any(axis=1)
        X_np = X_np[mask]

    if len(X_np) < 2:
        raise ValueError("Insufficient non-null observations for correlation")

    corr_matrix = np.corrcoef(X_np, rowvar=False)

    result = pl.DataFrame(
        corr_matrix,
        schema=list(feature_names),
    ).with_columns(pl.Series("feature", feature_names)).select(["feature", *feature_names])

    return result


def highly_correlated_pairs(
    corr_matrix: pl.DataFrame,
    threshold: float = 0.8,
) -> pl.DataFrame:
    """Find pairs of features with correlation above threshold.

    Useful for identifying redundant features to remove.

    Args:
        corr_matrix: Correlation matrix from spearman_matrix() or pearson_matrix().
        threshold: Correlation threshold (absolute value).

    Returns:
        DataFrame with columns: feature_1, feature_2, correlation
        Only includes pairs where abs(correlation) >= threshold.
    """
    feature_names = [c for c in corr_matrix.columns if c != "feature"]

    pairs = []
    for i, f1 in enumerate(feature_names):
        for j, f2 in enumerate(feature_names):
            if j <= i:  # Only upper triangle, exclude diagonal
                continue

            corr_value = corr_matrix.filter(pl.col("feature") == f1)[f2].item()

            if corr_value is not None and abs(corr_value) >= threshold:
                pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": corr_value,
                })

    if not pairs:
        return pl.DataFrame(
            schema={"feature_1": pl.Utf8, "feature_2": pl.Utf8, "correlation": pl.Float64}
        )

    return pl.DataFrame(pairs).sort("correlation", descending=True)


def cluster_features(
    corr_matrix: pl.DataFrame,
    method: str = "average",
    metric: str = "euclidean",
) -> pl.DataFrame:
    """Hierarchically cluster features based on correlation.

    Args:
        corr_matrix: Correlation matrix from spearman_matrix().
        method: Linkage method ('single', 'complete', 'average', 'ward').
        metric: Distance metric.

    Returns:
        DataFrame with feature and cluster assignment.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    feature_names = [c for c in corr_matrix.columns if c != "feature"]

    # Convert correlation to distance (1 - abs(corr))
    corr_values = corr_matrix.select(feature_names).to_numpy()
    distance_matrix = 1 - np.abs(corr_values)

    # Ensure diagonal is 0
    np.fill_diagonal(distance_matrix, 0)

    # Make symmetric (in case of floating point issues)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Convert to condensed form for linkage
    condensed = squareform(distance_matrix)

    # Perform hierarchical clustering
    Z = linkage(condensed, method=method)

    # Cut at a distance that creates reasonable clusters
    # Default: cut where distance > 0.5 (correlation < 0.5)
    clusters = fcluster(Z, t=0.5, criterion="distance")

    return pl.DataFrame({
        "feature": feature_names,
        "cluster": clusters,
    }).sort("cluster")
