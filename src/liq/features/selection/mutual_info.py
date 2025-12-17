"""Mutual Information calculation wrapper around sklearn.

Provides a Polars-friendly interface to sklearn's mutual_info_regression.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Default to all available cores
DEFAULT_N_JOBS = os.cpu_count() or 1


def mutual_info_scores(
    X: pl.DataFrame,
    y: pl.Series,
    *,
    n_neighbors: int = 3,
    random_state: int | None = 42,
    normalize: bool = False,
    drop_na: bool = True,
) -> pl.DataFrame:
    """Calculate Mutual Information between features and target.

    Wraps sklearn.feature_selection.mutual_info_regression to provide
    a Polars-native interface.

    Args:
        X: DataFrame of features (each column is a feature).
        y: Target Series (continuous values for regression).
        n_neighbors: Number of neighbors for MI estimation (default 3).
        random_state: Random seed for reproducibility.
        normalize: If True, normalize scores to [0, 1] range.
        drop_na: If True, drop rows with NaN values before calculation.

    Returns:
        DataFrame with columns: feature, mi_score
        Sorted by mi_score descending.

    Example:
        >>> features = pl.DataFrame({"rsi_14": [...], "macd": [...], "atr": [...]})
        >>> target = pl.Series("returns", [...])
        >>> scores = mutual_info_scores(features, target)
        >>> print(scores)
        shape: (3, 2)
        ┌─────────┬──────────┐
        │ feature ┆ mi_score │
        ├─────────┼──────────┤
        │ rsi_14  ┆ 0.15     │
        │ atr     ┆ 0.12     │
        │ macd    ┆ 0.08     │
        └─────────┴──────────┘
    """
    from sklearn.feature_selection import mutual_info_regression

    feature_names = X.columns

    # Convert to numpy, handling NaN values
    X_np: NDArray[np.floating] = X.to_numpy()
    y_np: NDArray[np.floating] = y.to_numpy()

    if drop_na:
        # Create mask for rows without NaN in either X or y
        mask = ~(np.isnan(X_np).any(axis=1) | np.isnan(y_np))
        X_np = X_np[mask]
        y_np = y_np[mask]

    if len(X_np) == 0:
        raise ValueError("No valid samples after dropping NaN values")

    # Calculate MI scores
    mi_scores: NDArray[np.floating] = mutual_info_regression(
        X_np,
        y_np,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    if normalize and mi_scores.max() > 0:
        mi_scores = mi_scores / mi_scores.max()

    # Create result DataFrame
    result = pl.DataFrame({
        "feature": feature_names,
        "mi_score": mi_scores,
    }).sort("mi_score", descending=True)

    return result


def _compute_mi_single_feature(
    col_name: str,
    x_col: "NDArray[np.floating]",
    y_np: "NDArray[np.floating]",
    n_neighbors: int,
    random_state: int | None,
    min_samples: int,
) -> dict:
    """Compute MI for a single feature (worker function for parallel execution).

    Args:
        col_name: Feature column name.
        x_col: Feature values as numpy array.
        y_np: Target values as numpy array.
        n_neighbors: Number of neighbors for MI estimation.
        random_state: Random seed for reproducibility.
        min_samples: Minimum valid samples required.

    Returns:
        Dict with feature, mi_score, n_samples.
    """
    from sklearn.feature_selection import mutual_info_regression

    # Create mask for valid rows (non-NaN, non-inf in both feature and target)
    mask = (
        ~np.isnan(x_col) & ~np.isnan(y_np) &
        ~np.isinf(x_col) & ~np.isinf(y_np)
    )
    x_valid = x_col[mask]
    y_valid = y_np[mask]

    n_samples = len(x_valid)

    if n_samples < min_samples:
        return {
            "feature": col_name,
            "mi_score": np.nan,
            "n_samples": n_samples,
        }

    try:
        mi_score = mutual_info_regression(
            x_valid.reshape(-1, 1),
            y_valid,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )[0]
    except Exception:
        mi_score = np.nan

    return {
        "feature": col_name,
        "mi_score": mi_score,
        "n_samples": n_samples,
    }


def mutual_info_scores_per_feature(
    X: pl.DataFrame,
    y: pl.Series,
    *,
    n_neighbors: int = 3,
    random_state: int | None = 42,
    normalize: bool = False,
    min_samples: int = 100,
    n_jobs: int | None = None,
) -> pl.DataFrame:
    """Calculate MI for each feature independently, handling NaN per-feature.

    Unlike mutual_info_scores which drops rows with NaN in ANY feature,
    this function calculates MI for each feature using only rows where
    that specific feature and target are both valid. This is essential
    when features have different warmup periods.

    Uses parallel processing for faster computation on multi-core systems.

    Args:
        X: DataFrame of features (each column is a feature).
        y: Target Series (continuous values for regression).
        n_neighbors: Number of neighbors for MI estimation (default 3).
        random_state: Random seed for reproducibility.
        normalize: If True, normalize scores to [0, 1] range.
        min_samples: Minimum valid samples required to calculate MI.
        n_jobs: Number of parallel workers. None = all cores.

    Returns:
        DataFrame with columns: feature, mi_score, n_samples
        Sorted by mi_score descending.

    Example:
        >>> features = pl.DataFrame({"rsi_14": [...], "ema_200": [...]})
        >>> target = pl.Series("returns", [...])
        >>> scores = mutual_info_scores_per_feature(features, target, n_jobs=8)
    """
    if n_jobs is None:
        n_jobs = DEFAULT_N_JOBS

    feature_names = X.columns
    y_np: NDArray[np.floating] = y.to_numpy()

    # For small number of features, sequential is faster due to process overhead
    if len(feature_names) < 10 or n_jobs == 1:
        results = []
        for col in feature_names:
            x_col = X[col].to_numpy()
            result = _compute_mi_single_feature(
                col, x_col, y_np, n_neighbors, random_state, min_samples
            )
            results.append(result)
    else:
        # Parallel execution for many features
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _compute_mi_single_feature,
                    col,
                    X[col].to_numpy(),
                    y_np,
                    n_neighbors,
                    random_state,
                    min_samples,
                ): col
                for col in feature_names
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    col_name = futures[future]
                    results.append({
                        "feature": col_name,
                        "mi_score": np.nan,
                        "n_samples": 0,
                    })

    result = pl.DataFrame(results)

    if normalize:
        max_score = result["mi_score"].max()
        if max_score is not None and max_score > 0:
            result = result.with_columns(
                (pl.col("mi_score") / max_score).alias("mi_score")
            )

    return result.sort("mi_score", descending=True, nulls_last=True)


def mutual_info_matrix(
    X: pl.DataFrame,
    *,
    n_neighbors: int = 3,
    random_state: int | None = 42,
    drop_na: bool = True,
) -> pl.DataFrame:
    """Calculate pairwise Mutual Information between all features.

    Useful for understanding feature redundancy.

    Args:
        X: DataFrame of features.
        n_neighbors: Number of neighbors for MI estimation.
        random_state: Random seed for reproducibility.
        drop_na: If True, drop rows with NaN values.

    Returns:
        Square DataFrame with MI values between each pair of features.
    """
    from sklearn.feature_selection import mutual_info_regression

    feature_names = X.columns
    n_features = len(feature_names)

    X_np: NDArray[np.floating] = X.to_numpy()

    if drop_na:
        mask = ~np.isnan(X_np).any(axis=1)
        X_np = X_np[mask]

    if len(X_np) == 0:
        raise ValueError("No valid samples after dropping NaN values")

    # Calculate pairwise MI
    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        mi_values = mutual_info_regression(
            X_np,
            X_np[:, i],
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        mi_matrix[i, :] = mi_values

    # Make symmetric by averaging
    mi_matrix = (mi_matrix + mi_matrix.T) / 2

    # Create DataFrame
    result = pl.DataFrame(
        mi_matrix,
        schema=list(feature_names),
    ).with_columns(pl.Series("feature", feature_names)).select(["feature", *feature_names])

    return result
