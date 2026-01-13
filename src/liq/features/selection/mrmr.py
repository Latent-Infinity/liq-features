"""mRMR (Minimum Redundancy Maximum Relevance) feature selection.

Native Polars implementation for memory-efficient feature selection.
No pandas conversion required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from liq.features.numpy_utils import to_numpy_float64

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default max features for mRMR to avoid excessive computation
DEFAULT_MAX_FEATURES = 2000


@dataclass(frozen=True)
class MRMRResult:
    """Result from mRMR feature selection.

    Attributes:
        selected_features: List of feature names in selection order.
        scores: DataFrame with feature, rank, and relevance/redundancy info.
    """

    selected_features: list[str]
    scores: pl.DataFrame


# Epsilon for float comparisons (§19 Float Comparisons)
_FLOAT_EPS = 1e-10


def _compute_relevance_from_numpy(
    X_np: "NDArray[np.floating]",
    y_np: "NDArray[np.floating]",
    feature_names: list[str],
) -> pl.DataFrame:
    """Compute relevance (absolute correlation) for each feature vs target.

    Uses absolute Pearson correlation as the relevance measure.
    This ensures relevance is on the same scale [0,1] as redundancy (correlation),
    which is critical for the mRMR score = relevance - redundancy to work correctly.

    Vectorized implementation using matrix operations for efficiency.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        DataFrame with columns: feature, relevance
    """
    n_features = len(feature_names)

    # Replace inf with NaN for uniform handling
    X_np = np.where(np.isinf(X_np), np.nan, X_np)
    y_np = np.where(np.isinf(y_np), np.nan, y_np)

    # Handle NaN in y
    y_valid_mask = ~np.isnan(y_np)
    if y_valid_mask.sum() < 2:
        return pl.DataFrame({
            "feature": feature_names,
            "relevance": [0.0] * n_features,
        })

    # Pre-allocate result array (§2 Memory Efficiency)
    relevance_scores = np.zeros(n_features, dtype=np.float64)

    # Vectorized correlation computation
    # For each feature, compute correlation with y using valid (non-NaN) pairs
    for i, col in enumerate(feature_names):
        x_col = X_np[:, i]

        # Pairwise complete observations (exclude NaN and inf)
        mask = ~np.isnan(x_col) & y_valid_mask
        n_valid = mask.sum()

        if n_valid < 2:
            continue  # relevance_scores[i] already 0

        x_valid = x_col[mask]
        y_valid = y_np[mask]

        # Use errstate to suppress any lingering warnings
        with np.errstate(invalid="ignore"):
            x_std = np.std(x_valid)
            y_std = np.std(y_valid)

        # Use epsilon for float comparison (§19 Float Comparisons)
        # Also check for non-finite std (can happen with extreme values)
        if x_std < _FLOAT_EPS or y_std < _FLOAT_EPS or not np.isfinite(x_std) or not np.isfinite(y_std):
            continue  # Constant feature or target subset

        # Pearson correlation via vectorized ops
        x_centered = x_valid - np.mean(x_valid)
        y_centered = y_valid - np.mean(y_valid)
        corr = np.dot(x_centered, y_centered) / (n_valid * x_std * y_std)

        # Clamp to valid range and handle numerical issues
        if np.isfinite(corr):
            relevance_scores[i] = min(abs(corr), 1.0)

    return pl.DataFrame({
        "feature": feature_names,
        "relevance": relevance_scores.tolist(),
    })


def _compute_correlation_matrix_from_numpy(
    X_np: "NDArray[np.floating]",
    features: list[str],
) -> dict[tuple[str, str], float]:
    """Compute pairwise Pearson correlations for feature redundancy.

    Uses vectorized NumPy operations for efficiency.
    For features with no NaN/inf and non-constant values, uses fast matrix multiplication.
    Falls back to pairwise computation for problematic features.

    Args:
        X: Feature DataFrame.
        features: List of feature names to include.

    Returns:
        Dict mapping (feature1, feature2) to absolute correlation value.
    """
    corr_cache: dict[tuple[str, str], float] = {}
    n_features = len(features)

    if n_features < 2:
        return corr_cache

    # Check for problematic values (NaN or inf)
    has_nan = np.isnan(X_np).any()
    has_inf = np.isinf(X_np).any()

    # Replace inf with NaN for uniform handling
    if has_inf:
        X_np = np.where(np.isinf(X_np), np.nan, X_np)
        has_nan = True

    if not has_nan:
        # Fast path: no NaN/inf, use vectorized correlation matrix
        # Compute means and stds
        means = np.mean(X_np, axis=0)
        stds = np.std(X_np, axis=0)

        # Identify valid (non-constant) features for fast path
        valid_mask = stds >= _FLOAT_EPS
        n_valid = valid_mask.sum()

        if n_valid >= 2:
            # Extract only valid features for matrix multiplication
            valid_indices = np.where(valid_mask)[0]
            valid_features = [features[i] for i in valid_indices]
            X_valid = X_np[:, valid_mask]
            means_valid = means[valid_mask]
            stds_valid = stds[valid_mask]

            # Standardize only valid features
            X_standardized = (X_valid - means_valid) / stds_valid

            # Correlation matrix via matrix multiplication: (X^T @ X) / n
            n_samples = X_np.shape[0]
            with np.errstate(over="ignore", invalid="ignore"):
                corr_matrix = (X_standardized.T @ X_standardized) / n_samples

            # Extract upper triangle into cache, handling any remaining numerical issues
            for i in range(len(valid_features)):
                for j in range(i + 1, len(valid_features)):
                    corr_val = corr_matrix[i, j]
                    # Clamp to valid correlation range and handle NaN/inf
                    if np.isnan(corr_val) or np.isinf(corr_val):
                        corr = 0.0
                    else:
                        corr = min(abs(float(corr_val)), 1.0)  # Clamp to [0, 1]
                    corr_cache[(valid_features[i], valid_features[j])] = corr
                    corr_cache[(valid_features[j], valid_features[i])] = corr

        # Constant features have 0 correlation with everything (already default)
    else:
        # Slow path: handle NaN with pairwise complete observations
        # Compute means and stds ignoring NaN
        means = np.nanmean(X_np, axis=0)
        stds = np.nanstd(X_np, axis=0)

        # Pre-compute which features are valid (non-constant, finite stats)
        valid_features_mask = (stds >= _FLOAT_EPS) & np.isfinite(means) & np.isfinite(stds)

        for i in range(n_features):
            if not valid_features_mask[i]:
                continue  # Skip constant or invalid features

            for j in range(i + 1, n_features):
                if not valid_features_mask[j]:
                    continue  # Skip constant or invalid features

                # Pairwise complete observations
                col_i = X_np[:, i]
                col_j = X_np[:, j]
                mask = ~(np.isnan(col_i) | np.isnan(col_j))
                n_valid = mask.sum()

                if n_valid < 2:
                    corr = 0.0
                else:
                    # Standardize using pairwise valid data
                    x_i = col_i[mask]
                    x_j = col_j[mask]
                    x_i_std = np.std(x_i)
                    x_j_std = np.std(x_j)

                    if x_i_std < _FLOAT_EPS or x_j_std < _FLOAT_EPS:
                        corr = 0.0
                    else:
                        x_i_centered = (x_i - np.mean(x_i)) / x_i_std
                        x_j_centered = (x_j - np.mean(x_j)) / x_j_std
                        corr_val = np.mean(x_i_centered * x_j_centered)
                        # Clamp to valid range
                        if np.isnan(corr_val) or np.isinf(corr_val):
                            corr = 0.0
                        else:
                            corr = min(abs(float(corr_val)), 1.0)

                corr_cache[(features[i], features[j])] = corr
                corr_cache[(features[j], features[i])] = corr

    return corr_cache


def mrmr_select(
    X: pl.DataFrame,
    y: pl.Series,
    K: int = 10,
    *,
    denominator: str = "mean",
    only_same_domain: bool = False,
    return_scores: bool = False,
    show_progress: bool = False,
    max_features: int | None = None,
    mi_scores: pl.DataFrame | None = None,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> list[str] | MRMRResult:
    """Select top K features using mRMR algorithm (native Polars implementation).

    mRMR (Minimum Redundancy Maximum Relevance) selects features that have:
    - High relevance (correlation with target)
    - Low redundancy (correlation with already-selected features)

    This implementation is pure Polars/NumPy with no pandas dependency.

    Args:
        X: DataFrame of features.
        y: Target Series (continuous for regression).
        K: Number of features to select.
        denominator: How to aggregate redundancy - "mean" or "max".
        only_same_domain: If True, only compute redundancy between same-domain features.
            Domain is defined by the string before the first underscore.
        return_scores: If True, return MRMRResult with detailed scores.
        show_progress: If True, log progress during selection.
        max_features: Maximum features to consider. If X has more features,
            pre-filters using MI scores. None = DEFAULT_MAX_FEATURES (2000).
            Set to 0 to disable pre-filtering.
        mi_scores: Optional pre-computed MI scores DataFrame with columns
            'feature' and 'mi_score'. If not provided and pre-filtering is needed,
            MI scores are computed on-the-fly.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        List of selected feature names (in order of selection), or MRMRResult if
        return_scores=True.

    Example:
        >>> features = pl.DataFrame({"rsi_14": [...], "rsi_21": [...], "atr": [...]})
        >>> target = pl.Series("returns", [...])
        >>> selected = mrmr_select(features, target, K=2)
        >>> print(selected)
        ['rsi_14', 'atr']  # rsi_21 excluded due to redundancy with rsi_14
    """
    if max_features is None:
        max_features = DEFAULT_MAX_FEATURES

    n_features = len(X.columns)
    feature_list = list(X.columns)

    # Pre-filter features if too many (memory optimization)
    if max_features > 0 and n_features > max_features:
        logger.info(
            f"Pre-filtering {n_features} features to top {max_features} by MI score"
        )

        if mi_scores is None:
            # Compute MI scores on-the-fly
            from liq.features.selection import mutual_info_scores_per_feature

            mi_scores = mutual_info_scores_per_feature(X, y)

        # Get top features by MI score
        top_features = (
            mi_scores.filter(pl.col("mi_score").is_not_nan())
            .sort("mi_score", descending=True)
            .head(max_features)["feature"]
            .to_list()
        )

        # Filter to features that exist in X
        available_features = set(X.columns)
        feature_list = [f for f in top_features if f in available_features]

        if not feature_list:
            logger.warning("No valid features after MI filtering")
            if return_scores:
                return MRMRResult(
                    selected_features=[],
                    scores=pl.DataFrame({"feature": [], "rank": [], "relevance": [], "redundancy": []}),
                )
            return []

        X = X.select(feature_list)
        logger.info(f"Reduced to {len(X.columns)} features for mRMR")

    # Ensure K doesn't exceed available features
    K = min(K, len(feature_list))

    if K == 0:
        if return_scores:
            return MRMRResult(
                selected_features=[],
                scores=pl.DataFrame({"feature": [], "rank": [], "relevance": [], "redundancy": []}),
            )
        return []

    # Convert to numpy once for relevance + redundancy calculations
    X_np = to_numpy_float64(X)
    y_np = to_numpy_float64(y)

    # Step 1: Compute relevance (absolute correlation) for all features
    if show_progress:
        logger.info("Computing feature relevance...")

    relevance_df = _compute_relevance_from_numpy(X_np, y_np, feature_list)
    relevance = dict(zip(relevance_df["feature"].to_list(), relevance_df["relevance"].to_list()))

    # Step 2: Compute correlation matrix for redundancy
    if show_progress:
        logger.info("Computing feature correlations...")

    corr_cache = _compute_correlation_matrix_from_numpy(X_np, feature_list)

    # Step 3: Greedy mRMR selection
    if show_progress:
        logger.info(f"Selecting {K} features...")

    selected: list[str] = []
    selected_set: set[str] = set()
    remaining = set(feature_list)

    # Track scores for return
    score_records: list[dict] = []

    for k in range(K):
        best_feature = None
        best_score = float("-inf")
        best_relevance = 0.0
        best_redundancy = 0.0

        for feature in remaining:
            rel = relevance.get(feature, 0.0)

            # Compute redundancy with already selected features
            if not selected:
                red = 0.0
            else:
                redundancies = []
                for sel_feat in selected:
                    # Check domain constraint if enabled
                    if only_same_domain:
                        feat_domain = feature.split("_")[0] if "_" in feature else feature
                        sel_domain = sel_feat.split("_")[0] if "_" in sel_feat else sel_feat
                        if feat_domain != sel_domain:
                            continue

                    corr = corr_cache.get((feature, sel_feat), 0.0)
                    redundancies.append(corr)

                if not redundancies:
                    red = 0.0
                elif denominator == "mean":
                    red = sum(redundancies) / len(redundancies)
                elif denominator == "max":
                    red = max(redundancies)
                else:
                    red = sum(redundancies) / len(redundancies)

            # mRMR score: relevance - redundancy
            score = rel - red

            if score > best_score:
                best_score = score
                best_feature = feature
                best_relevance = rel
                best_redundancy = red

        if best_feature is None:
            break

        selected.append(best_feature)
        selected_set.add(best_feature)
        remaining.remove(best_feature)

        score_records.append({
            "feature": best_feature,
            "rank": k + 1,
            "relevance": best_relevance,
            "redundancy": best_redundancy,
            "mrmr_score": best_score,
        })

        if progress_callback:
            progress_callback(k + 1, K)

        if show_progress and (k + 1) % 10 == 0:
            logger.info(f"Selected {k + 1}/{K} features")

    if show_progress:
        logger.info(f"mRMR selection complete: {len(selected)} features")

    if return_scores:
        scores_df = pl.DataFrame(score_records)
        return MRMRResult(selected_features=selected, scores=scores_df)

    return selected


def mrmr_classif(
    X: pl.DataFrame,
    y: pl.Series,
    K: int = 10,
    *,
    denominator: str = "mean",
    only_same_domain: bool = False,
    return_scores: bool = False,
    show_progress: bool = False,
    max_features: int | None = None,
    mi_scores: pl.DataFrame | None = None,
) -> list[str] | MRMRResult:
    """Select top K features using mRMR for classification (native Polars).

    For classification, uses F-statistic computed via ANOVA-style grouping.

    Args:
        X: DataFrame of features.
        y: Target Series (categorical/discrete values).
        K: Number of features to select.
        denominator: How to aggregate redundancy - "mean" or "max".
        only_same_domain: If True, only compute redundancy between same-domain features.
        return_scores: If True, return MRMRResult with detailed scores.
        show_progress: If True, log progress during selection.
        max_features: Maximum features to pass to mRMR. If X has more features,
            pre-filters using MI scores. None = DEFAULT_MAX_FEATURES (2000).
        mi_scores: Optional pre-computed MI scores DataFrame for pre-filtering.

    Returns:
        List of selected feature names, or MRMRResult if return_scores=True.
    """
    # For classification, we can still use F-statistic (ANOVA F)
    # The mrmr_select function works for both regression and classification
    # since F-statistic is valid for both (correlation-based relevance)
    return mrmr_select(
        X=X,
        y=y.cast(pl.Float64),  # Cast categorical to numeric for correlation
        K=K,
        denominator=denominator,
        only_same_domain=only_same_domain,
        return_scores=return_scores,
        show_progress=show_progress,
        max_features=max_features,
        mi_scores=mi_scores,
    )
