"""mRMR (Minimum Redundancy Maximum Relevance) feature selection wrapper.

Wraps the mrmr-selection library for Polars DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class MRMRResult:
    """Result from mRMR feature selection.

    Attributes:
        selected_features: List of feature names in selection order.
        scores: DataFrame with feature, rank, and relevance/redundancy info.
    """

    selected_features: list[str]
    scores: pl.DataFrame


def mrmr_select(
    X: pl.DataFrame,
    y: pl.Series,
    K: int = 10,
    *,
    relevance: str = "f",
    redundancy: str = "c",
    denominator: str = "mean",
    cat_encoding: str = "leave_one_out",
    only_same_domain: bool = False,
    return_scores: bool = False,
    show_progress: bool = False,
) -> list[str] | MRMRResult:
    """Select top K features using mRMR algorithm.

    mRMR selects features that have high relevance (correlation/MI with target)
    while minimizing redundancy (correlation with already-selected features).

    Args:
        X: DataFrame of features.
        y: Target Series (continuous for regression, categorical for classification).
        K: Number of features to select.
        relevance: Relevance method - "f" (F-statistic) or "rf" (random forest).
        redundancy: Redundancy method - "c" (Pearson correlation).
        denominator: How to aggregate redundancy - "mean" or "max".
        cat_encoding: Encoding for categorical features.
        only_same_domain: If True, only compute redundancy between same-domain features.
        return_scores: If True, return MRMRResult with detailed scores.
        show_progress: If True, show progress bar during selection.

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
    try:
        from mrmr import mrmr_regression
    except ImportError as e:
        raise ImportError(
            "mrmr-selection package required. Install with: pip install mrmr-selection"
        ) from e

    # mrmr expects pandas, convert from polars
    X_pd = X.to_pandas()
    y_pd = y.to_pandas()

    # Ensure K doesn't exceed available features
    K = min(K, len(X.columns))

    # Run mRMR selection
    selected = mrmr_regression(
        X=X_pd,
        y=y_pd,
        K=K,
        relevance=relevance,
        redundancy=redundancy,
        denominator=denominator,
        cat_encoding=cat_encoding,
        only_same_domain=only_same_domain,
        show_progress=show_progress,
    )

    if not return_scores:
        return list(selected)

    # Build scores DataFrame with selection order
    scores = pl.DataFrame({
        "feature": selected,
        "rank": list(range(1, len(selected) + 1)),
    })

    return MRMRResult(selected_features=list(selected), scores=scores)


def mrmr_classif(
    X: pl.DataFrame,
    y: pl.Series,
    K: int = 10,
    *,
    relevance: str = "f",
    redundancy: str = "c",
    denominator: str = "mean",
    cat_encoding: str = "leave_one_out",
    only_same_domain: bool = False,
    return_scores: bool = False,
    show_progress: bool = False,
) -> list[str] | MRMRResult:
    """Select top K features using mRMR for classification tasks.

    Same as mrmr_select but uses classification-appropriate relevance metrics.

    Args:
        X: DataFrame of features.
        y: Target Series (categorical/discrete values).
        K: Number of features to select.
        relevance: Relevance method - "f" (F-statistic) or "rf" (random forest).
        redundancy: Redundancy method - "c" (Pearson correlation).
        denominator: How to aggregate redundancy - "mean" or "max".
        cat_encoding: Encoding for categorical features.
        only_same_domain: If True, only compute redundancy between same-domain features.
        return_scores: If True, return MRMRResult with detailed scores.
        show_progress: If True, show progress bar during selection.

    Returns:
        List of selected feature names, or MRMRResult if return_scores=True.
    """
    try:
        from mrmr import mrmr_classif as _mrmr_classif
    except ImportError as e:
        raise ImportError(
            "mrmr-selection package required. Install with: pip install mrmr-selection"
        ) from e

    X_pd = X.to_pandas()
    y_pd = y.to_pandas()

    K = min(K, len(X.columns))

    selected = _mrmr_classif(
        X=X_pd,
        y=y_pd,
        K=K,
        relevance=relevance,
        redundancy=redundancy,
        denominator=denominator,
        cat_encoding=cat_encoding,
        only_same_domain=only_same_domain,
        show_progress=show_progress,
    )

    if not return_scores:
        return list(selected)

    scores = pl.DataFrame({
        "feature": selected,
        "rank": list(range(1, len(selected) + 1)),
    })

    return MRMRResult(selected_features=list(selected), scores=scores)
