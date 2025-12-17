"""Feature selection utilities for the LIQ stack.

This module provides wrappers around industry-standard feature selection algorithms:
- Mutual Information (sklearn)
- mRMR - Minimum Redundancy Maximum Relevance (mrmr-selection)
- Spearman correlation matrix (scipy/polars)

Example:
    >>> import polars as pl
    >>> from liq.features.selection import mutual_info_scores, mrmr_select, spearman_matrix
    >>>
    >>> # Compute MI scores for all features against target
    >>> mi_scores = mutual_info_scores(features_df, target_series)
    >>>
    >>> # Select top K features using mRMR
    >>> selected = mrmr_select(features_df, target_series, K=20)
    >>>
    >>> # Compute correlation matrix between features
    >>> corr_matrix = spearman_matrix(features_df)
"""

from liq.features.selection.correlation import spearman_matrix
from liq.features.selection.mrmr import mrmr_select
from liq.features.selection.mutual_info import (
    mutual_info_scores,
    mutual_info_scores_per_feature,
)

__all__ = [
    "mutual_info_scores",
    "mutual_info_scores_per_feature",
    "mrmr_select",
    "spearman_matrix",
]
