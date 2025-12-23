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

__all__ = [
    # MI scoring
    "mutual_info_scores",
    "mutual_info_scores_per_feature",
    # mRMR
    "mrmr_select",
    # Correlation
    "spearman_matrix",
    # Significance testing
    "bootstrap_mi",
    "permutation_test_mi",
    "paired_bootstrap_difference",
    "batch_bootstrap_mi",
    "batch_permutation_test",
    "batch_paired_difference",
    "apply_fdr_correction",
    "run_significance_analysis",
    # Result types
    "BootstrapResult",
    "PermutationResult",
    "PairedDifferenceResult",
    "SignificanceReport",
]
