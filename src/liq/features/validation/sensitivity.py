"""MI estimator sensitivity analysis.

Tests robustness of MI estimates across different k-NN values.
A robust MI estimate should be relatively stable across k values.

The k-NN parameter in the KSG estimator controls the number of
neighbors used to estimate local density. Typical values are 3-7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from liq.features.validation.exceptions import ConfigurationError
from liq.features.validation.results import SensitivityResult

if TYPE_CHECKING:
    import polars as pl


def mi_sensitivity_analysis(
    X: "pl.DataFrame",
    y: "pl.Series",
    feature: str,
    *,
    k_values: list[int] | None = None,
    random_state: int | None = None,
) -> SensitivityResult:
    """Analyze MI sensitivity to k-NN parameter for a single feature.

    Computes MI at each k value and calculates statistics on the
    variability of the estimates.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        feature: Name of feature column to analyze.
        k_values: List of k-NN values to test. Default is [1, 3, 5, 7].
        random_state: Random seed for reproducibility.

    Returns:
        SensitivityResult with MI values at each k and stability metrics.

    Raises:
        ValueError: If feature not found or k values invalid.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7]

    # Validate k values
    for k in k_values:
        if k < 1:
            raise ConfigurationError(
                "k value must be positive",
                parameter="k_neighbors",
                value=k,
                valid_range="k >= 1",
            )

    # Extract feature column
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Get data as numpy arrays, handling NaN
    x_col = X[feature].to_numpy()
    y_arr = y.to_numpy()

    # Drop NaN rows
    valid_mask = ~(np.isnan(x_col) | np.isnan(y_arr))
    x_valid = x_col[valid_mask].reshape(-1, 1)
    y_valid = y_arr[valid_mask]

    # Compute MI at each k value
    mi_values = []
    for k in k_values:
        mi = mutual_info_regression(
            x_valid,
            y_valid,
            n_neighbors=k,
            random_state=random_state,
        )[0]
        mi_values.append(float(mi))

    # Compute statistics
    mean_mi = float(np.mean(mi_values))
    std_mi = float(np.std(mi_values, ddof=1)) if len(mi_values) > 1 else 0.0
    cv_mi = std_mi / mean_mi if mean_mi > 0 else 0.0

    # For single feature, rank is always 1 at each k
    rank_at_each = [1] * len(k_values)
    rank_correlation = 1.0  # Perfect correlation for single feature

    return SensitivityResult(
        feature=feature,
        parameter_name="k_neighbors",
        parameter_values=k_values,
        mi_values=mi_values,
        rank_at_each=rank_at_each,
        mean_mi=mean_mi,
        std_mi=std_mi,
        cv_mi=cv_mi,
        rank_correlation=rank_correlation,
    )


def batch_sensitivity_analysis(
    X: "pl.DataFrame",
    y: "pl.Series",
    features: list[str] | None = None,
    *,
    k_values: list[int] | None = None,
    random_state: int | None = None,
) -> dict[str, SensitivityResult]:
    """Analyze MI sensitivity for multiple features.

    Computes MI for each feature at each k value and tracks how
    rankings change across k values.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        features: List of feature names. If None, uses all columns.
        k_values: List of k-NN values to test. Default is [1, 3, 5, 7].
        random_state: Random seed for reproducibility.

    Returns:
        Dict mapping feature name to SensitivityResult.

    Raises:
        ValueError: If any feature not found.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7]

    # Validate k values
    for k in k_values:
        if k < 1:
            raise ConfigurationError(
                "k value must be positive",
                parameter="k_neighbors",
                value=k,
                valid_range="k >= 1",
            )

    if features is None:
        features = X.columns

    # Validate features
    for feature in features:
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Get target as numpy
    y_arr = y.to_numpy()

    # Compute MI for all features at each k value
    # Structure: mi_matrix[k_idx][feature_idx] = MI value
    mi_matrix = []

    for k in k_values:
        mi_row = []
        for feature in features:
            x_col = X[feature].to_numpy()

            # Handle NaN
            valid_mask = ~(np.isnan(x_col) | np.isnan(y_arr))
            x_valid = x_col[valid_mask].reshape(-1, 1)
            y_valid = y_arr[valid_mask]

            mi = mutual_info_regression(
                x_valid,
                y_valid,
                n_neighbors=k,
                random_state=random_state,
            )[0]
            mi_row.append(float(mi))

        mi_matrix.append(mi_row)

    mi_matrix = np.array(mi_matrix)  # Shape: (n_k, n_features)

    # Compute rankings at each k value (1 = highest MI)
    # Negative to sort descending
    rank_matrix = np.empty_like(mi_matrix, dtype=int)
    for k_idx in range(len(k_values)):
        rank_matrix[k_idx] = np.argsort(-mi_matrix[k_idx]).argsort() + 1

    # Build results for each feature
    results = {}
    n_features = len(features)

    for f_idx, feature in enumerate(features):
        mi_values = mi_matrix[:, f_idx].tolist()
        rank_at_each = rank_matrix[:, f_idx].tolist()

        mean_mi = float(np.mean(mi_values))
        std_mi = float(np.std(mi_values, ddof=1)) if len(mi_values) > 1 else 0.0
        cv_mi = std_mi / mean_mi if mean_mi > 0 else 0.0

        # Compute rank correlation across k values for this feature
        if len(k_values) > 1:
            # Compare rankings at adjacent k values
            rank_corrs = []
            for i in range(len(k_values) - 1):
                # Get rankings of all features at k[i] and k[i+1]
                ranks_k1 = rank_matrix[i, :]
                ranks_k2 = rank_matrix[i + 1, :]
                rho, _ = spearmanr(ranks_k1, ranks_k2)
                if not np.isnan(rho):
                    rank_corrs.append(rho)

            rank_correlation = float(np.mean(rank_corrs)) if rank_corrs else 1.0
        else:
            rank_correlation = 1.0

        results[feature] = SensitivityResult(
            feature=feature,
            parameter_name="k_neighbors",
            parameter_values=k_values,
            mi_values=mi_values,
            rank_at_each=rank_at_each,
            mean_mi=mean_mi,
            std_mi=std_mi,
            cv_mi=cv_mi,
            rank_correlation=rank_correlation,
        )

    return results
