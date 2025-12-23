"""Rolling window temporal stability analysis.

Analyzes how stable MI-based feature rankings are over time using
rolling windows. This helps detect:
- Regime changes (sudden shifts in feature importance)
- Gradual drift in feature relationships
- Overall temporal stability of rankings

Key metrics:
- Adjacent window rank correlations
- Mean and minimum correlation
- Regime change detection (correlation drops)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from liq.features.validation.exceptions import (
    ConfigurationError,
    InsufficientDataError,
)
from liq.features.validation.results import TemporalStabilityResult

if TYPE_CHECKING:
    import polars as pl

# Minimum samples required per window
MIN_WINDOW_SAMPLES = 20


def rolling_mi_analysis(
    X: "pl.DataFrame",
    y: "pl.Series",
    features: list[str] | None = None,
    *,
    window_size: int,
    step_size: int,
    n_neighbors: int = 3,
    regime_threshold: float = 0.5,
    random_state: int | None = None,
) -> TemporalStabilityResult:
    """Analyze MI stability using rolling windows.

    Computes MI for each feature in each window and tracks how
    rankings correlate between adjacent windows.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        features: List of feature names. If None, uses all columns.
        window_size: Number of rows in each window.
        step_size: Number of rows to step between windows.
        n_neighbors: k-NN parameter for MI estimation.
        regime_threshold: Correlation below this indicates regime change.
        random_state: Random seed for reproducibility.

    Returns:
        TemporalStabilityResult with time series of MI and stability metrics.

    Raises:
        ConfigurationError: If window_size or step_size invalid.
        InsufficientDataError: If not enough data for windowing.
    """
    # Validate parameters
    if window_size <= 0:
        raise ConfigurationError(
            "window_size must be positive",
            parameter="window_size",
            value=window_size,
            valid_range="window_size > 0",
        )

    if step_size <= 0:
        raise ConfigurationError(
            "step_size must be positive",
            parameter="step_size",
            value=step_size,
            valid_range="step_size > 0",
        )

    # Get features
    if features is None:
        features = list(X.columns)

    n_samples = len(y)

    # Check if enough data
    if window_size > n_samples:
        raise InsufficientDataError(
            "window_size exceeds available data",
            required=window_size,
            actual=n_samples,
        )

    if window_size < MIN_WINDOW_SAMPLES:
        raise InsufficientDataError(
            f"window_size must be at least {MIN_WINDOW_SAMPLES}",
            required=MIN_WINDOW_SAMPLES,
            actual=window_size,
        )

    # Get numpy arrays
    y_arr = y.to_numpy()

    # Compute window starts
    window_starts = []
    start = 0
    while start + window_size <= n_samples:
        window_starts.append(start)
        start += step_size

    n_windows = len(window_starts)

    if n_windows == 0:
        raise InsufficientDataError(
            "No complete windows possible",
            required=window_size,
            actual=n_samples,
        )

    # Compute MI for each window
    mi_by_window: list[dict[str, float]] = []
    rank_by_window: list[dict[str, int]] = []

    for start_idx in window_starts:
        end_idx = start_idx + window_size

        # Get window data
        y_window = y_arr[start_idx:end_idx]

        mi_dict = {}
        for feature in features:
            x_arr = X[feature].to_numpy()
            x_window = x_arr[start_idx:end_idx]

            # Handle NaN
            valid = ~(np.isnan(x_window) | np.isnan(y_window))
            if valid.sum() >= n_neighbors + 1:
                mi = mutual_info_regression(
                    x_window[valid].reshape(-1, 1),
                    y_window[valid],
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                )[0]
            else:
                mi = 0.0

            mi_dict[feature] = float(mi)

        mi_by_window.append(mi_dict)

        # Compute rankings for this window (1 = highest MI)
        sorted_features = sorted(features, key=lambda f: mi_dict[f], reverse=True)
        rank_dict = {f: i + 1 for i, f in enumerate(sorted_features)}
        rank_by_window.append(rank_dict)

    # Compute adjacent window correlations
    adjacent_correlations = []

    for i in range(n_windows - 1):
        ranks_i = [rank_by_window[i][f] for f in features]
        ranks_next = [rank_by_window[i + 1][f] for f in features]

        if len(features) > 1:
            rho, _ = spearmanr(ranks_i, ranks_next)
            if np.isnan(rho):
                rho = 1.0  # Default to 1.0 if correlation undefined
        else:
            rho = 1.0  # Single feature always perfectly correlated

        adjacent_correlations.append(float(rho))

    # Compute summary statistics
    if adjacent_correlations:
        mean_correlation = float(np.mean(adjacent_correlations))
        min_correlation = float(np.min(adjacent_correlations))
    else:
        mean_correlation = 1.0
        min_correlation = 1.0

    # Detect regime changes
    regime_changes = []
    for i, corr in enumerate(adjacent_correlations):
        if corr < regime_threshold:
            regime_changes.append(i)

    return TemporalStabilityResult(
        features=features,
        window_size=window_size,
        step_size=step_size,
        n_windows=n_windows,
        window_starts=window_starts,
        mi_by_window=mi_by_window,
        rank_by_window=rank_by_window,
        adjacent_correlations=adjacent_correlations,
        mean_correlation=mean_correlation,
        min_correlation=min_correlation,
        regime_changes=regime_changes,
    )
