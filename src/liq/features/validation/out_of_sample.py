"""Out-of-sample validation for MI feature rankings.

Validates that MI-based feature rankings are stable when measured on
held-out test data. This is critical for ensuring that top-ranked
features generalize beyond the training set.

Key metrics:
- Spearman correlation between train/test rankings
- Top-K feature overlap
- Individual feature MI stability
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
from liq.features.validation.logging_config import (
    get_logger,
    log_function_entry,
    log_function_exit,
    log_result,
)
from liq.features.validation.results import OutOfSampleResult

if TYPE_CHECKING:
    import polars as pl

logger = get_logger("out_of_sample")

# Minimum samples required per split for reliable MI estimation
MIN_SAMPLES_PER_SPLIT = 10


def validate_oos(
    X: "pl.DataFrame",
    y: "pl.Series",
    features: list[str] | None = None,
    *,
    test_ratio: float = 0.2,
    temporal: bool = True,
    n_neighbors: int = 3,
    random_state: int | None = None,
) -> OutOfSampleResult:
    """Validate MI rankings using train/test split.

    Computes MI on both training and test sets and measures how well
    the feature rankings correlate between the two.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        features: List of feature names to validate. If None, uses all columns.
        test_ratio: Proportion of data for test set (0 < test_ratio < 1).
        temporal: If True, use temporal split (first n% train, last m% test).
                  If False, shuffle before splitting.
        n_neighbors: k-NN parameter for MI estimation.
        random_state: Random seed for reproducibility (used when temporal=False).

    Returns:
        OutOfSampleResult with train/test MI values, rankings, and correlation.

    Raises:
        ConfigurationError: If test_ratio is invalid.
        InsufficientDataError: If not enough samples for split.
        ValueError: If features not found in DataFrame.
    """
    log_function_entry(
        logger, "validate_oos",
        n_features=len(features) if features else "all",
        test_ratio=test_ratio, temporal=temporal,
    )

    # Validate test_ratio
    if test_ratio <= 0 or test_ratio >= 1:
        raise ConfigurationError(
            "test_ratio must be between 0 and 1 (exclusive)",
            parameter="test_ratio",
            value=test_ratio,
            valid_range="0 < test_ratio < 1",
        )

    # Get features
    if features is None:
        features = list(X.columns)

    # Validate features exist
    for feature in features:
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Get data as numpy arrays
    n_samples = len(y)
    y_arr = y.to_numpy()

    # Check sample size
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    if n_train < MIN_SAMPLES_PER_SPLIT or n_test < MIN_SAMPLES_PER_SPLIT:
        raise InsufficientDataError(
            f"Insufficient samples for split (need at least {MIN_SAMPLES_PER_SPLIT} per split)",
            required=MIN_SAMPLES_PER_SPLIT * 2,
            actual=n_samples,
            context={"n_train": n_train, "n_test": n_test},
        )

    # Create split indices
    if temporal:
        # Temporal split: first n_train for training, last n_test for testing
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_samples)
    else:
        # Shuffled split
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

    # Split target
    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    # Compute MI for each feature on train and test
    train_mi = {}
    test_mi = {}

    for feature in features:
        x_arr = X[feature].to_numpy()

        # Handle NaN - drop rows with NaN for each feature independently
        x_train = x_arr[train_idx]
        x_test = x_arr[test_idx]

        # Train set
        valid_train = ~(np.isnan(x_train) | np.isnan(y_train))
        if valid_train.sum() >= n_neighbors + 1:
            mi_train = mutual_info_regression(
                x_train[valid_train].reshape(-1, 1),
                y_train[valid_train],
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
        else:
            mi_train = 0.0
        train_mi[feature] = float(mi_train)

        # Test set
        valid_test = ~(np.isnan(x_test) | np.isnan(y_test))
        if valid_test.sum() >= n_neighbors + 1:
            mi_test = mutual_info_regression(
                x_test[valid_test].reshape(-1, 1),
                y_test[valid_test],
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
        else:
            mi_test = 0.0
        test_mi[feature] = float(mi_test)

    # Compute rankings (1 = highest MI)
    train_sorted = sorted(features, key=lambda f: train_mi[f], reverse=True)
    test_sorted = sorted(features, key=lambda f: test_mi[f], reverse=True)

    train_ranking = {f: i + 1 for i, f in enumerate(train_sorted)}
    test_ranking = {f: i + 1 for i, f in enumerate(test_sorted)}

    # Compute Spearman correlation between rankings
    if len(features) > 1:
        train_ranks = [train_ranking[f] for f in features]
        test_ranks = [test_ranking[f] for f in features]
        rho, _ = spearmanr(train_ranks, test_ranks)
        spearman_correlation = float(rho) if not np.isnan(rho) else 0.0
    else:
        spearman_correlation = 1.0  # Single feature always perfectly correlated

    log_result(
        logger, "OOS validation complete",
        n_features=len(features), spearman_rho=f"{spearman_correlation:.4f}",
        n_train=n_train, n_test=n_test,
    )
    log_function_exit(
        logger, "validate_oos",
        f"spearman_rho={spearman_correlation:.4f}",
    )

    return OutOfSampleResult(
        features=features,
        train_mi=train_mi,
        test_mi=test_mi,
        train_ranking=train_ranking,
        test_ranking=test_ranking,
        spearman_correlation=spearman_correlation,
        n_train=n_train,
        n_test=n_test,
        test_ratio=test_ratio,
        temporal_split=temporal,
    )
