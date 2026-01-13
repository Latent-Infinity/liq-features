"""Statistical significance testing for mutual information scores.

This module provides:
1. Bootstrap confidence intervals for MI estimates
2. Permutation tests for H0: MI = 0
3. Paired difference tests for comparing close vs midrange targets
4. FDR correction for multiple comparisons

Example:
    >>> from liq.features.selection.significance import bootstrap_mi, permutation_test_mi
    >>>
    >>> # Single feature bootstrap CI
    >>> result = bootstrap_mi(X, y, "natr_timeperiod5", n_bootstrap=10000)
    >>> print(f"MI = {result.point_estimate:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    >>>
    >>> # Permutation test
    >>> perm = permutation_test_mi(X, y, "natr_timeperiod5", n_permutations=10000)
    >>> print(f"p-value = {perm.p_value:.4f}")
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl

from liq.features.numpy_utils import to_numpy_float64
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_regression

from liq.features.selection.significance_results import (
    BootstrapResult,
    PairedDifferenceResult,
    PermutationResult,
    SignificanceReport,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _compute_mi_from_arrays(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    n_neighbors: int = 3,
    random_state: int | None = None,
) -> float:
    """Compute MI between two 1D arrays using sklearn.

    Args:
        x: Feature values (1D array, no NaN).
        y: Target values (1D array, no NaN).
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.

    Returns:
        Mutual information score.
    """
    if len(x) < n_neighbors + 1:
        return np.nan

    # sklearn expects 2D feature array
    x_2d = x.reshape(-1, 1)
    mi = mutual_info_regression(
        x_2d, y, n_neighbors=n_neighbors, random_state=random_state
    )
    return float(mi[0])


def _get_valid_mask(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Get mask for valid (non-NaN, non-inf) entries in both arrays."""
    return (
        ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    )


def bootstrap_mi(
    X: pl.DataFrame,
    y: pl.Series,
    feature: str,
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for MI score.

    Uses normal approximation CI (point_estimate ± z * std_error) with the
    standard error estimated from bootstrap resampling. This approach is more
    appropriate for k-NN based MI estimation since resampling with replacement
    can introduce upward bias in MI estimates due to duplicate points.

    Args:
        X: DataFrame with feature columns.
        y: Target Series.
        feature: Name of feature column to analyze.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and confidence interval.

    Raises:
        ValueError: If feature not in DataFrame or insufficient valid samples.
    """
    from scipy import stats

    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Convert to numpy
    x_arr = to_numpy_float64(X[feature])
    y_arr = to_numpy_float64(y)

    # Get valid indices
    valid_mask = _get_valid_mask(x_arr, y_arr)
    n_valid = valid_mask.sum()

    if n_valid < 100:
        raise ValueError(f"Insufficient valid samples: {n_valid} < 100")

    x_valid = x_arr[valid_mask]
    y_valid = y_arr[valid_mask]

    # Compute point estimate on full data
    point_estimate = _compute_mi_from_arrays(
        x_valid, y_valid, n_neighbors=n_neighbors, random_state=random_state
    )

    # Bootstrap resampling
    rng = np.random.default_rng(random_state)
    bootstrap_mis = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, n_valid, size=n_valid)
        x_boot = x_valid[indices]
        y_boot = y_valid[indices]

        # Compute MI on resampled data
        # Use different random state for each iteration to avoid correlation
        bootstrap_mis[i] = _compute_mi_from_arrays(
            x_boot, y_boot, n_neighbors=n_neighbors, random_state=random_state + i
        )

    # Standard error from bootstrap distribution
    std_error = float(np.nanstd(bootstrap_mis))

    # Use normal approximation CI centered on point estimate
    # This avoids bias from percentile method with k-NN MI estimation
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = max(0.0, point_estimate - z * std_error)  # MI >= 0
    ci_upper = point_estimate + z * std_error

    return BootstrapResult(
        feature=feature,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_samples=int(n_valid),
    )


def permutation_test_mi(
    X: pl.DataFrame,
    y: pl.Series,
    feature: str,
    *,
    n_permutations: int = 10000,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> PermutationResult:
    """Test null hypothesis that MI = 0 via permutation.

    Shuffles target values to break the X-y relationship, creating a null
    distribution of MI scores. P-value is the proportion of null MI values
    that are >= the observed MI.

    Args:
        X: DataFrame with feature columns.
        y: Target Series.
        feature: Name of feature column to analyze.
        n_permutations: Number of permutation iterations.
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.

    Returns:
        PermutationResult with observed MI and p-value.

    Raises:
        ValueError: If feature not in DataFrame or insufficient valid samples.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Convert to numpy
    x_arr = to_numpy_float64(X[feature])
    y_arr = to_numpy_float64(y)

    # Get valid indices
    valid_mask = _get_valid_mask(x_arr, y_arr)
    n_valid = valid_mask.sum()

    if n_valid < 100:
        raise ValueError(f"Insufficient valid samples: {n_valid} < 100")

    x_valid = x_arr[valid_mask]
    y_valid = y_arr[valid_mask]

    # Compute observed MI
    observed_mi = _compute_mi_from_arrays(
        x_valid, y_valid, n_neighbors=n_neighbors, random_state=random_state
    )

    # Permutation null distribution
    rng = np.random.default_rng(random_state)
    null_mis = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle y to break relationship
        y_shuffled = rng.permutation(y_valid)

        # Compute MI on shuffled data
        null_mis[i] = _compute_mi_from_arrays(
            x_valid, y_shuffled, n_neighbors=n_neighbors, random_state=random_state + i
        )

    # P-value: proportion of null >= observed
    p_value = float(np.mean(null_mis >= observed_mi))

    # Null distribution statistics
    null_mean = float(np.nanmean(null_mis))
    null_std = float(np.nanstd(null_mis))

    return PermutationResult(
        feature=feature,
        observed_mi=observed_mi,
        p_value=p_value,
        null_mean=null_mean,
        null_std=null_std,
        n_permutations=n_permutations,
        n_samples=int(n_valid),
    )


def paired_bootstrap_difference(
    X: pl.DataFrame,
    y_close: pl.Series,
    y_midrange: pl.Series,
    feature: str,
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> PairedDifferenceResult:
    """Test if MI(feature, midrange) > MI(feature, close).

    Uses paired bootstrap: same resampled indices for both targets to account
    for correlation. Tests H0: MI_midrange - MI_close <= 0.

    Args:
        X: DataFrame with feature columns.
        y_close: Close target Series.
        y_midrange: Midrange target Series.
        feature: Name of feature column to analyze.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level for difference CI.
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.

    Returns:
        PairedDifferenceResult with difference estimate and p-value.

    Raises:
        ValueError: If feature not in DataFrame or insufficient valid samples.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Convert to numpy
    x_arr = to_numpy_float64(X[feature])
    y_close_arr = to_numpy_float64(y_close)
    y_midrange_arr = to_numpy_float64(y_midrange)

    # Get valid indices (must be valid for all three arrays)
    valid_mask = (
        _get_valid_mask(x_arr, y_close_arr) &
        _get_valid_mask(x_arr, y_midrange_arr)
    )
    n_valid = valid_mask.sum()

    if n_valid < 100:
        raise ValueError(f"Insufficient valid samples: {n_valid} < 100")

    x_valid = x_arr[valid_mask]
    y_close_valid = y_close_arr[valid_mask]
    y_midrange_valid = y_midrange_arr[valid_mask]

    # Compute point estimates on full data
    close_mi = _compute_mi_from_arrays(
        x_valid, y_close_valid, n_neighbors=n_neighbors, random_state=random_state
    )
    midrange_mi = _compute_mi_from_arrays(
        x_valid, y_midrange_valid, n_neighbors=n_neighbors, random_state=random_state
    )
    difference = midrange_mi - close_mi

    # Paired bootstrap
    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Same indices for both targets (paired)
        indices = rng.integers(0, n_valid, size=n_valid)
        x_boot = x_valid[indices]
        y_close_boot = y_close_valid[indices]
        y_midrange_boot = y_midrange_valid[indices]

        # Compute MI difference
        mi_close = _compute_mi_from_arrays(
            x_boot, y_close_boot, n_neighbors=n_neighbors, random_state=random_state + i
        )
        mi_midrange = _compute_mi_from_arrays(
            x_boot, y_midrange_boot, n_neighbors=n_neighbors, random_state=random_state + i
        )
        bootstrap_diffs[i] = mi_midrange - mi_close

    # Calculate percentile CI for difference
    alpha = 1 - confidence_level
    ci_lower = float(np.nanpercentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.nanpercentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    # P-value: proportion of differences <= 0 (testing H0: diff <= 0)
    p_value = float(np.mean(bootstrap_diffs <= 0))

    return PairedDifferenceResult(
        feature=feature,
        close_mi=close_mi,
        midrange_mi=midrange_mi,
        difference=difference,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
        n_samples=int(n_valid),
    )


def batch_bootstrap_mi(
    X: pl.DataFrame,
    y: pl.Series,
    features: list[str] | None = None,
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    n_neighbors: int = 3,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[BootstrapResult]:
    """Compute bootstrap CI for multiple features.

    Args:
        X: DataFrame with feature columns.
        y: Target Series.
        features: List of feature names. If None, uses all columns.
        n_bootstrap: Number of bootstrap iterations per feature.
        confidence_level: Confidence level for CIs.
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel workers. None = all cores.
        progress_callback: Optional callback(current, total).

    Returns:
        List of BootstrapResult objects.
    """
    if features is None:
        features = X.columns

    if n_jobs is None:
        import os
        n_jobs = os.cpu_count() or 1

    results: list[BootstrapResult] = []
    completed = 0

    def process_feature(feat: str, seed: int) -> BootstrapResult | None:
        try:
            return bootstrap_mi(
                X, y, feat,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                n_neighbors=n_neighbors,
                random_state=seed,
            )
        except ValueError as e:
            logger.warning(f"Bootstrap failed for {feat}: {e}")
            return None

    # Use thread pool for parallel execution
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(process_feature, feat, random_state + i): feat
            for i, feat in enumerate(features)
        }

        for future in as_completed(futures):
            feat = futures[future]
            completed += 1

            if progress_callback:
                progress_callback(completed, len(features))

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Bootstrap failed for {feat}: {e}")

    # Sort by feature name for consistent ordering
    results.sort(key=lambda r: r.feature)
    return results


def batch_permutation_test(
    X: pl.DataFrame,
    y: pl.Series,
    features: list[str] | None = None,
    *,
    n_permutations: int = 10000,
    n_neighbors: int = 3,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[PermutationResult]:
    """Compute permutation tests for multiple features.

    Args:
        X: DataFrame with feature columns.
        y: Target Series.
        features: List of feature names. If None, uses all columns.
        n_permutations: Number of permutation iterations per feature.
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel workers. None = all cores.
        progress_callback: Optional callback(current, total).

    Returns:
        List of PermutationResult objects.
    """
    if features is None:
        features = X.columns

    if n_jobs is None:
        import os
        n_jobs = os.cpu_count() or 1

    results: list[PermutationResult] = []
    completed = 0

    def process_feature(feat: str, seed: int) -> PermutationResult | None:
        try:
            return permutation_test_mi(
                X, y, feat,
                n_permutations=n_permutations,
                n_neighbors=n_neighbors,
                random_state=seed,
            )
        except ValueError as e:
            logger.warning(f"Permutation test failed for {feat}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(process_feature, feat, random_state + i): feat
            for i, feat in enumerate(features)
        }

        for future in as_completed(futures):
            feat = futures[future]
            completed += 1

            if progress_callback:
                progress_callback(completed, len(features))

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Permutation test failed for {feat}: {e}")

    results.sort(key=lambda r: r.feature)
    return results


def batch_paired_difference(
    X: pl.DataFrame,
    y_close: pl.Series,
    y_midrange: pl.Series,
    features: list[str] | None = None,
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    n_neighbors: int = 3,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[PairedDifferenceResult]:
    """Compute paired difference tests for multiple features.

    Args:
        X: DataFrame with feature columns.
        y_close: Close target Series.
        y_midrange: Midrange target Series.
        features: List of feature names. If None, uses all columns.
        n_bootstrap: Number of bootstrap iterations per feature.
        confidence_level: Confidence level for CIs.
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel workers. None = all cores.
        progress_callback: Optional callback(current, total).

    Returns:
        List of PairedDifferenceResult objects.
    """
    if features is None:
        features = X.columns

    if n_jobs is None:
        import os
        n_jobs = os.cpu_count() or 1

    results: list[PairedDifferenceResult] = []
    completed = 0

    def process_feature(feat: str, seed: int) -> PairedDifferenceResult | None:
        try:
            return paired_bootstrap_difference(
                X, y_close, y_midrange, feat,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                n_neighbors=n_neighbors,
                random_state=seed,
            )
        except ValueError as e:
            logger.warning(f"Paired test failed for {feat}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(process_feature, feat, random_state + i): feat
            for i, feat in enumerate(features)
        }

        for future in as_completed(futures):
            feat = futures[future]
            completed += 1

            if progress_callback:
                progress_callback(completed, len(features))

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Paired test failed for {feat}: {e}")

    results.sort(key=lambda r: r.feature)
    return results


def apply_fdr_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: List of p-values to correct.
        alpha: Significance level.

    Returns:
        Tuple of (significant_flags, adjusted_p_values).
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Convert to numpy for easier manipulation
    p_arr = np.array(p_values)

    # Handle NaN p-values
    nan_mask = np.isnan(p_arr)

    # Sort p-values (excluding NaN)
    valid_indices = np.where(~nan_mask)[0]
    sorted_order = valid_indices[np.argsort(p_arr[valid_indices])]

    # Benjamini-Hochberg procedure
    adjusted = np.full(n, np.nan)
    significant = np.zeros(n, dtype=bool)

    m = len(sorted_order)
    if m > 0:
        # Calculate adjusted p-values
        ranks = np.arange(1, m + 1)
        raw_adjusted = p_arr[sorted_order] * m / ranks

        # Enforce monotonicity (cumulative minimum from right)
        adjusted_bh = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
        adjusted_bh = np.minimum(adjusted_bh, 1.0)  # Cap at 1

        # Map back to original order
        adjusted[sorted_order] = adjusted_bh

        # Determine significance
        significant[sorted_order] = adjusted_bh < alpha

    return significant.tolist(), adjusted.tolist()


def run_significance_analysis(
    X: pl.DataFrame,
    y_close: pl.Series,
    y_midrange: pl.Series,
    features: list[str],
    *,
    timeframe: str = "",
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    confidence_level: float = 0.95,
    n_neighbors: int = 3,
    random_state: int = 42,
    n_jobs: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> SignificanceReport:
    """Run complete significance analysis for close vs midrange comparison.

    Runs bootstrap CI, permutation tests, and paired difference tests for
    all specified features, then applies FDR correction.

    Args:
        X: DataFrame with feature columns.
        y_close: Close target Series.
        y_midrange: Midrange target Series.
        features: List of feature names to analyze.
        timeframe: Timeframe label for report.
        n_bootstrap: Number of bootstrap iterations.
        n_permutations: Number of permutation iterations.
        confidence_level: Confidence level for CIs.
        n_neighbors: k for k-NN MI estimation.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel workers.
        progress_callback: Optional callback(stage, current, total).

    Returns:
        SignificanceReport with all results.
    """
    report = SignificanceReport(
        timeframe=timeframe,
        n_features=len(features),
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        confidence_level=confidence_level,
    )

    def make_progress(stage: str):
        def callback(current: int, total: int):
            if progress_callback:
                progress_callback(stage, current, total)
        return callback

    # Stage 1: Bootstrap CI for close
    logger.info(f"Running bootstrap CI for close target ({len(features)} features)")
    report.close_bootstrap = batch_bootstrap_mi(
        X, y_close, features,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs,
        progress_callback=make_progress("Bootstrap (close)"),
    )

    # Stage 2: Bootstrap CI for midrange
    logger.info(f"Running bootstrap CI for midrange target ({len(features)} features)")
    report.midrange_bootstrap = batch_bootstrap_mi(
        X, y_midrange, features,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_neighbors=n_neighbors,
        random_state=random_state + 1000,
        n_jobs=n_jobs,
        progress_callback=make_progress("Bootstrap (midrange)"),
    )

    # Stage 3: Permutation test for close
    logger.info(f"Running permutation tests for close target ({len(features)} features)")
    report.close_permutation = batch_permutation_test(
        X, y_close, features,
        n_permutations=n_permutations,
        n_neighbors=n_neighbors,
        random_state=random_state + 2000,
        n_jobs=n_jobs,
        progress_callback=make_progress("Permutation (close)"),
    )

    # Stage 4: Permutation test for midrange
    logger.info(f"Running permutation tests for midrange target ({len(features)} features)")
    report.midrange_permutation = batch_permutation_test(
        X, y_midrange, features,
        n_permutations=n_permutations,
        n_neighbors=n_neighbors,
        random_state=random_state + 3000,
        n_jobs=n_jobs,
        progress_callback=make_progress("Permutation (midrange)"),
    )

    # Stage 5: Paired difference tests
    logger.info(f"Running paired difference tests ({len(features)} features)")
    report.paired_results = batch_paired_difference(
        X, y_close, y_midrange, features,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_neighbors=n_neighbors,
        random_state=random_state + 4000,
        n_jobs=n_jobs,
        progress_callback=make_progress("Paired difference"),
    )

    # Stage 6: FDR correction on paired p-values
    if report.paired_results:
        p_values = [r.p_value for r in report.paired_results]
        significant, adjusted = apply_fdr_correction(p_values, alpha=0.05)
        report.fdr_significant = significant
        report.fdr_adjusted_p_values = adjusted

    logger.info(f"Significance analysis complete: {report.n_significant_paired_fdr}/{len(features)} significant after FDR")

    return report
