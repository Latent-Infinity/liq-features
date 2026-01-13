"""Effect size calculations for comparing MI distributions.

Implements Cohen's d with pooled standard deviation and bootstrap
confidence intervals for quantifying the magnitude of differences
between close and midrange MI scores.

Cohen's d interpretation:
- |d| < 0.2: negligible effect
- 0.2 <= |d| < 0.5: small effect
- 0.5 <= |d| < 0.8: medium effect
- |d| >= 0.8: large effect
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from liq.features.validation.logging_config import (
    get_logger,
    log_function_entry,
    log_function_exit,
    log_result,
)
from liq.features.validation.results import EffectSizeResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger("effect_size")


def pooled_std(
    group1: NDArray[np.floating],
    group2: NDArray[np.floating],
) -> float:
    """Calculate pooled standard deviation for two groups.

    Uses the formula:
        s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

    This is appropriate when assuming homogeneity of variance.

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Pooled standard deviation.

    Raises:
        ValueError: If either group has fewer than 2 samples.
    """
    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"Each group must have at least 2 samples (got n1={n1}, n2={n2})"
        )

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled variance formula
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    return float(np.sqrt(pooled_var))


def cohens_d(
    group1: NDArray[np.floating],
    group2: NDArray[np.floating],
) -> float:
    """Calculate Cohen's d effect size between two groups.

    Cohen's d is the standardized mean difference:
        d = (mean1 - mean2) / pooled_std

    A negative d indicates group1 has lower mean than group2.

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cohen's d effect size.

    Raises:
        ValueError: If sample sizes are too small or variance is zero.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1 = len(group1)
    n2 = len(group2)

    log_function_entry(logger, "cohens_d", n1=n1, n2=n2)

    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"Each group must have at least 2 samples for sample size "
            f"calculation (got n1={n1}, n2={n2})"
        )

    pstd = pooled_std(group1, group2)

    if pstd == 0:
        raise ValueError(
            "Cannot compute Cohen's d: pooled variance is zero "
            "(both groups have identical values)"
        )

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    d = float((mean1 - mean2) / pstd)

    log_result(logger, "Effect size computed", cohens_d=f"{d:.4f}")
    log_function_exit(logger, "cohens_d", f"d={d:.4f}")

    return d


def cohens_d_ci(
    group1: NDArray[np.floating],
    group2: NDArray[np.floating],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> EffectSizeResult:
    """Calculate Cohen's d with bootstrap confidence interval.

    Uses percentile bootstrap to estimate the CI for Cohen's d.
    This is robust and doesn't assume normality of the d distribution.

    Args:
        group1: First group of values.
        group2: Second group of values.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level for CI (e.g., 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        EffectSizeResult with Cohen's d, CI, and interpretation.

    Raises:
        ValueError: If sample sizes are too small or variance is zero.
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1 = len(group1)
    n2 = len(group2)

    log_function_entry(
        logger, "cohens_d_ci",
        n1=n1, n2=n2, n_bootstrap=n_bootstrap, confidence_level=confidence_level,
    )

    logger.info(f"Computing Cohen's d with {n_bootstrap} bootstrap iterations")

    # Point estimate
    d = cohens_d(group1, group2)
    pstd = pooled_std(group1, group2)
    mean_diff = float(np.mean(group1) - np.mean(group2))

    # Bootstrap with pre-generated indices for performance
    rng = np.random.default_rng(random_state)

    # Pre-generate all bootstrap indices at once (vectorized)
    all_idx1 = rng.integers(0, n1, size=(n_bootstrap, n1))
    all_idx2 = rng.integers(0, n2, size=(n_bootstrap, n2))

    bootstrap_ds = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Use pre-generated indices
        boot_g1 = group1[all_idx1[i]]
        boot_g2 = group2[all_idx2[i]]

        # Check for zero variance in bootstrap sample
        boot_pstd = pooled_std(boot_g1, boot_g2)
        if boot_pstd == 0:
            # Skip this iteration (very rare with continuous data)
            bootstrap_ds[i] = d
        else:
            boot_mean_diff = np.mean(boot_g1) - np.mean(boot_g2)
            bootstrap_ds[i] = boot_mean_diff / boot_pstd

    # Percentile CI
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_ds, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_ds, 100 * (1 - alpha / 2)))

    # Interpretation
    interpretation = EffectSizeResult.interpret_cohens_d(d)

    log_result(
        logger, "Bootstrap CI computed",
        d=f"{d:.4f}", ci=f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        interpretation=interpretation,
    )
    log_function_exit(logger, "cohens_d_ci", f"d={d:.4f}, {interpretation}")

    return EffectSizeResult(
        cohens_d=d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_group1=n1,
        n_group2=n2,
        pooled_std=pstd,
        mean_diff=mean_diff,
        interpretation=interpretation,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def batch_cohens_d(
    mi_close: dict[str, float],
    mi_midrange: dict[str, float],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, EffectSizeResult]:
    """Calculate Cohen's d for each feature comparing close vs midrange MI.

    This is a convenience function for comparing MI scores across features.
    It treats each feature's bootstrap MI distribution as a group.

    Note: This requires bootstrap MI values, not just point estimates.
    For point estimates, use a paired t-test or Wilcoxon signed-rank test.

    Args:
        mi_close: Dict mapping feature name to list of bootstrap MI values.
        mi_midrange: Dict mapping feature name to list of bootstrap MI values.
        n_bootstrap: Number of bootstrap iterations for CI.
        confidence_level: Confidence level for CI.
        random_state: Random seed for reproducibility.

    Returns:
        Dict mapping feature name to EffectSizeResult.
    """
    results = {}
    common_features = set(mi_close.keys()) & set(mi_midrange.keys())

    for feature in common_features:
        close_vals = np.asarray(mi_close[feature])
        midrange_vals = np.asarray(mi_midrange[feature])

        if len(close_vals) >= 2 and len(midrange_vals) >= 2:
            results[feature] = cohens_d_ci(
                close_vals,
                midrange_vals,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )

    return results
