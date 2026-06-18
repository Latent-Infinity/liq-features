"""Regime feature helpers for volatility forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

GAP_DOMINATED_VOL = "GAP_DOMINATED_VOL"
INTRADAY_RANGE_DOMINATED_VOL = "INTRADAY_RANGE_DOMINATED_VOL"
JUMP_DAY = "JUMP_DAY"


@dataclass(frozen=True)
class SemivarianceEstimate:
    downside_rv: float
    upside_rv: float
    bar_rv: float
    window: int
    coverage: dict[str, object]


@dataclass(frozen=True)
class AsymmetryRegressionResult:
    b0: float
    b1: float
    b2: float
    b3: float
    observations: int


def _clean_returns(returns: Sequence[float]) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must contain only finite values")
    return arr


def compute_semivariance(returns: Sequence[float], window: int) -> SemivarianceEstimate:
    """Compute downside/upside realized semivariance over the trailing window."""

    if window <= 0:
        raise ValueError("window must be positive")
    arr = _clean_returns(returns)
    observed = min(arr.size, window)
    tail = arr[-window:]
    downside = float(np.sum(np.minimum(tail, 0.0) ** 2))
    upside = float(np.sum(np.maximum(tail, 0.0) ** 2))
    return SemivarianceEstimate(
        downside_rv=downside,
        upside_rv=upside,
        bar_rv=downside + upside,
        window=window,
        coverage={
            "observed_count": int(observed),
            "has_full_window": bool(arr.size >= window),
        },
    )


def compute_asymmetry_regression(returns: Sequence[float]) -> AsymmetryRegressionResult:
    """Regress next-period squared returns on lagged return asymmetry terms."""

    arr = _clean_returns(returns)
    if arr.size < 3:
        raise ValueError("at least three returns are required")
    lagged = arr[:-1]
    target = arr[1:] ** 2
    negative = (lagged < 0.0).astype(float)
    design = np.column_stack(
        [
            np.ones_like(lagged),
            lagged,
            negative,
            lagged * negative,
        ]
    )
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return AsymmetryRegressionResult(
        b0=_zero_near(float(coef[0])),
        b1=_zero_near(float(coef[1])),
        b2=_zero_near(float(coef[2])),
        b3=_zero_near(float(coef[3])),
        observations=int(target.size),
    )


def _estimate_value(estimate: Mapping[str, float] | object, key: str) -> float:
    value = estimate.get(key, 0.0) if isinstance(estimate, Mapping) else getattr(estimate, key, 0.0)
    return float(value)


def _estimate_flag(estimate: Mapping[str, object] | object, key: str) -> bool:
    if isinstance(estimate, Mapping):
        return bool(estimate.get(key, False))
    return bool(getattr(estimate, key, False))


def _zero_near(value: float, *, tolerance: float = 1e-8) -> float:
    return 0.0 if abs(value) <= tolerance else float(value)


def derive_gap_jump_labels(
    estimate: Mapping[str, object] | object,
    *,
    threshold_gap: float,
    threshold_range: float,
    threshold_jump: float,
) -> set[str]:
    """Derive deterministic gap/jump regime labels for volatility forecasting.

    Inputs (read from ``estimate`` by attribute or mapping key):
        - ``overnight_gap_var``: overnight gap contribution to total variance
        - ``intraday_range_var``: intraday-range contribution to total variance
        - ``jump_var``: jump contribution to total variance
        - ``total_var``: total variance (denominator)
        - ``jump_flag`` (optional): estimator-emitted jump flag; if true,
          ``JUMP_DAY`` fires regardless of the ratio threshold.
    """

    for name, value in (
        ("threshold_gap", threshold_gap),
        ("threshold_range", threshold_range),
        ("threshold_jump", threshold_jump),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    total_var = max(_estimate_value(estimate, "total_var"), 1e-12)
    overnight_gap_var = max(_estimate_value(estimate, "overnight_gap_var"), 0.0)
    intraday_range_var = max(_estimate_value(estimate, "intraday_range_var"), 0.0)
    jump_var = max(_estimate_value(estimate, "jump_var"), 0.0)
    labels: set[str] = set()
    if overnight_gap_var / total_var >= threshold_gap:
        labels.add(GAP_DOMINATED_VOL)
    if intraday_range_var / total_var >= threshold_range:
        labels.add(INTRADAY_RANGE_DOMINATED_VOL)
    if jump_var / total_var >= threshold_jump or _estimate_flag(estimate, "jump_flag"):
        labels.add(JUMP_DAY)
    return labels


def resolve_multi_label(labels: set[str], scores: Mapping[str, float]) -> set[str]:
    """Resolve overlapping conservatism labels by keeping the highest multiplier."""

    if not labels:
        return set()
    missing = labels.difference(scores)
    if missing:
        raise KeyError(f"missing scores for labels: {sorted(missing)}")
    max_score = max(float(scores[label]) for label in labels)
    return {label for label in labels if float(scores[label]) == max_score}


__all__ = [
    "AsymmetryRegressionResult",
    "GAP_DOMINATED_VOL",
    "INTRADAY_RANGE_DOMINATED_VOL",
    "JUMP_DAY",
    "SemivarianceEstimate",
    "compute_asymmetry_regression",
    "compute_semivariance",
    "derive_gap_jump_labels",
    "resolve_multi_label",
]
