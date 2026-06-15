"""Windowed aggregation for the canonical risk-variance estimator.

Each estimator's windowed value is a function of its per-bar
contributions (research plan ``[APPENDIX_FORMULAS]``). The simple
estimators (CtC, Parkinson, GK, RS, GK-YZ) aggregate via a trailing
mean over ``window`` bars with ``min_periods`` enforcement. Yang-Zhang
composes three per-bar means with the canonical
``k_yz = 0.34 / (1.34 + (n + 1) / (n - 1))`` weighting.

Output series obey two contract invariants:

- **NaN until ``min_periods`` is satisfied.** Float ``NaN`` is the
  sentinel; never back-fill or carry-forward an earlier value.
- **Caller-owned cadence.** The per-bar input series defines the
  emission cadence. The windowed aggregator does not know about
  calendar or session boundaries.

Scale invariance is by construction at the formula layer (each
per-bar contribution is a polynomial in *differences* of log prices,
so multiplying OHLC by ``k`` shifts every log by ``log(k)``, which
cancels). The windowed mean preserves that invariance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

NAN = float("nan")


def _validate_window(window: int, min_periods: int) -> None:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if min_periods <= 0:
        raise ValueError(f"min_periods must be positive, got {min_periods}")
    if min_periods > window:
        raise ValueError(f"min_periods ({min_periods}) cannot exceed window ({window})")


def _windowed_mean(
    values: Sequence[float],
    *,
    window: int,
    min_periods: int,
) -> list[float]:
    """Trailing mean over ``window`` samples with ``min_periods`` gate.

    A value at index ``i`` is the mean over indices ``[max(0, i - window + 1), i]``
    of the **non-NaN** samples, provided at least ``min_periods`` of them
    are present; otherwise NaN.
    """
    _validate_window(window, min_periods)
    out: list[float] = []
    for end in range(len(values)):
        start = max(0, end - window + 1)
        slice_ = [v for v in values[start : end + 1] if not math.isnan(v)]
        if len(slice_) < min_periods:
            out.append(NAN)
        else:
            out.append(sum(slice_) / len(slice_))
    return out


def trailing_mean(
    values: Iterable[float],
    *,
    window: int,
    min_periods: int,
) -> list[float]:
    """Public wrapper around the trailing mean, used by every
    non-Yang-Zhang estimator. Returns a list of floats with NaN
    sentinels until ``min_periods`` is met."""
    return _windowed_mean(list(values), window=window, min_periods=min_periods)


def yz_k_weight(window: int) -> float:
    """Yang-Zhang's per-window weighting:
    ``k_yz = 0.34 / (1.34 + (n + 1) / (n - 1))`` per
    ``[APPENDIX_FORMULAS]``.

    Requires ``window >= 2`` — the formula is undefined at ``n == 1``
    because the denominator's ``(n - 1)`` is zero. The research-plan
    Yang-Zhang is multi-period by design.
    """
    if window < 2:
        raise ValueError(
            f"yz_k_weight: Yang-Zhang requires window >= 2 (got {window}); "
            "the closed-form k_yz is undefined at n == 1."
        )
    n = float(window)
    return 0.34 / (1.34 + (n + 1.0) / (n - 1.0))


def yang_zhang_var(
    *,
    overnight_terms: Sequence[float],
    open_close_terms: Sequence[float],
    rs_terms: Sequence[float],
    window: int,
    min_periods: int,
) -> list[float]:
    """Compose the Yang-Zhang windowed variance:
    ``var = mean(overnight) + k * mean(open_close) + (1 - k) * mean(rs)``
    where ``k = yz_k_weight(window)``.

    All three input series must share the same length; the output series
    has that same length with NaN until ``min_periods`` is satisfied in
    every component (the most restrictive gate wins).

    Yang-Zhang requires the previous-bar close to build the overnight
    term, so callers typically pass NaN for the first slot of
    ``overnight_terms``. The windowed mean drops NaN samples, so the
    practical ``min_periods`` floor becomes ``min_periods + 1`` for any
    window that includes index 0; downstream tests rely on this.
    """
    if not (len(overnight_terms) == len(open_close_terms) == len(rs_terms)):
        raise ValueError(
            "yang_zhang_var: overnight_terms, open_close_terms, and rs_terms "
            "must share the same length"
        )
    k = yz_k_weight(window)
    ov_w = _windowed_mean(list(overnight_terms), window=window, min_periods=min_periods)
    oc_w = _windowed_mean(list(open_close_terms), window=window, min_periods=min_periods)
    rs_w = _windowed_mean(list(rs_terms), window=window, min_periods=min_periods)
    out: list[float] = []
    for ov, oc, r in zip(ov_w, oc_w, rs_w, strict=True):
        if math.isnan(ov) or math.isnan(oc) or math.isnan(r):
            out.append(NAN)
        else:
            out.append(ov + k * oc + (1.0 - k) * r)
    return out


__all__ = [
    "NAN",
    "trailing_mean",
    "yang_zhang_var",
    "yz_k_weight",
]
