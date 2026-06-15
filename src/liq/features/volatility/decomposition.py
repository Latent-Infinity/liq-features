"""Variance decomposition: ``estimator_dispersion`` + §5.4 quality flags.

The decomposition layer turns a dict of per-estimator variance outputs
into:

- a scalar ``estimator_dispersion`` (sample std across estimators),
  used as the §5.4 "disagreement is information" signal, AND
- a tuple of quality flags following the §5.4 derivation table:

  | Pattern | Flag |
  | --- | --- |
  | Parkinson low + CtC high | ``GAP_DOMINATED_VOL`` |
  | RS high + CtC low | ``INTRADAY_RANGE_DOMINATED_VOL`` |
  | wide spread overall | ``HIGH_ESTIMATOR_DISAGREEMENT`` |
  | range estimators ≠ CtC | ``CTC_DISAGREES_WITH_RANGE`` |

The continuous / overnight / jump component series live on
``VolEstimate.components`` and are populated by the windowed code path
in ``estimate.py``; this module is the *signal-derivation* side of the
decomposition story.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from liq.features.volatility.quality import (
    FLAG_CTC_DISAGREES_WITH_RANGE,
    FLAG_GAP_DOMINATED_VOL,
    FLAG_HIGH_ESTIMATOR_DISAGREEMENT,
    FLAG_INTRADAY_RANGE_DOMINATED_VOL,
)

if TYPE_CHECKING:
    pass


_RANGE_KEYS = ("parkinson", "garman_klass", "rogers_satchell")

# Thresholds for the §5.4 pattern detectors. The defaults are
# multiplicative ratios; the dispersion threshold is absolute.
_DOMINATION_RATIO = 4.0  # one estimator >= 4× another → "dominates"
_CTC_RANGE_DIVERGENCE_RATIO = 4.0  # CtC and range-mean differ by ≥ this factor


def estimator_dispersion(estimates: dict[str, float]) -> float:
    """Return ``stdev_pop({estimates.values()})`` — the §5.4 signal."""
    if not estimates:
        raise ValueError("estimator_dispersion: requires at least one estimate")
    values = list(estimates.values())
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)


def _safe_get(estimates: dict[str, float], key: str) -> float | None:
    val = estimates.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def _range_mean(estimates: dict[str, float]) -> float | None:
    """Mean of the available range estimators (P, GK, RS). Returns
    ``None`` when none are present."""
    available = [v for k in _RANGE_KEYS if (v := _safe_get(estimates, k)) is not None]
    if not available:
        return None
    return sum(available) / len(available)


def derive_quality_flags(
    estimates: dict[str, float],
    *,
    dispersion_threshold: float,
) -> tuple[str, ...]:
    """Apply the §5.4 derivation table to ``estimates``.

    ``dispersion_threshold`` is the absolute std-across-estimators value
    above which ``HIGH_ESTIMATOR_DISAGREEMENT`` fires; callers pass
    ``VolQualityPolicy.estimator_dispersion_threshold``.

    Missing estimator keys are tolerated — degraded-fallback callers can
    pass a partial dict without the function crashing.
    """
    flags: list[str] = []

    ctc = _safe_get(estimates, "ctc")
    parkinson = _safe_get(estimates, "parkinson")
    rs = _safe_get(estimates, "rogers_satchell")
    range_mean = _range_mean(estimates)

    # GAP_DOMINATED_VOL: Parkinson (intraday-only) is small but CtC
    # (includes the gap) is large → the gap accounts for most variance.
    if (
        ctc is not None
        and parkinson is not None
        and parkinson > 0
        and ctc >= _DOMINATION_RATIO * parkinson
    ):
        flags.append(FLAG_GAP_DOMINATED_VOL)

    # INTRADAY_RANGE_DOMINATED_VOL: RS (intraday range) is large but CtC
    # is small → most of the move was intraday churn that reverted.
    if ctc is not None and rs is not None and ctc > 0 and rs >= _DOMINATION_RATIO * ctc:
        flags.append(FLAG_INTRADAY_RANGE_DOMINATED_VOL)

    # HIGH_ESTIMATOR_DISAGREEMENT: scalar dispersion exceeds policy bar.
    if len(estimates) >= 2 and estimator_dispersion(estimates) > dispersion_threshold:
        flags.append(FLAG_HIGH_ESTIMATOR_DISAGREEMENT)

    # CTC_DISAGREES_WITH_RANGE: CtC and the range-mean differ by a wide
    # multiplicative factor (in either direction).
    if ctc is not None and range_mean is not None and min(ctc, range_mean) > 0:
        ratio = max(ctc, range_mean) / min(ctc, range_mean)
        if ratio >= _CTC_RANGE_DIVERGENCE_RATIO:
            flags.append(FLAG_CTC_DISAGREES_WITH_RANGE)

    return tuple(flags)


__all__ = [
    "derive_quality_flags",
    "estimator_dispersion",
]
