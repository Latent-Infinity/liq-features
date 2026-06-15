"""Realized variance / bipower variation / jump variation helpers + RV-noise gate.

The minute-mode estimators (``compute_rv``, ``compute_bpv``,
``compute_jv``, ``rv_noise_gate``) implement research plan §5.3 +
``[APPENDIX_FORMULAS]``. Every helper takes log-returns (not log-prices)
so the caller is responsible for the diff-of-log conversion.

Contract notes:

- ``compute_rv`` is the sum of squared log-returns, the canonical
  realized-variance estimator.
- ``compute_bpv`` is jump-robust: it weights *adjacent* return
  magnitudes so a single large return shows up in only two terms (vs.
  one squared term in RV), keeping its variance bounded by the
  continuous part.
- ``compute_jv = max(RV - BPV, 0)`` isolates the jump variance under
  the Barndorff-Nielsen / Shephard decomposition.
- ``rv_noise_gate`` enforces research plan §5.3's hard rule: when
  ``RV_1m`` materially exceeds ``RV_5m / RV_15m`` *without*
  corresponding price movement, the gate fires and the caller must
  fall back to ``RV_5m`` (or kernel) and flag the bar
  ``NOISY_RV_TARGET``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_PI_OVER_2 = math.pi / 2.0
# A 1m-vs-5m RV ratio above this without commensurate price movement
# is treated as microstructure noise per §5.3. The threshold is tuned
# so a true high-vol session (where RV scales mildly with finer
# sampling) does not trip the gate.
_RV_NOISE_RATIO_THRESHOLD = 2.0
# When ``price_movement² / RV_5m < this``, the inflated 1m RV is
# considered "unexplained by actual directional price movement" —
# diagnostic of microstructure noise. RV_5m is a low-noise reference
# of the true variance for the session.
_UNEXPLAINED_THRESHOLD = 0.5


def compute_rv(intra_bar_log_returns: Sequence[float]) -> float:
    """Realized variance: ``Σ r_i²``.

    ``intra_bar_log_returns`` is the sequence of log-returns at the
    chosen sampling interval (1m / 5m / 15m) within one session.
    Returns the sum of their squares.
    """
    return sum(r * r for r in intra_bar_log_returns)


def compute_bpv(intra_bar_log_returns: Sequence[float]) -> float:
    """Bipower variation: ``(π/2) · Σ_{i>=1} |r_{i-1}| · |r_i|``.

    Jump-robust estimator of the integrated continuous variance.
    Returns ``0.0`` for fewer than two returns (BPV requires a pair).
    """
    if len(intra_bar_log_returns) < 2:
        return 0.0
    total = 0.0
    prev_abs = abs(intra_bar_log_returns[0])
    for r in intra_bar_log_returns[1:]:
        cur_abs = abs(r)
        total += prev_abs * cur_abs
        prev_abs = cur_abs
    return _PI_OVER_2 * total


def compute_jv(intra_bar_log_returns: Sequence[float]) -> float:
    """Jump variation: ``max(compute_rv - compute_bpv, 0)``.

    Under the Barndorff-Nielsen / Shephard decomposition the continuous
    part is captured by BPV; the residual is the jump contribution.
    Clipped at zero so ``jv_t`` stays non-negative in finite samples.
    """
    rv = compute_rv(intra_bar_log_returns)
    bpv = compute_bpv(intra_bar_log_returns)
    return max(rv - bpv, 0.0)


def rv_noise_gate(
    rv_by_interval: dict[str, float],
    *,
    price_movement: float,
    ratio_threshold: float = _RV_NOISE_RATIO_THRESHOLD,
    unexplained_threshold: float = _UNEXPLAINED_THRESHOLD,
) -> bool:
    """Return ``True`` when ``RV_1m`` looks like microstructure noise.

    Implements research plan §5.3. The gate fires when BOTH:

    - ``RV_1m`` is at least ``ratio_threshold`` times each of ``RV_5m``
      and ``RV_15m`` (the finer-sampling-inflation pattern), AND
    - the session's price movement does not justify that inflation —
      operationalized as ``price_movement² < unexplained_threshold ·
      RV_5m`` (the directional move squared explains less than half of
      the lower-noise RV reference, so the inflation cannot be
      attributed to a real directional shock).

    Returns ``True`` if the gate fires (caller falls back to coarser
    sampling and flags ``NOISY_RV_TARGET``); ``False`` otherwise.

    Args:
        rv_by_interval: must contain ``"1m"``, ``"5m"``, and ``"15m"``
            keys.
        price_movement: signed log-return of the session
            (``log(close / open)``); only the magnitude is used.

    Raises:
        KeyError: when any of the required intervals is missing.
    """
    rv_1m = rv_by_interval["1m"]
    rv_5m = rv_by_interval["5m"]
    rv_15m = rv_by_interval["15m"]
    inflated_vs_5m = rv_1m >= ratio_threshold * rv_5m
    inflated_vs_15m = rv_1m >= ratio_threshold * rv_15m
    if not (inflated_vs_5m and inflated_vs_15m):
        return False
    return (price_movement * price_movement) < unexplained_threshold * rv_5m


__all__ = [
    "compute_bpv",
    "compute_jv",
    "compute_rv",
    "rv_noise_gate",
]
