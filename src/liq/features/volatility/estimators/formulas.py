"""Exact per-bar variance contributions per research plan
``[APPENDIX_FORMULAS]``.

Every function takes **log prices** — the contract caller (in
``estimate.py``) is responsible for taking ``log(price)`` before
dispatching. This keeps the closed-form expressions exactly as written
in the appendix and makes scale invariance free: scaling the original
OHLC by ``k > 0`` shifts every log by ``log(k)``, which cancels out of
every formula (each is a polynomial in *differences* of logs).

Notation (research plan appendix):

```
o, h, l, c = log(open), log(high), log(low), log(close)
c_prev     = log(prev close)
u = h - o,  d = l - o,  k = c - o
```

The registry exposes one function per estimator + the three per-bar
quantities that compose Yang-Zhang's windowed form.
"""

from __future__ import annotations

import math

_LN2 = math.log(2.0)
_FOUR_LN2 = 4.0 * _LN2
_TWO_LN2_MINUS_1 = 2.0 * _LN2 - 1.0


def ctc_var_contribution(c: float, c_prev: float) -> float:
    """Close-to-close per-bar variance contribution: ``(c - c_prev)^2``.

    Captures total close-to-close return variation including the
    overnight gap. Idealized GBM efficiency = 1×; the baseline that
    every other estimator is compared against.
    """
    diff = c - c_prev
    return diff * diff


def parkinson_var_contribution(h: float, low: float) -> float:
    """Parkinson per-bar variance contribution:
    ``(1 / (4 * ln2)) * (h - l)^2``.

    Idealized GBM efficiency ≈ 5× CtC. Uses only the bar's high/low so
    is the cleanest intraday-continuous baseline, but ignores O/C and
    therefore underestimates when overnight gaps are present.
    """
    span = h - low
    return span * span / _FOUR_LN2


def garman_klass_var_contribution(o: float, h: float, low: float, c: float) -> float:
    """Garman-Klass per-bar variance contribution:
    ``0.5 * (h - l)^2 - (2 * ln2 - 1) * (c - o)^2``.

    Idealized GBM efficiency ≈ 7–8× CtC. Biased by overnight jumps and
    by intrabar drift; the canonical drift-aware extension is
    Rogers-Satchell.
    """
    span = h - low
    co = c - o
    return 0.5 * span * span - _TWO_LN2_MINUS_1 * co * co


def rogers_satchell_var_contribution(o: float, h: float, low: float, c: float) -> float:
    """Rogers-Satchell per-bar variance contribution:
    ``u * (u - k) + d * (d - k)`` where ``u = h - o``, ``d = l - o``,
    ``k = c - o``.

    Idealized GBM efficiency ≈ 6–8× CtC. Handles nonzero drift —
    distinguishes it from Garman-Klass. Sensitive to jumps and
    requires a reliable open.
    """
    u = h - o
    d = low - o
    k = c - o
    return u * (u - k) + d * (d - k)


def gk_yang_zhang_var_contribution(
    o: float, h: float, low: float, c: float, c_prev: float
) -> float:
    """Garman-Klass-Yang-Zhang per-bar variance contribution:
    ``(o - c_prev)^2 + garman_klass_var``.

    Extends Garman-Klass with the opening-jump term so the close-to-close
    variation is captured. Idealized GBM efficiency ≈ 7× CtC.
    Assumes zero intrabar drift.
    """
    oc_prev = o - c_prev
    return oc_prev * oc_prev + garman_klass_var_contribution(o, h, low, c)


def yz_overnight_term(o: float, c_prev: float) -> float:
    """Yang-Zhang per-bar overnight term: ``(o - c_prev)^2``.

    Aggregated by the windowed form into ``overnight_var`` — see
    ``windowed.yang_zhang_var``.
    """
    diff = o - c_prev
    return diff * diff


def yz_open_close_term(o: float, c: float) -> float:
    """Yang-Zhang per-bar open-close term: ``(c - o)^2``.

    Aggregated by the windowed form into ``open_close_var`` and
    weighted by ``k_yz``.
    """
    diff = c - o
    return diff * diff


def yz_rs_term(o: float, h: float, low: float, c: float) -> float:
    """Yang-Zhang per-bar Rogers-Satchell term, identical to
    :func:`rogers_satchell_var_contribution`. Re-exported under a
    Yang-Zhang-specific name so the windowed composition reads
    naturally:

    ``yz_var = mean(overnight) + k_yz * mean(open_close) + (1 - k_yz) * mean(rs)``
    """
    return rogers_satchell_var_contribution(o, h, low, c)


__all__ = [
    "ctc_var_contribution",
    "garman_klass_var_contribution",
    "gk_yang_zhang_var_contribution",
    "parkinson_var_contribution",
    "rogers_satchell_var_contribution",
    "yz_open_close_term",
    "yz_overnight_term",
    "yz_rs_term",
]
