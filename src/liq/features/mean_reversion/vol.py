"""Trailing range volatility helpers for mean-reversion scans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from liq.features.mean_reversion.base import NAN, _high_low, _validate_lookback


def trailing_range_vol(
    bars: pl.DataFrame | Mapping[str, Sequence[float]],
    lookback: int,
) -> list[float]:
    """Trailing mean of bar ranges ``high - low``.

    Output uses ``NaN`` until a full ``lookback`` bars is available. The
    helper is cadence-agnostic; callers own calendar/session policy.
    """
    _validate_lookback(lookback)
    high, low = _high_low(bars)
    ranges = [h - lo for h, lo in zip(high, low, strict=True)]
    out: list[float] = []
    for end in range(len(ranges)):
        start = end - lookback + 1
        if start < 0:
            out.append(NAN)
            continue
        window = [v for v in ranges[start : end + 1] if not math.isnan(v)]
        out.append(sum(window) / len(window) if len(window) == lookback else NAN)
    return out


__all__ = ["trailing_range_vol"]
