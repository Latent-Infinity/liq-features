"""Rolling midrange baselines for mean-reversion scans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

NAN = float("nan")


def _validate_lookback(lookback: int) -> None:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")


def _column(values: pl.DataFrame | Mapping[str, Sequence[float]], name: str) -> list[float]:
    if isinstance(values, pl.DataFrame):
        if name not in values.columns:
            raise ValueError(f"bars must include {name!r} column")
        return [float(v) for v in values[name].to_list()]
    if isinstance(values, Mapping):
        if name not in values:
            raise ValueError(f"bars must include {name!r} column")
        raw = values[name]
        if not isinstance(raw, Sequence):
            raise ValueError(f"bars[{name!r}] must be a sequence")
        return [float(v) for v in raw]
    raise TypeError("bars must be a Polars DataFrame or mapping of column sequences")


def _high_low(
    bars: pl.DataFrame | Mapping[str, Sequence[float]],
) -> tuple[list[float], list[float]]:
    high = _column(bars, "high")
    low = _column(bars, "low")
    if len(high) != len(low):
        raise ValueError("high and low columns must have the same length")
    return high, low


def roll_extreme_midrange(
    bars: pl.DataFrame | Mapping[str, Sequence[float]],
    lookback: int,
) -> list[float]:
    """Trailing midpoint of the rolling high extreme and low extreme.

    Output uses ``NaN`` until a full ``lookback`` bars is available.
    """
    _validate_lookback(lookback)
    high, low = _high_low(bars)
    out: list[float] = []
    for end in range(len(high)):
        start = end - lookback + 1
        if start < 0:
            out.append(NAN)
            continue
        high_window = high[start : end + 1]
        low_window = low[start : end + 1]
        if any(math.isnan(v) for v in [*high_window, *low_window]):
            out.append(NAN)
            continue
        out.append((max(high_window) + min(low_window)) / 2.0)
    return out


def roll_mean_midrange(
    bars: pl.DataFrame | Mapping[str, Sequence[float]],
    lookback: int,
) -> list[float]:
    """Trailing mean of per-bar midranges ``(high + low) / 2``.

    Output uses ``NaN`` until a full ``lookback`` bars is available.
    """
    _validate_lookback(lookback)
    high, low = _high_low(bars)
    mids = [(h + lo) / 2.0 for h, lo in zip(high, low, strict=True)]
    out: list[float] = []
    for end in range(len(mids)):
        start = end - lookback + 1
        if start < 0:
            out.append(NAN)
            continue
        window = [v for v in mids[start : end + 1] if not math.isnan(v)]
        out.append(sum(window) / len(window) if len(window) == lookback else NAN)
    return out


__all__ = ["NAN", "roll_extreme_midrange", "roll_mean_midrange"]
