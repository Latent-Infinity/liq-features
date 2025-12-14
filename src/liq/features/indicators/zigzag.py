"""Zigzag pivot detection utility (non-TA-Lib)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from liq.signals import Signal


def zigzag_pivots(
    timestamps: Sequence[datetime],
    prices: Sequence[float | Decimal],
    *,
    pct: float = 0.01,
    symbol: str = "",
) -> List[Signal]:
    """Detect zigzag pivots and emit Signals at highs/lows.

    Args:
        timestamps: Bar timestamps aligned with prices
        prices: Price series (floats/Decimals)
        pct: Reversal threshold (0.01 = 1%)
        symbol: Optional symbol to assign to signals

    Returns:
        List of Signals: low pivots -> long, high pivots -> short
    """

    if len(prices) != len(timestamps):
        raise ValueError("timestamps and prices must be the same length")
    if not timestamps:
        return []

    def _ts(ts: datetime) -> datetime:
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    from liq.signals import Signal  # local import to avoid circular dependency

    signals: list[Signal] = []
    last_pivot = Decimal(str(prices[0]))
    last_ts = _ts(timestamps[0])
    direction = 0  # 0 unknown, 1 uptrend, -1 downtrend

    for ts_raw, price_raw in zip(timestamps[1:], prices[1:]):
        price = Decimal(str(price_raw))
        ts = _ts(ts_raw)
        change = (price - last_pivot) / last_pivot

        if direction == 0:
            if change >= Decimal(pct):
                direction = 1
                last_pivot = price
                last_ts = ts
            elif change <= Decimal(-pct):
                direction = -1
                last_pivot = price
                last_ts = ts
            continue

        if direction == 1:
            if price > last_pivot:
                last_pivot = price
                last_ts = ts
            elif change <= Decimal(-pct):
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=last_ts,
                        direction="short",
                        strength=1.0,
                        metadata={"pivot": "high", "price": float(last_pivot)},
                    )
                )
                direction = -1
                last_pivot = price
                last_ts = ts
        else:
            if price < last_pivot:
                last_pivot = price
                last_ts = ts
            elif change >= Decimal(pct):
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=last_ts,
                        direction="long",
                        strength=1.0,
                        metadata={"pivot": "low", "price": float(last_pivot)},
                    )
                )
                direction = 1
                last_pivot = price
                last_ts = ts

    return signals


__all__ = ["zigzag_pivots"]
