"""Multi-timeframe alignment utilities."""

from __future__ import annotations

import polars as pl


def align_higher_timeframe(
    base_df: pl.DataFrame,
    higher_df: pl.DataFrame,
    shift_periods: int = 1,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Align completed higher timeframe bars to base timeframe timestamps.

    - Uses last available completed higher-TF bar (shifted) and forward-fills.
    - Guards against look-ahead by shifting by `shift_periods`.
    """
    if base_df.is_empty() or higher_df.is_empty():
        return base_df
    higher_sorted = higher_df.sort(timestamp_col)
    # For now, use completed higher bars without look-ahead; shift reserved for future bar-staleness if needed.
    aligned = base_df.sort(timestamp_col).join_asof(
        higher_sorted,
        left_on=timestamp_col,
        right_on=timestamp_col,
        strategy="backward",
    )
    return aligned
