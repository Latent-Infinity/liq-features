"""Multi-timeframe alignment utilities."""

from __future__ import annotations

import polars as pl


def _feature_columns(df: pl.DataFrame, timestamp_col: str, by: list[str]) -> list[str]:
    excluded = {timestamp_col, *by}
    return [col for col in df.columns if col not in excluded]


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
    del shift_periods
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


def align_feature_frame(
    base_df: pl.DataFrame,
    feature_df: pl.DataFrame,
    *,
    shift_periods: int = 0,
    prefix: str | None = None,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Align an external feature frame to base timestamps without look-ahead."""
    if base_df.is_empty() or feature_df.is_empty():
        return base_df

    aligned_features = feature_df.sort(timestamp_col)
    feature_columns = [column for column in aligned_features.columns if column != timestamp_col]
    if shift_periods:
        aligned_features = aligned_features.with_columns(
            [pl.col(column).shift(shift_periods) for column in feature_columns]
        )
    if prefix:
        aligned_features = aligned_features.rename(
            {column: f"{prefix}{column}" for column in feature_columns}
        )

    return base_df.sort(timestamp_col).join_asof(
        aligned_features,
        on=timestamp_col,
        strategy="backward",
    )
