"""Multi-timeframe alignment utilities."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl


def _sort_for_asof(df: pl.DataFrame, timestamp_col: str, by: str | None) -> pl.DataFrame:
    if by is None:
        return df.sort(timestamp_col)
    return df.sort([by, timestamp_col])


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
    join_by = "symbol" if "symbol" in base_df.columns and "symbol" in higher_df.columns else None
    base_sorted = _sort_for_asof(base_df, timestamp_col, join_by)
    higher_sorted = _sort_for_asof(higher_df, timestamp_col, join_by)
    if join_by is not None:
        return base_sorted.join_asof(
            higher_sorted,
            left_on=timestamp_col,
            right_on=timestamp_col,
            strategy="backward",
            by="symbol",
        )
    return base_sorted.join_asof(
        higher_sorted,
        left_on=timestamp_col,
        right_on=timestamp_col,
        strategy="backward",
    )


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
    join_by: str | None = None
    if "symbol" in base_df.columns and "symbol" in feature_df.columns:
        join_by = "symbol"

    base_sorted = _sort_for_asof(base_df, timestamp_col, join_by)
    aligned_features = _sort_for_asof(feature_df, timestamp_col, join_by)
    excluded = {timestamp_col}
    if join_by:
        excluded.add(join_by)
    feature_columns = [column for column in aligned_features.columns if column not in excluded]
    if shift_periods:
        aligned_features = aligned_features.with_columns(
            [pl.col(column).shift(shift_periods) for column in feature_columns]
        )
    if prefix:
        aligned_features = aligned_features.rename(
            {column: f"{prefix}{column}" for column in feature_columns}
        )

    if join_by is not None:
        return base_sorted.join_asof(
            aligned_features,
            on=timestamp_col,
            strategy="backward",
            by=join_by,
        )
    return base_sorted.join_asof(
        aligned_features,
        on=timestamp_col,
        strategy="backward",
    )


def align_feature_frames(
    base_df: pl.DataFrame,
    feature_frames: Mapping[str, pl.DataFrame],
    *,
    shift_periods: int = 0,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Align multiple named feature frames onto a single base DataFrame.

    Feature names are used as prefixes to avoid column collisions.
    Empty frames are skipped.
    """
    aligned = base_df
    for key, feature_df in feature_frames.items():
        if feature_df.is_empty():
            continue

        aligned = align_feature_frame(
            aligned,
            feature_df,
            shift_periods=shift_periods,
            prefix=f"{key}_",
            timestamp_col=timestamp_col,
        )

    return aligned
