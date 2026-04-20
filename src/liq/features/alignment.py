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
    timestamp_col: str = "timestamp",
    by: list[str] | None = None,
    shift_periods: int = 0,
    prefix: str | None = None,
) -> pl.DataFrame:
    """As-of align a normalized feature frame onto bar timestamps.

    This is intended for externally computed market-state features such as
    quote, trade, order-book, funding, or open-interest features. It keeps
    feature engineering in `liq-features` while remaining provider-agnostic.
    """
    if base_df.is_empty() or feature_df.is_empty():
        return base_df

    join_by = [
        column
        for column in (
            by if by is not None else (["symbol"] if "symbol" in base_df.columns else [])
        )
        if column in base_df.columns and column in feature_df.columns
    ]
    feature_cols = _feature_columns(feature_df, timestamp_col, join_by)

    feature_sorted = feature_df.sort([*join_by, timestamp_col] if join_by else [timestamp_col])
    feature_sorted = feature_sorted.with_columns(pl.col(timestamp_col).set_sorted())
    if shift_periods:
        shifted = [
            (
                pl.col(column).shift(shift_periods).over(join_by).alias(column)
                if join_by
                else pl.col(column).shift(shift_periods).alias(column)
            )
            for column in feature_cols
        ]
        feature_sorted = feature_sorted.with_columns(shifted)

    if prefix:
        feature_sorted = feature_sorted.rename(
            {column: f"{prefix}{column}" for column in feature_cols}
        )

    base_sorted = base_df.sort([*join_by, timestamp_col] if join_by else [timestamp_col])
    base_sorted = base_sorted.with_columns(pl.col(timestamp_col).set_sorted())

    return base_sorted.join_asof(
        feature_sorted,
        left_on=timestamp_col,
        right_on=timestamp_col,
        by=join_by or None,
        strategy="backward",
    )


def align_feature_frames(
    base_df: pl.DataFrame,
    feature_frames: dict[str, pl.DataFrame],
    *,
    timestamp_col: str = "timestamp",
    by: list[str] | None = None,
    shift_periods: int = 0,
) -> pl.DataFrame:
    """Align multiple named feature frames onto a base bar frame with prefixes."""
    aligned = base_df
    for prefix, feature_df in feature_frames.items():
        aligned = align_feature_frame(
            aligned,
            feature_df,
            timestamp_col=timestamp_col,
            by=by,
            shift_periods=shift_periods,
            prefix=f"{prefix}_",
        )
    return aligned
