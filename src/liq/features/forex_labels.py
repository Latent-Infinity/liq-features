"""Forex label helpers"""

from __future__ import annotations

import polars as pl


def _true_range(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    return pl.coalesce([tr, tr1])


def make_forex_labels(
    df: pl.DataFrame,
    *,
    horizon_bars: int = 72,
    price_col: str = "close",
    atr_window: int = 14,
    threshold_k: float = 0.5,
    spread_bps_col: str | None = None,
) -> pl.DataFrame:
    """Create future return and 3-class direction labels.

    Label mapping: 0=DOWN, 1=HOLD, 2=UP.
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not in DataFrame")

    price = pl.col(price_col)
    future_price = price.shift(-horizon_bars)
    future_return = (future_price.log() - price.log()).alias("future_return")

    if spread_bps_col and spread_bps_col in df.columns:
        threshold = (pl.col(spread_bps_col) * (threshold_k / 10000.0)).alias("threshold")
    else:
        if not {"high", "low", "close"}.issubset(df.columns):
            raise ValueError("ATR threshold requires high/low/close columns")
        tr = _true_range(pl.col("high"), pl.col("low"), pl.col("close"))
        atr = tr.rolling_mean(window_size=atr_window, min_samples=atr_window)
        threshold = (atr * threshold_k).alias("threshold")

    df = df.with_columns([future_return, threshold])

    label = (
        pl.when(pl.col("future_return") > pl.col("threshold"))
        .then(2)
        .when(pl.col("future_return") < -pl.col("threshold"))
        .then(0)
        .otherwise(1)
        .alias("label")
    )

    df = df.with_columns(label)
    df = df.drop_nulls(["future_return", "threshold", "label"])
    return df
