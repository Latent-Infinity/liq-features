"""Forex-specific feature builder"""

from __future__ import annotations

import polars as pl


def _safe_pct_change(series: pl.Expr) -> pl.Expr:
    prev = series.shift(1)
    return pl.when(prev.is_not_null() & (prev != 0)).then(series / prev - 1).otherwise(0.0)


def _true_range(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    return pl.coalesce([tr, tr1])


def build_features(
    df_raw: pl.DataFrame,
    *,
    atr_window: int = 14,
    zscore_window: int = 50,
    warmup_cut: int | None = None,
) -> pl.DataFrame:
    """Build SCHEMA_FOREX_PRICE features from raw OHLCV data.

    Returns a DataFrame that includes timestamp/symbol columns (if present),
    raw price columns, and feature columns.
    """
    required = {"high", "low", "close"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_raw.sort("timestamp") if "timestamp" in df_raw.columns else df_raw

    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("volume"))

    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")
    volume = pl.col("volume")

    midrange = (high + low) / 2
    log_mid = pl.when(midrange > 0).then(midrange.log()).otherwise(None)
    midrange_ret = log_mid.diff()

    range_val = high - low
    range_ret = _safe_pct_change(range_val)
    volume_ret = _safe_pct_change(volume)

    tr = _true_range(high, low, close)
    atr = tr.rolling_mean(window_size=atr_window, min_samples=atr_window)
    atr_mean = atr.rolling_mean(window_size=zscore_window, min_samples=zscore_window)
    atr_std = atr.rolling_std(window_size=zscore_window, min_samples=zscore_window)
    atr_zscore = (atr - atr_mean) / (atr_std + 1e-10)

    df = df.with_columns(
        [
            midrange_ret.alias("midrange_ret"),
            range_ret.alias("range_ret"),
            volume_ret.alias("volume_ret"),
            atr_zscore.alias("atr_zscore"),
        ]
    )

    cut = warmup_cut if warmup_cut is not None else max(50, zscore_window * 3)
    if len(df) > cut:
        df = df.slice(cut)

    df = df.drop_nulls(["midrange_ret", "range_ret", "volume_ret", "atr_zscore"])
    return df
