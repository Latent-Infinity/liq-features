"""Volatility estimators (Yang-Zhang, Garman-Klass)."""

from __future__ import annotations

import polars as pl


def yang_zhang(df: pl.DataFrame) -> float:
    """Compute Yang-Zhang volatility (annualized)."""
    if df.height < 2:
        return 0.0
    log_ho = (df["high"] / df["open"]).log()
    log_lo = (df["low"] / df["open"]).log()
    log_co = (df["close"] / df["open"]).log()
    log_oc = (df["open"] / df["close"].shift(1)).log().fill_null(0)
    rs = (log_ho * (log_ho - log_lo)).sum() / (df.height - 1)
    close_vol = log_co.var()
    open_vol = log_oc.var()
    return float((open_vol + 0.164333 * close_vol + 0.835667 * rs) ** 0.5)


def garman_klass(df: pl.DataFrame) -> float:
    """Compute Garman-Klass volatility (annualized)."""
    if df.height < 2:
        return 0.0
    log_hl = (df["high"] / df["low"]).log()
    log_co = (df["close"] / df["open"]).log()
    return float(((0.5 * log_hl ** 2) - (2 * log_co ** 2)).mean() ** 0.5)
