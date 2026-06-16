"""Market microstructure feature builders."""

from __future__ import annotations

import math

import polars as pl


def _timestamp_col(df: pl.DataFrame) -> str:
    for candidate in ("timestamp", "ts"):
        if candidate in df.columns:
            return candidate
    raise ValueError("DataFrame must include a 'timestamp' or 'ts' column")


def _sort_columns(df: pl.DataFrame, timestamp_col: str) -> list[str]:
    return ["symbol", timestamp_col] if "symbol" in df.columns else [timestamp_col]


def _lag(expr: pl.Expr, df: pl.DataFrame) -> pl.Expr:
    return expr.shift(1).over("symbol") if "symbol" in df.columns else expr.shift(1)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr, default: float = 0.0) -> pl.Expr:
    return (
        pl.when(denominator.is_not_null() & (denominator != 0))
        .then(numerator / denominator)
        .otherwise(default)
    )


def corwin_schultz_spread(df: pl.DataFrame) -> float:
    """Estimate bid-ask spread from high/low bars using Corwin-Schultz."""
    if df.height < 2:
        return 0.0

    log_high_low = (df["high"] / df["low"]).log()
    beta = float((log_high_low**2 + log_high_low.shift(1) ** 2).sum())
    gamma = float(((log_high_low + log_high_low.shift(1)) ** 2).sum())
    if beta <= 0.0 or gamma <= 0.0:
        return 0.0

    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / (3.0 - 2.0 * math.sqrt(2.0))
    spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return max(0.0, float(spread))


def build_quote_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build quote-level spread and size-imbalance features."""
    return df.with_columns(
        [
            ((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2)).alias(
                "quote_spread_pct"
            ),
            (
                (pl.col("bid_size") - pl.col("ask_size"))
                / (pl.col("bid_size") + pl.col("ask_size"))
            ).alias("quote_imbalance"),
        ]
    )


def build_trade_bar_features(df: pl.DataFrame, *, every: str = "1h") -> pl.DataFrame:
    """Aggregate trades into time bars with signed volume imbalance."""
    signed_quantity = (
        pl.when(pl.col("side") == "buy")
        .then(pl.col("quantity"))
        .when(pl.col("side") == "sell")
        .then(-pl.col("quantity"))
        .otherwise(0.0)
    )
    return (
        df.with_columns(signed_quantity.alias("signed_quantity"))
        .group_by_dynamic("timestamp", every=every, group_by="symbol")
        .agg(
            [
                pl.col("price").last().alias("trade_last_price"),
                pl.col("quantity").sum().alias("trade_volume"),
                pl.col("signed_quantity").sum().alias("trade_signed_volume"),
            ]
        )
        .with_columns(
            (pl.col("trade_signed_volume") / pl.col("trade_volume")).alias("trade_imbalance")
        )
        .sort(["symbol", "timestamp"])
    )


def build_order_book_features(df: pl.DataFrame, *, depth: int = 1) -> pl.DataFrame:
    """Build depth-limited order-book imbalance features."""
    depth_df = df.filter(pl.col("level") < depth)
    side_sizes = (
        depth_df.group_by(["symbol", "timestamp", "side"])
        .agg(pl.col("size").sum().alias("size"))
        .pivot(index=["symbol", "timestamp"], on="side", values="size")
    )
    bid = pl.col("bid") if "bid" in side_sizes.columns else pl.lit(0.0)
    ask = pl.col("ask") if "ask" in side_sizes.columns else pl.lit(0.0)
    return side_sizes.with_columns(
        ((bid - ask) / (bid + ask)).fill_nan(0.0).alias("depth_imbalance")
    ).select(["symbol", "timestamp", "depth_imbalance"])
