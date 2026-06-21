"""Microstructure feature builders."""

from __future__ import annotations

import math

import polars as pl


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

    if "high" not in df.columns or "low" not in df.columns:
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
    """Build quote-level spread and imbalance features."""
    for col in ("bid", "ask"):
        if col not in df.columns:
            raise ValueError("Missing required columns")

    sort_expr = ["symbol", "timestamp"] if "symbol" in df.columns else ["timestamp"]
    out = (
        df.sort(sort_expr)
        .with_columns(
            [
                ((pl.col("bid") + pl.col("ask")) / 2).alias("quote_mid"),
                (pl.col("ask") - pl.col("bid")).alias("quoted_spread"),
            ]
        )
        .with_columns(
            [
                _safe_ratio(pl.col("quoted_spread") * 10000.0, pl.col("quote_mid"), 0.0).alias(
                    "quoted_spread_bps"
                ),
                _safe_ratio(
                    pl.col("quote_mid") - _lag(pl.col("quote_mid"), df),
                    _lag(pl.col("quote_mid"), df),
                    0.0,
                ).alias("mid_return"),
                _safe_ratio(
                    pl.col("quoted_spread") - _lag(pl.col("quoted_spread"), df),
                    _lag(pl.col("quoted_spread"), df),
                    0.0,
                ).alias("spread_change"),
            ]
        )
    )

    if "bid_size" in df.columns and "ask_size" in df.columns:
        out = out.with_columns(
            [
                _safe_ratio(
                    pl.col("bid_size") - pl.col("ask_size"),
                    pl.col("bid_size") + pl.col("ask_size"),
                    0.0,
                ).alias("quote_imbalance"),
                _safe_ratio(
                    pl.col("bid") * pl.col("ask_size") + pl.col("ask") * pl.col("bid_size"),
                    pl.col("bid_size") + pl.col("ask_size"),
                    0.0,
                ).alias("microprice"),
            ]
        )
    else:
        out = out.with_columns(
            [pl.lit(0.0).alias("quote_imbalance"), pl.col("quote_mid").alias("microprice")]
        )

    out = out.with_columns(
        (
            _safe_ratio(
                pl.col("microprice") - pl.col("quote_mid"),
                pl.col("quote_mid"),
                0.0,
            )
            * 10000.0
        ).alias("microprice_edge_bps")
    )

    return out


def build_trade_bar_features(df: pl.DataFrame, *, every: str = "1h") -> pl.DataFrame:
    """Aggregate trades into bars with signed and imbalance quantities."""
    for col in ("timestamp", "price", "quantity", "side"):
        if col not in df.columns:
            raise ValueError("Missing required columns")

    if "symbol" not in df.columns:
        raise ValueError("Missing required columns")

    side = pl.col("side")
    signed_quantity = (
        pl.when(side == "buy")
        .then(pl.col("quantity"))
        .when(side == "sell")
        .then(-pl.col("quantity"))
        .otherwise(0.0)
    )

    traded = df.with_columns(
        [
            signed_quantity.alias("signed_quantity"),
            (pl.col("price") * pl.col("quantity")).alias("notional"),
        ]
    )

    agg = (
        traded.group_by_dynamic("timestamp", every=every, group_by="symbol")
        .agg(
            [
                pl.col("timestamp").count().alias("trade_count"),
                pl.col("price").last().alias("trade_last_price"),
                pl.col("quantity").sum().alias("trade_volume"),
                pl.col("notional").sum().alias("notional"),
                pl.col("signed_quantity").sum().alias("trade_signed_volume"),
            ]
        )
        .with_columns(
            [
                _safe_ratio(pl.col("notional"), pl.col("trade_volume")).alias("trade_vwap"),
                _safe_ratio(pl.col("trade_signed_volume"), pl.col("trade_volume")).alias(
                    "trade_imbalance"
                ),
                pl.lit(0).alias("quote_count"),
            ]
        )
        .sort(["symbol", "timestamp"])
    )

    return agg.select(
        [
            "symbol",
            "timestamp",
            "trade_count",
            "trade_last_price",
            "trade_volume",
            "trade_vwap",
            "trade_signed_volume",
            "trade_imbalance",
        ]
    )


def build_order_book_features(df: pl.DataFrame, *, depth: int = 1) -> pl.DataFrame:
    """Build depth-limited order-book imbalance features."""
    if depth <= 0:
        raise ValueError("depth must be positive")

    for col in ("timestamp", "symbol", "side", "price", "size", "level"):
        if col not in df.columns:
            raise ValueError("Missing required columns")

    depth_df = df.filter(pl.col("level") < depth)

    bid = depth_df.filter(pl.col("side") == "bid")
    ask = depth_df.filter(pl.col("side") == "ask")

    bid_agg = bid.group_by(["symbol", "timestamp"]).agg(
        [pl.col("price").max().alias("best_bid"), pl.col("size").sum().alias("bid_size")]
    )
    ask_agg = ask.group_by(["symbol", "timestamp"]).agg(
        [pl.col("price").min().alias("best_ask"), pl.col("size").sum().alias("ask_size")]
    )

    key_cols = list({"symbol", "timestamp"}.intersection(df.columns))
    if not key_cols:
        key_cols = ["timestamp"]
    keys = df.select(key_cols).unique().sort(key_cols)

    out = keys.join(bid_agg, on=key_cols, how="left").join(ask_agg, on=key_cols, how="left")
    out = out.with_columns(
        [
            (pl.col("best_ask") - pl.col("best_bid")).alias("quoted_spread"),
            _safe_ratio(
                pl.col("bid_size") - pl.col("ask_size"),
                pl.col("bid_size") + pl.col("ask_size"),
                0.0,
            ).alias("depth_imbalance"),
        ]
    )

    return out.select(
        [
            "symbol",
            "timestamp",
            "best_bid",
            "best_ask",
            "quoted_spread",
            "depth_imbalance",
        ]
    )


def build_funding_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build funding-rate derived microstructure features."""
    for col in ("funding_rate", "mark_price"):
        if col not in df.columns:
            raise ValueError("Missing required columns")
    if "timestamp" not in df.columns:
        raise ValueError("Missing required columns")

    out = df.with_columns(
        [
            (pl.col("funding_rate") * 10000.0).alias("funding_rate_bps"),
            pl.when(pl.col("funding_rate") > 0)
            .then(1)
            .when(pl.col("funding_rate") < 0)
            .then(-1)
            .otherwise(0)
            .alias("funding_direction"),
            _safe_ratio(
                pl.col("funding_rate") - _lag(pl.col("funding_rate"), df),
                pl.col("funding_rate"),
                0.0,
            ).alias("funding_rate_change"),
            _safe_ratio(
                pl.col("mark_price") - _lag(pl.col("mark_price"), df),
                _lag(pl.col("mark_price"), df),
                0.0,
            ).alias("mark_price_return"),
        ]
    )

    out = out.with_columns(
        [
            pl.col("funding_rate_change").fill_null(0.0),
            pl.col("mark_price_return").fill_null(0.0),
        ]
    )
    return out


def build_open_interest_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build open-interest derived growth and level features."""
    for col in ("open_interest", "open_interest_value", "circulating_supply"):
        if col not in df.columns:
            raise ValueError("Missing required columns")
    if "timestamp" not in df.columns:
        raise ValueError("Missing required columns")

    out = df.with_columns(
        [
            (pl.col("open_interest") - _lag(pl.col("open_interest"), df)).alias(
                "open_interest_change"
            ),
            _safe_ratio(
                pl.col("open_interest") - _lag(pl.col("open_interest"), df),
                _lag(pl.col("open_interest"), df),
                0.0,
            ).alias("open_interest_pct_change"),
            _safe_ratio(
                pl.col("open_interest_value") - _lag(pl.col("open_interest_value"), df),
                _lag(pl.col("open_interest_value"), df),
                0.0,
            ).alias("open_interest_value_pct_change"),
            _safe_ratio(pl.col("open_interest"), pl.col("circulating_supply"), 0.0).alias(
                "oi_to_supply"
            ),
        ]
    )

    out = out.with_columns(
        [
            pl.col("open_interest_change").fill_null(0.0),
            pl.col("open_interest_pct_change").fill_null(0.0),
            pl.col("open_interest_value_pct_change").fill_null(0.0),
        ]
    )
    return out
