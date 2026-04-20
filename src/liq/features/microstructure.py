"""Provider-agnostic microstructure and market-state feature builders."""

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
    """Estimate bid-ask spread using Corwin-Schultz estimator."""
    if df.height < 2:
        return 0.0
    log_hl = (df["high"] / df["low"]).log()
    beta = (log_hl**2 + log_hl.shift(1) ** 2).sum()
    gamma = ((log_hl + log_hl.shift(1)) ** 2).sum()
    beta = float(beta)
    gamma = float(gamma)
    if gamma == 0 or beta == 0:
        return 0.0
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / (3 - 2 * math.sqrt(2))
    spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    return spread


def build_quote_features(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Add quote-level features from normalized bid/ask snapshots.

    Required columns:
    - `timestamp` or `ts`
    - `bid`
    - `ask`

    Optional columns:
    - `symbol`
    - `bid_size`
    - `ask_size`
    """
    required = {"bid", "ask"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    timestamp_col = _timestamp_col(df_raw)
    df = df_raw.sort(_sort_columns(df_raw, timestamp_col))

    bid = pl.col("bid")
    ask = pl.col("ask")
    mid = (bid + ask) / 2.0
    spread = ask - bid
    prev_mid = _lag(mid, df)
    prev_spread = _lag(spread, df)

    features: list[pl.Expr] = [
        mid.alias("quote_mid"),
        spread.alias("quoted_spread"),
        (_safe_ratio(spread, mid) * 10_000.0).alias("quoted_spread_bps"),
        _safe_ratio(mid - prev_mid, prev_mid).alias("mid_return"),
        (spread - prev_spread).fill_null(0.0).alias("spread_change"),
    ]

    if {"bid_size", "ask_size"} <= set(df.columns):
        bid_size = pl.col("bid_size")
        ask_size = pl.col("ask_size")
        total_size = bid_size + ask_size
        microprice = _safe_ratio((ask * bid_size) + (bid * ask_size), total_size)
        features.extend(
            [
                _safe_ratio(bid_size - ask_size, total_size).alias("quote_imbalance"),
                microprice.alias("microprice"),
                (_safe_ratio(microprice - mid, mid) * 10_000.0).alias("microprice_edge_bps"),
            ]
        )

    return df.with_columns(features)


def build_trade_bar_features(
    df_raw: pl.DataFrame,
    *,
    every: str = "1m",
) -> pl.DataFrame:
    """Aggregate normalized trades into bar-aligned trade features.

    Required columns:
    - `timestamp` or `ts`
    - `price`
    - `quantity`
    - `side`
    """
    required = {"price", "quantity", "side"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    timestamp_col = _timestamp_col(df_raw)
    sort_cols = _sort_columns(df_raw, timestamp_col)
    df = df_raw.sort(sort_cols).with_columns(
        [
            (pl.col("price") * pl.col("quantity")).alias("_notional"),
            pl.when(pl.col("side") == "buy")
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .alias("_buy_quantity"),
            pl.when(pl.col("side") == "sell")
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .alias("_sell_quantity"),
            pl.when(pl.col("side") == "buy")
            .then(pl.col("price") * pl.col("quantity"))
            .otherwise(0.0)
            .alias("_buy_notional"),
            pl.when(pl.col("side") == "sell")
            .then(pl.col("price") * pl.col("quantity"))
            .otherwise(0.0)
            .alias("_sell_notional"),
        ]
    )

    group_by: list[str] | None = ["symbol"] if "symbol" in df.columns else None
    if group_by:
        aggregated = df.group_by_dynamic(timestamp_col, every=every, group_by=group_by).agg(
            [
                pl.len().alias("trade_count"),
                pl.col("quantity").sum().alias("trade_volume"),
                pl.col("_notional").sum().alias("trade_notional"),
                pl.col("_buy_quantity").sum().alias("buy_volume"),
                pl.col("_sell_quantity").sum().alias("sell_volume"),
                pl.col("_buy_notional").sum().alias("buy_notional"),
                pl.col("_sell_notional").sum().alias("sell_notional"),
            ]
        )
    else:
        aggregated = df.group_by_dynamic(timestamp_col, every=every).agg(
            [
                pl.len().alias("trade_count"),
                pl.col("quantity").sum().alias("trade_volume"),
                pl.col("_notional").sum().alias("trade_notional"),
                pl.col("_buy_quantity").sum().alias("buy_volume"),
                pl.col("_sell_quantity").sum().alias("sell_volume"),
                pl.col("_buy_notional").sum().alias("buy_notional"),
                pl.col("_sell_notional").sum().alias("sell_notional"),
            ]
        )

    trade_volume = pl.col("trade_volume")
    trade_notional = pl.col("trade_notional")
    buy_volume = pl.col("buy_volume")
    sell_volume = pl.col("sell_volume")
    buy_notional = pl.col("buy_notional")
    sell_notional = pl.col("sell_notional")

    return aggregated.sort(_sort_columns(aggregated, timestamp_col)).with_columns(
        [
            _safe_ratio(trade_notional, trade_volume).alias("trade_vwap"),
            _safe_ratio(buy_volume - sell_volume, trade_volume).alias("trade_imbalance"),
            _safe_ratio(buy_notional - sell_notional, trade_notional).alias(
                "trade_notional_imbalance"
            ),
            _safe_ratio(trade_volume, pl.col("trade_count")).alias("avg_trade_size"),
        ]
    )


def build_order_book_features(
    df_raw: pl.DataFrame,
    *,
    depth: int = 5,
) -> pl.DataFrame:
    """Collapse normalized order-book rows into snapshot-level features."""
    required = {"snapshot_id", "side", "price", "size", "level"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if depth <= 0:
        raise ValueError("depth must be positive")

    timestamp_col = _timestamp_col(df_raw)
    sort_cols = _sort_columns(df_raw, timestamp_col) + ["snapshot_id", "side", "level"]
    df = df_raw.sort(sort_cols).filter(pl.col("level") < depth)
    group_keys = [timestamp_col, "snapshot_id"]
    if "symbol" in df.columns:
        group_keys.insert(0, "symbol")

    grouped = df.group_by(group_keys).agg(
        [
            pl.col("price").filter(pl.col("side") == "bid").max().alias("best_bid"),
            pl.col("price").filter(pl.col("side") == "ask").min().alias("best_ask"),
            pl.col("size").filter(pl.col("side") == "bid").sum().alias("bid_depth_total"),
            pl.col("size").filter(pl.col("side") == "ask").sum().alias("ask_depth_total"),
            pl.col("size")
            .filter((pl.col("side") == "bid") & (pl.col("level") == 0))
            .first()
            .alias("best_bid_size"),
            pl.col("size")
            .filter((pl.col("side") == "ask") & (pl.col("level") == 0))
            .first()
            .alias("best_ask_size"),
        ]
    )

    best_bid = pl.col("best_bid")
    best_ask = pl.col("best_ask")
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2.0
    bid_depth = pl.col("bid_depth_total")
    ask_depth = pl.col("ask_depth_total")
    best_bid_size = pl.col("best_bid_size")
    best_ask_size = pl.col("best_ask_size")
    top_size = best_bid_size + best_ask_size

    return grouped.sort(_sort_columns(grouped, timestamp_col) + ["snapshot_id"]).with_columns(
        [
            spread.alias("quoted_spread"),
            (_safe_ratio(spread, mid) * 10_000.0).alias("quoted_spread_bps"),
            _safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth).alias("depth_imbalance"),
            _safe_ratio((best_ask * best_bid_size) + (best_bid * best_ask_size), top_size).alias(
                "microprice"
            ),
        ]
    )


def build_funding_features(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Add simple funding-state features from normalized funding observations."""
    if "funding_rate" not in df_raw.columns:
        raise ValueError("Missing required columns: ['funding_rate']")

    timestamp_col = _timestamp_col(df_raw)
    df = df_raw.sort(_sort_columns(df_raw, timestamp_col))
    funding_rate = pl.col("funding_rate")
    prev_rate = _lag(funding_rate, df)

    features: list[pl.Expr] = [
        (funding_rate * 10_000.0).alias("funding_rate_bps"),
        funding_rate.abs().alias("funding_rate_abs"),
        (funding_rate.abs() * 10_000.0).alias("funding_rate_abs_bps"),
        (funding_rate - prev_rate).fill_null(0.0).alias("funding_rate_change"),
        pl.when(funding_rate > 0)
        .then(1)
        .when(funding_rate < 0)
        .then(-1)
        .otherwise(0)
        .alias("funding_direction"),
    ]

    if "mark_price" in df.columns:
        mark_price = pl.col("mark_price")
        prev_mark = _lag(mark_price, df)
        features.append(_safe_ratio(mark_price - prev_mark, prev_mark).alias("mark_price_return"))

    return df.with_columns(features)


def build_open_interest_features(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Add simple open-interest-state features from normalized OI observations."""
    if "open_interest" not in df_raw.columns:
        raise ValueError("Missing required columns: ['open_interest']")

    timestamp_col = _timestamp_col(df_raw)
    df = df_raw.sort(_sort_columns(df_raw, timestamp_col))
    oi = pl.col("open_interest")
    prev_oi = _lag(oi, df)

    features: list[pl.Expr] = [
        (oi - prev_oi).fill_null(0.0).alias("open_interest_change"),
        _safe_ratio(oi - prev_oi, prev_oi).alias("open_interest_pct_change"),
    ]

    if "open_interest_value" in df.columns:
        oi_value = pl.col("open_interest_value")
        prev_value = _lag(oi_value, df)
        features.extend(
            [
                (oi_value - prev_value).fill_null(0.0).alias("open_interest_value_change"),
                _safe_ratio(oi_value - prev_value, prev_value).alias(
                    "open_interest_value_pct_change"
                ),
            ]
        )

    if "circulating_supply" in df.columns:
        features.append(_safe_ratio(oi, pl.col("circulating_supply")).alias("oi_to_supply"))

    return df.with_columns(features)
