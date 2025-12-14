"""Batch computation of technical indicators.

This module provides functions for computing multiple indicators at once
and returning a merged DataFrame with all results.

Example:
    >>> from liq.features import compute_indicators
    >>> from liq.store.parquet import ParquetStore
    >>>
    >>> storage = ParquetStore("/data/features")
    >>> result = compute_indicators(
    ...     bars=ohlcv_df,
    ...     symbol="EUR_USD",
    ...     timeframe="1h",
    ...     indicators=[
    ...         ("rsi", {"period": 14}),
    ...         ("macd", {}),
    ...         ("atr", {"period": 14}),
    ...     ],
    ...     storage=storage,
    ... )
"""

from typing import Any

import polars as pl

from liq.store.protocols import TimeSeriesStore


def compute_indicators(
    bars: pl.DataFrame,
    symbol: str,
    timeframe: str,
    indicators: list[tuple[str, dict[str, Any]]],
    storage: TimeSeriesStore | None = None,
) -> pl.DataFrame:
    """Compute multiple indicators and return merged DataFrame.

    This is a convenience function for batch indicator computation per PRD §9.4.
    All indicators are computed and their results merged into a single DataFrame.

    Args:
        bars: OHLCV DataFrame with timestamp/ts column
        symbol: Symbol for cache key (e.g., "EUR_USD")
        timeframe: Timeframe for cache key (e.g., "1h")
        indicators: List of (indicator_name, params) tuples
            e.g., [("rsi", {"period": 14}), ("macd", {})]
        storage: Optional TimeSeriesStore for caching results

    Returns:
        Merged DataFrame with timestamp and all indicator columns

    Raises:
        ValueError: If unknown indicator is requested

    Example:
        >>> result = compute_indicators(
        ...     bars=df,
        ...     symbol="EUR_USD",
        ...     timeframe="1h",
        ...     indicators=[("rsi", {"period": 14}), ("ema", {"period": 20})],
        ...     storage=store,
        ... )
    """
    from liq.features.indicators import get_indicator

    # Determine timestamp column
    ts_col = "ts" if "ts" in bars.columns else "timestamp"

    # Start with timestamp column
    result = bars.select([pl.col(ts_col)])

    for indicator_name, params in indicators:
        # Get indicator class (raises ValueError if unknown)
        IndicatorClass = get_indicator(indicator_name)

        # Instantiate with storage and params
        indicator = IndicatorClass(storage=storage, params=params)

        # Compute indicator
        indicator_result = indicator.compute(
            bars,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Rename columns to avoid conflicts
        # Skip timestamp column during rename
        rename_map = {}
        for col in indicator_result.columns:
            if col in (ts_col, "ts", "timestamp"):
                continue
            # Prefix with indicator name if generic "value" column
            if col == "value":
                rename_map[col] = f"{indicator_name}"
            else:
                # Keep multi-output names like "macd", "signal", "histogram"
                rename_map[col] = col

        if rename_map:
            indicator_result = indicator_result.rename(rename_map)

        # Join on timestamp
        join_col = ts_col if ts_col in indicator_result.columns else (
            "ts" if "ts" in indicator_result.columns else "timestamp"
        )
        result_join_col = ts_col

        # Select non-timestamp columns from indicator result
        indicator_cols = [c for c in indicator_result.columns if c not in ("ts", "timestamp")]

        if indicator_cols:
            result = result.join(
                indicator_result.select([join_col] + indicator_cols),
                left_on=result_join_col,
                right_on=join_col,
                how="left",
            )

    return result


def cache_stats(storage: TimeSeriesStore) -> pl.DataFrame:
    """Get statistics about cached indicators.

    Introspects the storage to provide metadata about cached indicator
    computations per PRD §8.3.

    Args:
        storage: TimeSeriesStore instance to inspect

    Returns:
        DataFrame with columns:
        - indicator: Indicator name
        - timeframe: Timeframe string
        - params_id: Parameter hash
        - row_count: Number of cached rows
        - size_mb: Approximate size in megabytes

    Example:
        >>> from liq.features import cache_stats
        >>> stats = cache_stats(storage)
        >>> print(stats)
    """
    # Get all keys from storage
    all_keys = storage.list_keys()

    # Filter to indicator keys (format: symbol/indicators/name/params:timeframe)
    indicator_keys = [k for k in all_keys if "/indicators/" in k]

    if not indicator_keys:
        return pl.DataFrame({
            "indicator": [],
            "timeframe": [],
            "params_id": [],
            "row_count": [],
            "size_mb": [],
        }).cast({
            "indicator": pl.Utf8,
            "timeframe": pl.Utf8,
            "params_id": pl.Utf8,
            "row_count": pl.Int64,
            "size_mb": pl.Float64,
        })

    rows = []
    for key in indicator_keys:
        # Parse key: symbol/indicators/name/params_hash:timeframe
        parts = key.split("/")
        if len(parts) >= 4 and parts[1] == "indicators":
            indicator_name = parts[2]
            params_timeframe = parts[3] if len(parts) > 3 else ""

            # Split params:timeframe
            if ":" in params_timeframe:
                params_id, timeframe = params_timeframe.rsplit(":", 1)
            else:
                params_id = params_timeframe
                timeframe = "unknown"

            # Get row count and size
            try:
                df = storage.read(key)
                row_count = len(df)
                # Estimate size (rough approximation)
                size_mb = df.estimated_size() / (1024 * 1024)
            except Exception:
                row_count = 0
                size_mb = 0.0

            rows.append({
                "indicator": indicator_name,
                "timeframe": timeframe,
                "params_id": params_id,
                "row_count": row_count,
                "size_mb": round(size_mb, 4),
            })

    return pl.DataFrame(rows)
