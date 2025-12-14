"""Time series aggregation for OHLCV data.

This module provides efficient aggregation of high-frequency OHLCV data
to lower frequencies (e.g., 1min -> 5min -> 1h -> 1d).

Design Principles:
    - SRP: Only handles time aggregation
    - KISS: Simple polars group_by operations
    - DIP: Works with any DataFrame with required columns

Example:
    >>> from liq.features.aggregation import Aggregator
    >>>
    >>> # Aggregate 1-minute bars to hourly
    >>> agg = Aggregator(source_timeframe="1min", target_timeframe="1h")
    >>> hourly_bars = agg.aggregate(minute_bars)
"""

from datetime import timedelta

import polars as pl


class Aggregator:
    """Aggregates OHLCV time series to larger timeframes.

    Transforms high-frequency candles to lower frequency using standard
    OHLCV aggregation rules:
    - open: first value in period
    - high: maximum value in period
    - low: minimum value in period
    - close: last value in period
    - volume: sum of values in period

    Example:
        >>> agg = Aggregator(source_timeframe="1min", target_timeframe="1h")
        >>> hourly_data = agg.aggregate(minute_data)
    """

    # Timeframe to timedelta mapping
    TIMEFRAME_DELTAS: dict[str, timedelta] = {
        "1min": timedelta(minutes=1),
        "1m": timedelta(minutes=1),
        "5min": timedelta(minutes=5),
        "5m": timedelta(minutes=5),
        "15min": timedelta(minutes=15),
        "15m": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    def __init__(
        self,
        source_timeframe: str,
        target_timeframe: str,
        timestamp_col: str = "timestamp",
    ) -> None:
        """Initialize aggregator.

        Args:
            source_timeframe: Source data timeframe (e.g., "1min")
            target_timeframe: Target timeframe (e.g., "1h")
            timestamp_col: Name of timestamp column (default: "timestamp")

        Raises:
            ValueError: If timeframes are invalid or incompatible
        """
        if source_timeframe not in self.TIMEFRAME_DELTAS:
            raise ValueError(
                f"Unknown source timeframe: {source_timeframe}. "
                f"Valid: {list(self.TIMEFRAME_DELTAS.keys())}"
            )

        if target_timeframe not in self.TIMEFRAME_DELTAS:
            raise ValueError(
                f"Unknown target timeframe: {target_timeframe}. "
                f"Valid: {list(self.TIMEFRAME_DELTAS.keys())}"
            )

        source_delta = self.TIMEFRAME_DELTAS[source_timeframe]
        target_delta = self.TIMEFRAME_DELTAS[target_timeframe]

        if target_delta <= source_delta:
            raise ValueError(
                f"Target timeframe ({target_timeframe}) must be larger than "
                f"source timeframe ({source_timeframe})"
            )

        self._source_timeframe = source_timeframe
        self._target_timeframe = target_timeframe
        self._timestamp_col = timestamp_col
        self._target_delta = target_delta

    @property
    def source_timeframe(self) -> str:
        """Get the source timeframe."""
        return self._source_timeframe

    @property
    def target_timeframe(self) -> str:
        """Get the target timeframe."""
        return self._target_timeframe

    def aggregate(
        self, df: pl.DataFrame, include_incomplete: bool = False
    ) -> pl.DataFrame:
        """Aggregate OHLCV data to target timeframe.

        Args:
            df: DataFrame with OHLCV columns and timestamp
            include_incomplete: If False (default), exclude partial bars at the end.
                Per PRD §7.3, incomplete bars should be excluded for backtesting.

        Returns:
            Aggregated DataFrame at target timeframe

        Raises:
            ValueError: If required columns are missing

        Design:
            - SRP: Only performs aggregation
            - KISS: Simple polars group_by
        """
        required_cols = {self._timestamp_col, "open", "high", "low", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.is_empty():
            return df

        # Truncate timestamps to target timeframe
        truncated = df.with_columns(
            [
                pl.col(self._timestamp_col)
                .dt.truncate(self._format_truncate_interval())
                .alias("_period"),
            ]
        )

        # Build aggregation expressions
        agg_exprs = [
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.len().alias("_bar_count"),  # Track bar count for completeness check
        ]

        # Add volume if present
        if "volume" in df.columns:
            agg_exprs.append(pl.col("volume").sum().alias("volume"))

        # Perform aggregation
        result = (
            truncated.sort(self._timestamp_col)
            .group_by("_period")
            .agg(agg_exprs)
            .rename({"_period": self._timestamp_col})
            .sort(self._timestamp_col)
        )

        # Filter incomplete bars if requested
        if not include_incomplete and len(result) > 0:
            # Calculate expected bars per period
            source_delta = self.TIMEFRAME_DELTAS[self._source_timeframe]
            expected_bars = int(self._target_delta / source_delta)

            # Keep only bars with expected count (complete periods)
            result = result.filter(pl.col("_bar_count") >= expected_bars)

        # Remove internal column
        result = result.drop("_bar_count")

        return result

    def _format_truncate_interval(self) -> str:
        """Format the target timeframe as a polars truncate interval string.

        Returns:
            Interval string for polars dt.truncate()
        """
        delta = self._target_delta

        if delta >= timedelta(weeks=1):
            return f"{delta.days // 7}w"
        elif delta >= timedelta(days=1):
            return f"{delta.days}d"
        elif delta >= timedelta(hours=1):
            return f"{int(delta.total_seconds() // 3600)}h"
        else:
            return f"{int(delta.total_seconds() // 60)}m"


def aggregate_to_timeframe(
    df: pl.DataFrame,
    source_timeframe: str,
    target_timeframe: str,
    timestamp_col: str = "timestamp",
    include_incomplete: bool = False,
) -> pl.DataFrame:
    """Convenience function to aggregate OHLCV data.

    Args:
        df: DataFrame with OHLCV columns
        source_timeframe: Source data timeframe (e.g., "1min")
        target_timeframe: Target timeframe (e.g., "1h")
        timestamp_col: Name of timestamp column
        include_incomplete: If False (default), exclude partial bars at the end.

    Returns:
        Aggregated DataFrame

    Example:
        >>> hourly = aggregate_to_timeframe(minute_bars, "1min", "1h")
        >>> # Exclude incomplete final hour
        >>> hourly = aggregate_to_timeframe(minute_bars, "1min", "1h", include_incomplete=False)
    """
    agg = Aggregator(
        source_timeframe=source_timeframe,
        target_timeframe=target_timeframe,
        timestamp_col=timestamp_col,
    )
    return agg.aggregate(df, include_incomplete=include_incomplete)
