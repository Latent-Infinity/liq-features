"""Trend indicators.

This module provides trend-following technical indicators including
moving averages (EMA, SMA), Bollinger Bands, ADX, and ATR.

Design Principles:
    - SRP: Each indicator handles only its own calculation
    - OCP: New trend indicators can be added without modification
    - DRY: Common patterns abstracted in BaseIndicator
"""

from typing import Any, ClassVar

import polars as pl

from liq.features.indicators.base import BaseIndicator
from liq.features.indicators.registry import register_indicator


@register_indicator
class EMA(BaseIndicator):
    """Exponential Moving Average (EMA) indicator.

    Applies exponential weighting to recent prices, giving more
    importance to recent data.

    Parameters:
        period: Lookback period (default: 20)
        column: Column to compute EMA on (default: "close")
    """

    name: ClassVar[str] = "ema"
    default_params: ClassVar[dict[str, Any]] = {"period": 20, "column": "close"}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute EMA values.

        Args:
            df: DataFrame with price column and 'ts'

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]
        column = self._params["column"]

        result = df.select(
            [
                pl.col("ts"),
                pl.col(column).ewm_mean(span=period, adjust=False).alias("value"),
            ]
        )

        return result.filter(pl.col("value").is_not_nan()).tail(-period)


@register_indicator
class SMA(BaseIndicator):
    """Simple Moving Average (SMA) indicator.

    Calculates the arithmetic mean of prices over a lookback period.

    Parameters:
        period: Lookback period (default: 20)
        column: Column to compute SMA on (default: "close")
    """

    name: ClassVar[str] = "sma"
    default_params: ClassVar[dict[str, Any]] = {"period": 20, "column": "close"}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute SMA values.

        Args:
            df: DataFrame with price column and 'ts'

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]
        column = self._params["column"]

        result = df.select(
            [
                pl.col("ts"),
                pl.col(column).rolling_mean(window_size=period).alias("value"),
            ]
        )

        return result.filter(pl.col("value").is_not_nan())


@register_indicator
class BBANDS(BaseIndicator):
    """Bollinger Bands indicator.

    Calculates upper, middle (SMA), and lower bands based on
    standard deviation of price.

    Parameters:
        period: Lookback period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        column: Column to compute bands on (default: "close")

    Returns:
        DataFrame with columns: ts, upper, middle, lower
    """

    name: ClassVar[str] = "bbands"
    default_params: ClassVar[dict[str, Any]] = {
        "period": 20,
        "std_dev": 2.0,
        "column": "close",
    }

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Bollinger Bands.

        Args:
            df: DataFrame with price column and 'ts'

        Returns:
            DataFrame with columns: ts, upper, middle, lower
        """
        period = self._params["period"]
        std_dev = self._params["std_dev"]
        column = self._params["column"]

        result = df.select(
            [
                pl.col("ts"),
                pl.col(column).rolling_mean(window_size=period).alias("middle"),
                pl.col(column).rolling_std(window_size=period).alias("std"),
            ]
        )

        result = result.with_columns(
            [
                (pl.col("middle") + std_dev * pl.col("std")).alias("upper"),
                (pl.col("middle") - std_dev * pl.col("std")).alias("lower"),
            ]
        )

        return result.select(["ts", "upper", "middle", "lower"]).filter(
            pl.col("middle").is_not_nan()
        )


@register_indicator
class ATR(BaseIndicator):
    """Average True Range (ATR) indicator.

    Measures market volatility by calculating the average of true ranges.

    Parameters:
        period: Lookback period (default: 14)
    """

    name: ClassVar[str] = "atr"
    default_params: ClassVar[dict[str, Any]] = {"period": 14}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute ATR values.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'ts' columns

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]

        # Calculate True Range components
        result = df.select(
            [
                pl.col("ts"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close").shift(1).alias("prev_close"),
            ]
        )

        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        result = result.with_columns(
            [
                pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("prev_close")).abs(),
                    (pl.col("low") - pl.col("prev_close")).abs(),
                ).alias("true_range")
            ]
        )

        # ATR = EMA of True Range (Wilder's smoothing)
        alpha = 1.0 / period
        result = result.with_columns(
            [pl.col("true_range").ewm_mean(alpha=alpha, adjust=False).alias("value")]
        )

        return result.select(["ts", "value"]).filter(pl.col("value").is_not_nan()).tail(-period)


@register_indicator
class ADX(BaseIndicator):
    """Average Directional Index (ADX) indicator.

    Measures trend strength regardless of direction.

    Parameters:
        period: Lookback period (default: 14)

    Returns:
        DataFrame with columns: ts, adx, plus_di, minus_di
    """

    name: ClassVar[str] = "adx"
    default_params: ClassVar[dict[str, Any]] = {"period": 14}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute ADX values.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'ts' columns

        Returns:
            DataFrame with columns: ts, adx, plus_di, minus_di
        """
        period = self._params["period"]
        alpha = 1.0 / period

        # Calculate directional movement
        result = df.select(
            [
                pl.col("ts"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close").shift(1).alias("prev_close"),
                pl.col("high").shift(1).alias("prev_high"),
                pl.col("low").shift(1).alias("prev_low"),
            ]
        )

        # True Range
        result = result.with_columns(
            [
                pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("prev_close")).abs(),
                    (pl.col("low") - pl.col("prev_close")).abs(),
                ).alias("tr"),
                # +DM = high - prev_high (if positive and > low - prev_low)
                pl.when(
                    (pl.col("high") - pl.col("prev_high"))
                    > (pl.col("prev_low") - pl.col("low"))
                )
                .then(
                    pl.when(pl.col("high") - pl.col("prev_high") > 0)
                    .then(pl.col("high") - pl.col("prev_high"))
                    .otherwise(0)
                )
                .otherwise(0)
                .alias("plus_dm"),
                # -DM = prev_low - low (if positive and > high - prev_high)
                pl.when(
                    (pl.col("prev_low") - pl.col("low"))
                    > (pl.col("high") - pl.col("prev_high"))
                )
                .then(
                    pl.when(pl.col("prev_low") - pl.col("low") > 0)
                    .then(pl.col("prev_low") - pl.col("low"))
                    .otherwise(0)
                )
                .otherwise(0)
                .alias("minus_dm"),
            ]
        )

        # Smooth TR and DMs
        result = result.with_columns(
            [
                pl.col("tr").ewm_mean(alpha=alpha, adjust=False).alias("atr"),
                pl.col("plus_dm").ewm_mean(alpha=alpha, adjust=False).alias("smooth_plus_dm"),
                pl.col("minus_dm").ewm_mean(alpha=alpha, adjust=False).alias("smooth_minus_dm"),
            ]
        )

        # Calculate +DI and -DI
        result = result.with_columns(
            [
                (pl.col("smooth_plus_dm") / pl.col("atr") * 100).alias("plus_di"),
                (pl.col("smooth_minus_dm") / pl.col("atr") * 100).alias("minus_di"),
            ]
        )

        # Calculate DX - handle case where both DIs are 0 to avoid NaN
        result = result.with_columns(
            [
                pl.when(pl.col("plus_di") + pl.col("minus_di") == 0)
                .then(0.0)
                .otherwise(
                    (pl.col("plus_di") - pl.col("minus_di")).abs()
                    / (pl.col("plus_di") + pl.col("minus_di"))
                    * 100
                )
                .alias("dx")
            ]
        )

        # ADX = EMA of DX (fill any remaining NaN with 0 to prevent propagation)
        result = result.with_columns(
            [pl.col("dx").fill_nan(0.0).ewm_mean(alpha=alpha, adjust=False).alias("adx")]
        )

        return result.select(["ts", "adx", "plus_di", "minus_di"]).tail(-period * 2)


@register_indicator
class WMA(BaseIndicator):
    """Weighted Moving Average (WMA) indicator.

    Applies linearly decreasing weights to prices, with most recent
    prices having the highest weight.

    Parameters:
        period: Lookback period (default: 20)
        column: Column to compute WMA on (default: "close")
    """

    name: ClassVar[str] = "wma"
    default_params: ClassVar[dict[str, Any]] = {"period": 20, "column": "close"}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute WMA values.

        Args:
            df: DataFrame with price column and 'ts'

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]
        column = self._params["column"]

        # Create weights: 1, 2, 3, ..., n
        weights = list(range(1, period + 1))
        weight_sum = sum(weights)

        # Use rolling_map with weighted average
        # Polars doesn't have built-in WMA, so we compute manually
        values = df[column].to_numpy()
        ts_values = df["ts"]

        import numpy as np

        wma_values = np.full(len(values), np.nan)

        for i in range(period - 1, len(values)):
            window = values[i - period + 1 : i + 1]
            wma_values[i] = np.sum(window * weights) / weight_sum

        result = pl.DataFrame({"ts": ts_values, "value": wma_values})

        return result.filter(pl.col("value").is_not_nan())


@register_indicator
class HMA(BaseIndicator):
    """Hull Moving Average (HMA) indicator.

    A moving average that reduces lag by using weighted sliding windows.
    Considered one of the most responsive moving averages while maintaining
    smoothness.

    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    Parameters:
        period: Lookback period (default: 55, Fibonacci-based)
        column: Column to compute HMA on (default: "close")
    """

    name: ClassVar[str] = "hma"
    default_params: ClassVar[dict[str, Any]] = {"period": 55, "column": "close"}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute HMA values.

        Args:
            df: DataFrame with price column and 'ts'

        Returns:
            DataFrame with columns: ts, value
        """
        import math

        import numpy as np

        period = self._params["period"]
        column = self._params["column"]

        values = df[column].to_numpy()
        ts_values = df["ts"]
        n = len(values)

        # Helper function to compute WMA
        def compute_wma(data: np.ndarray, wma_period: int) -> np.ndarray:
            weights = np.arange(1, wma_period + 1, dtype=float)
            weight_sum = weights.sum()
            result = np.full(len(data), np.nan)
            for i in range(wma_period - 1, len(data)):
                window = data[i - wma_period + 1 : i + 1]
                result[i] = np.sum(window * weights) / weight_sum
            return result

        # Step 1: WMA(n/2)
        half_period = max(1, period // 2)
        wma_half = compute_wma(values, half_period)

        # Step 2: WMA(n)
        wma_full = compute_wma(values, period)

        # Step 3: 2*WMA(n/2) - WMA(n)
        diff = 2 * wma_half - wma_full

        # Step 4: WMA of diff with period = sqrt(n)
        sqrt_period = max(1, int(math.sqrt(period)))
        hma_values = compute_wma(diff, sqrt_period)

        result = pl.DataFrame({"ts": ts_values, "value": hma_values})

        return result.filter(pl.col("value").is_not_nan())


@register_indicator
class ATR_Midrange(BaseIndicator):
    """Average True Range using Midrange price.

    This indicator computes ATR using the midrange price (high+low)/2
    instead of close price. This is more representative of true market
    behavior as it captures the center of price action.

    Parameters:
        period: Lookback period (default: 14)
        input_column: Column name for midrange price (default: "midrange")
    """

    name: ClassVar[str] = "atr_midrange"
    default_params: ClassVar[dict[str, Any]] = {"period": 14, "input_column": "midrange"}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute ATR using midrange values.

        If input_column exists in df, uses it directly.
        Otherwise, computes midrange from high and low.

        Args:
            df: DataFrame with 'high', 'low', 'ts' columns

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]
        input_col = self._params["input_column"]

        # Compute midrange if not present
        if input_col not in df.columns:
            df = df.with_columns(
                [((pl.col("high") + pl.col("low")) / 2).alias(input_col)]
            )

        # Calculate True Range using midrange instead of close
        result = df.select(
            [
                pl.col("ts"),
                pl.col("high"),
                pl.col("low"),
                pl.col(input_col).shift(1).alias("prev_midrange"),
            ]
        )

        # True Range = max(high-low, |high-prev_midrange|, |low-prev_midrange|)
        result = result.with_columns(
            [
                pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("prev_midrange")).abs(),
                    (pl.col("low") - pl.col("prev_midrange")).abs(),
                ).alias("true_range")
            ]
        )

        # ATR = EMA of True Range (Wilder's smoothing)
        alpha = 1.0 / period
        result = result.with_columns(
            [pl.col("true_range").ewm_mean(alpha=alpha, adjust=False).alias("value")]
        )

        return result.select(["ts", "value"]).filter(pl.col("value").is_not_nan()).tail(-period)


@register_indicator
class ADX_Midrange(BaseIndicator):
    """Average Directional Index using Midrange price.

    This indicator computes ADX using the midrange price (high+low)/2
    for directional movement calculations.

    Parameters:
        period: Lookback period (default: 14)
        input_column: Column name for midrange price (default: "midrange")

    Returns:
        DataFrame with columns: ts, adx, plus_di, minus_di
    """

    name: ClassVar[str] = "adx_midrange"
    default_params: ClassVar[dict[str, Any]] = {"period": 14, "input_column": "midrange"}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute ADX using midrange values.

        Args:
            df: DataFrame with 'high', 'low', 'ts' columns

        Returns:
            DataFrame with columns: ts, adx, plus_di, minus_di
        """
        period = self._params["period"]
        input_col = self._params["input_column"]
        alpha = 1.0 / period

        # Compute midrange if not present
        if input_col not in df.columns:
            df = df.with_columns(
                [((pl.col("high") + pl.col("low")) / 2).alias(input_col)]
            )

        # Calculate directional movement using midrange
        result = df.select(
            [
                pl.col("ts"),
                pl.col("high"),
                pl.col("low"),
                pl.col(input_col).shift(1).alias("prev_midrange"),
                pl.col("high").shift(1).alias("prev_high"),
                pl.col("low").shift(1).alias("prev_low"),
            ]
        )

        # True Range using midrange
        result = result.with_columns(
            [
                pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("prev_midrange")).abs(),
                    (pl.col("low") - pl.col("prev_midrange")).abs(),
                ).alias("tr"),
                # +DM = high - prev_high (if positive and > low - prev_low)
                pl.when(
                    (pl.col("high") - pl.col("prev_high"))
                    > (pl.col("prev_low") - pl.col("low"))
                )
                .then(
                    pl.when(pl.col("high") - pl.col("prev_high") > 0)
                    .then(pl.col("high") - pl.col("prev_high"))
                    .otherwise(0)
                )
                .otherwise(0)
                .alias("plus_dm"),
                # -DM = prev_low - low (if positive and > high - prev_high)
                pl.when(
                    (pl.col("prev_low") - pl.col("low"))
                    > (pl.col("high") - pl.col("prev_high"))
                )
                .then(
                    pl.when(pl.col("prev_low") - pl.col("low") > 0)
                    .then(pl.col("prev_low") - pl.col("low"))
                    .otherwise(0)
                )
                .otherwise(0)
                .alias("minus_dm"),
            ]
        )

        # Smooth TR and DMs
        result = result.with_columns(
            [
                pl.col("tr").ewm_mean(alpha=alpha, adjust=False).alias("atr"),
                pl.col("plus_dm").ewm_mean(alpha=alpha, adjust=False).alias("smooth_plus_dm"),
                pl.col("minus_dm").ewm_mean(alpha=alpha, adjust=False).alias("smooth_minus_dm"),
            ]
        )

        # Calculate +DI and -DI
        result = result.with_columns(
            [
                (pl.col("smooth_plus_dm") / pl.col("atr") * 100).alias("plus_di"),
                (pl.col("smooth_minus_dm") / pl.col("atr") * 100).alias("minus_di"),
            ]
        )

        # Calculate DX - handle case where both DIs are 0 to avoid NaN
        result = result.with_columns(
            [
                pl.when(pl.col("plus_di") + pl.col("minus_di") == 0)
                .then(0.0)
                .otherwise(
                    (pl.col("plus_di") - pl.col("minus_di")).abs()
                    / (pl.col("plus_di") + pl.col("minus_di"))
                    * 100
                )
                .alias("dx")
            ]
        )

        # ADX = EMA of DX (fill any remaining NaN with 0 to prevent propagation)
        result = result.with_columns(
            [pl.col("dx").fill_nan(0.0).ewm_mean(alpha=alpha, adjust=False).alias("adx")]
        )

        return result.select(["ts", "adx", "plus_di", "minus_di"]).tail(-period * 2)
