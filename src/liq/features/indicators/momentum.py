"""Momentum indicators.

This module provides momentum-based technical indicators including
RSI, MACD, and Stochastic oscillators.

Design Principles:
    - SRP: Each indicator handles only its own calculation
    - OCP: New momentum indicators can be added without modification
    - DRY: Common patterns abstracted in BaseIndicator
"""

from typing import Any, ClassVar

import polars as pl

from liq.features.indicators.base import BaseIndicator
from liq.features.indicators.registry import register_indicator


@register_indicator
class RSI(BaseIndicator):
    """Relative Strength Index (RSI) indicator.

    RSI measures the speed and magnitude of price movements,
    oscillating between 0 and 100.

    Parameters:
        period: Lookback period (default: 14)

    Example:
        >>> rsi = RSI(params={"period": 14})
        >>> result = rsi.compute(df)
    """

    name: ClassVar[str] = "rsi"
    default_params: ClassVar[dict[str, Any]] = {"period": 14}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute RSI values.

        Args:
            df: DataFrame with 'close' and 'ts' columns

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]

        # Calculate price changes
        delta = df.select(
            [
                pl.col("ts"),
                (pl.col("close") - pl.col("close").shift(1)).alias("delta"),
            ]
        )

        # Separate gains and losses
        delta = delta.with_columns(
            [
                pl.when(pl.col("delta") > 0)
                .then(pl.col("delta"))
                .otherwise(0)
                .alias("gain"),
                pl.when(pl.col("delta") < 0)
                .then(-pl.col("delta"))
                .otherwise(0)
                .alias("loss"),
            ]
        )

        # Calculate EMA of gains and losses (Wilder's smoothing)
        alpha = 1.0 / period

        delta = delta.with_columns(
            [
                pl.col("gain").ewm_mean(alpha=alpha, adjust=False).alias("avg_gain"),
                pl.col("loss").ewm_mean(alpha=alpha, adjust=False).alias("avg_loss"),
            ]
        )

        # Calculate RS and RSI
        result = delta.with_columns(
            [
                pl.when(pl.col("avg_loss") == 0)
                .then(100.0)
                .otherwise(100.0 - (100.0 / (1.0 + pl.col("avg_gain") / pl.col("avg_loss"))))
                .alias("value"),
            ]
        )

        # Filter warmup period and return
        return result.select(["ts", "value"]).filter(pl.col("value").is_not_nan()).tail(-period)


@register_indicator
class MACD(BaseIndicator):
    """Moving Average Convergence Divergence (MACD) indicator.

    MACD shows the relationship between two EMAs of price,
    with a signal line for crossover signals.

    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        DataFrame with columns: ts, macd, signal, histogram
    """

    name: ClassVar[str] = "macd"
    default_params: ClassVar[dict[str, Any]] = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute MACD values.

        Args:
            df: DataFrame with 'close' and 'ts' columns

        Returns:
            DataFrame with columns: ts, macd, signal, histogram
        """
        fast = self._params["fast_period"]
        slow = self._params["slow_period"]
        signal_period = self._params["signal_period"]

        # Calculate EMAs
        result = df.select(
            [
                pl.col("ts"),
                pl.col("close").ewm_mean(span=fast, adjust=False).alias("ema_fast"),
                pl.col("close").ewm_mean(span=slow, adjust=False).alias("ema_slow"),
            ]
        )

        # MACD line = fast EMA - slow EMA
        result = result.with_columns(
            [(pl.col("ema_fast") - pl.col("ema_slow")).alias("macd")]
        )

        # Signal line = EMA of MACD
        result = result.with_columns(
            [pl.col("macd").ewm_mean(span=signal_period, adjust=False).alias("signal")]
        )

        # Histogram = MACD - Signal
        result = result.with_columns(
            [(pl.col("macd") - pl.col("signal")).alias("histogram")]
        )

        # Filter warmup and return
        return (
            result.select(["ts", "macd", "signal", "histogram"])
            .filter(pl.col("macd").is_not_nan())
            .tail(-slow)
        )


@register_indicator
class Stochastic(BaseIndicator):
    """Stochastic Oscillator indicator.

    Compares closing price to the price range over a lookback period,
    producing %K and %D values.

    Parameters:
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)
        smooth_k: %K smoothing period (default: 3)

    Returns:
        DataFrame with columns: ts, stoch_k, stoch_d
    """

    name: ClassVar[str] = "stochastic"
    default_params: ClassVar[dict[str, Any]] = {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3,
    }

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Stochastic values.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'ts' columns

        Returns:
            DataFrame with columns: ts, stoch_k, stoch_d
        """
        k_period = self._params["k_period"]
        d_period = self._params["d_period"]
        smooth_k = self._params["smooth_k"]

        # Calculate highest high and lowest low
        result = df.select(
            [
                pl.col("ts"),
                pl.col("close"),
                pl.col("high").rolling_max(window_size=k_period).alias("highest_high"),
                pl.col("low").rolling_min(window_size=k_period).alias("lowest_low"),
            ]
        )

        # Raw %K = (close - lowest) / (highest - lowest) * 100
        result = result.with_columns(
            [
                (
                    (pl.col("close") - pl.col("lowest_low"))
                    / (pl.col("highest_high") - pl.col("lowest_low"))
                    * 100
                ).alias("raw_k")
            ]
        )

        # Smooth %K
        result = result.with_columns(
            [pl.col("raw_k").rolling_mean(window_size=smooth_k).alias("stoch_k")]
        )

        # %D = SMA of %K
        result = result.with_columns(
            [pl.col("stoch_k").rolling_mean(window_size=d_period).alias("stoch_d")]
        )

        # Filter warmup and return
        return (
            result.select(["ts", "stoch_k", "stoch_d"])
            .filter(pl.col("stoch_k").is_not_nan() & pl.col("stoch_d").is_not_nan())
        )


@register_indicator
class ROC(BaseIndicator):
    """Rate of Change (ROC) indicator.

    Measures the percentage change in price over a specified period.

    Parameters:
        period: Lookback period (default: 10)

    Returns:
        DataFrame with columns: ts, value
    """

    name: ClassVar[str] = "roc"
    default_params: ClassVar[dict[str, Any]] = {"period": 10}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute ROC values.

        Args:
            df: DataFrame with 'close' and 'ts' columns

        Returns:
            DataFrame with columns: ts, value
        """
        period = self._params["period"]

        result = df.select(
            [
                pl.col("ts"),
                (
                    (pl.col("close") - pl.col("close").shift(period))
                    / pl.col("close").shift(period)
                    * 100
                ).alias("value"),
            ]
        )

        return result.filter(pl.col("value").is_not_nan())


@register_indicator
class Stochastic_Midrange(BaseIndicator):
    """Stochastic Oscillator using Midrange price.

    This indicator computes the Stochastic oscillator using the midrange
    price (high+low)/2 instead of close price. This is more representative
    of true market behavior.

    Parameters:
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)
        smooth_k: %K smoothing period (default: 3)
        input_column: Column name for midrange price (default: "midrange")

    Returns:
        DataFrame with columns: ts, stoch_k, stoch_d
    """

    name: ClassVar[str] = "stochastic_midrange"
    default_params: ClassVar[dict[str, Any]] = {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3,
        "input_column": "midrange",
    }

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Stochastic values using midrange.

        Args:
            df: DataFrame with 'high', 'low', 'ts' columns

        Returns:
            DataFrame with columns: ts, stoch_k, stoch_d
        """
        k_period = self._params["k_period"]
        d_period = self._params["d_period"]
        smooth_k = self._params["smooth_k"]
        input_col = self._params["input_column"]

        # Compute midrange if not present
        if input_col not in df.columns:
            df = df.with_columns(
                [((pl.col("high") + pl.col("low")) / 2).alias(input_col)]
            )

        # Calculate highest high and lowest low
        result = df.select(
            [
                pl.col("ts"),
                pl.col(input_col),
                pl.col("high").rolling_max(window_size=k_period).alias("highest_high"),
                pl.col("low").rolling_min(window_size=k_period).alias("lowest_low"),
            ]
        )

        # Raw %K = (midrange - lowest) / (highest - lowest) * 100
        result = result.with_columns(
            [
                (
                    (pl.col(input_col) - pl.col("lowest_low"))
                    / (pl.col("highest_high") - pl.col("lowest_low"))
                    * 100
                ).alias("raw_k")
            ]
        )

        # Smooth %K
        result = result.with_columns(
            [pl.col("raw_k").rolling_mean(window_size=smooth_k).alias("stoch_k")]
        )

        # %D = SMA of %K
        result = result.with_columns(
            [pl.col("stoch_k").rolling_mean(window_size=d_period).alias("stoch_d")]
        )

        # Filter warmup and return
        return (
            result.select(["ts", "stoch_k", "stoch_d"])
            .filter(pl.col("stoch_k").is_not_nan() & pl.col("stoch_d").is_not_nan())
        )
