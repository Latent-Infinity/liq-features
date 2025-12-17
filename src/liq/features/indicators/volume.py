"""Volume-based indicators.

This module provides volume analysis indicators including abnormal turnover
detection and normalized volume ratios.

Design Principles:
    - SRP: Each indicator handles only its own calculation
    - OCP: New volume indicators can be added without modification
    - DRY: Common patterns abstracted in BaseIndicator
"""

from typing import Any, ClassVar

import polars as pl

from liq.features.indicators.base import BaseIndicator
from liq.features.indicators.registry import register_indicator


@register_indicator
class AbnormalTurnover(BaseIndicator):
    """Abnormal Turnover Rate indicator.

    Measures deviation of current volume from historical average using z-score.
    Formula: (volume - rolling_mean(volume, window)) / rolling_std(volume, window)

    This indicator is useful for detecting unusual trading activity that may
    signal significant market events or regime changes.

    Parameters:
        window: Lookback period for rolling statistics (default: 55)
    """

    name: ClassVar[str] = "abnormal_turnover"
    default_params: ClassVar[dict[str, Any]] = {"window": 55}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute abnormal turnover z-score.

        Args:
            df: DataFrame with 'ts' and 'volume' columns

        Returns:
            DataFrame with columns: ts, value (z-score)

        Raises:
            ValueError: If 'volume' column is missing
        """
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column")

        window = self._params["window"]

        result = df.select([
            pl.col("ts"),
            pl.col("volume"),
        ]).with_columns([
            pl.col("volume").rolling_mean(window_size=window).alias("rolling_mean"),
            pl.col("volume").rolling_std(window_size=window).alias("rolling_std"),
        ]).with_columns([
            (
                (pl.col("volume") - pl.col("rolling_mean")) / pl.col("rolling_std")
            ).alias("value"),
        ]).select(["ts", "value"])

        return result.filter(pl.col("value").is_not_nan())


@register_indicator
class NormalizedVolume(BaseIndicator):
    """Normalized Trading Volume indicator.

    Normalizes volume by dividing by N-day moving average.
    Formula: volume / SMA(volume, window)

    Values > 1.0 indicate above-average volume, < 1.0 indicates below-average.

    Parameters:
        window: N-day moving average period (default: 55)
    """

    name: ClassVar[str] = "normalized_volume"
    default_params: ClassVar[dict[str, Any]] = {"window": 55}

    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute normalized volume ratio.

        Args:
            df: DataFrame with 'ts' and 'volume' columns

        Returns:
            DataFrame with columns: ts, value (ratio)

        Raises:
            ValueError: If 'volume' column is missing
        """
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column")

        window = self._params["window"]

        result = df.select([
            pl.col("ts"),
            pl.col("volume"),
        ]).with_columns([
            pl.col("volume").rolling_mean(window_size=window).alias("rolling_mean"),
        ]).with_columns([
            (pl.col("volume") / pl.col("rolling_mean")).alias("value"),
        ]).select(["ts", "value"])

        return result.filter(pl.col("value").is_not_nan())
