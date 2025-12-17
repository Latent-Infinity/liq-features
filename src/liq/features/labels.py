"""Labeling utilities (triple barrier).

This module provides label generation for ML training using the triple-barrier
method. Labels indicate whether a trade would hit profit target (+1), stop loss (-1),
or time out (0) within a maximum holding period.

Example:
    >>> from liq.features.labels import TripleBarrierConfig, triple_barrier_labels_adaptive
    >>> cfg = TripleBarrierConfig(profit_std_multiple=2.0, loss_std_multiple=2.0)
    >>> result = triple_barrier_labels_adaptive(df, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import polars as pl


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling.

    Supports two modes:
    1. Fixed thresholds: Set take_profit and stop_loss as percentage values
    2. Adaptive thresholds: Use profit_std_multiple and loss_std_multiple
       to set thresholds based on rolling volatility

    Parameters:
        take_profit: Fixed profit threshold as percentage (e.g., 0.02 = 2%)
        stop_loss: Fixed loss threshold as percentage (e.g., 0.01 = 1%)
        max_holding: Maximum bars to hold position before timeout (default: 5)
        profit_std_multiple: Multiple of volatility for profit target (default: 2.0)
        loss_std_multiple: Multiple of volatility for stop loss (default: 2.0)
        volatility_window: Window for volatility calculation (default: 20)
    """

    take_profit: float | None = None
    stop_loss: float | None = None
    max_holding: int = 5
    profit_std_multiple: float = 2.0
    loss_std_multiple: float = 2.0
    volatility_window: int = 20


def triple_barrier_labels(df: pl.DataFrame, cfg: TripleBarrierConfig) -> List[int]:
    """Generate +1/-1/0 labels using triple barrier method.

    This is the original iterative implementation for backward compatibility.

    Args:
        df: DataFrame with 'close' column
        cfg: Configuration with take_profit and stop_loss thresholds

    Returns:
        List of labels: +1 (profit), -1 (loss), 0 (timeout)

    Note:
        This function requires take_profit and stop_loss to be set in config.
        For adaptive thresholds, use triple_barrier_labels_adaptive().
    """
    closes = df["close"]
    labels: List[int] = []

    # Use fixed thresholds (backward compatible)
    tp_pct = cfg.take_profit if cfg.take_profit is not None else 0.02
    sl_pct = cfg.stop_loss if cfg.stop_loss is not None else 0.02

    for i in range(len(closes)):
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        horizon = min(len(closes), i + cfg.max_holding + 1)
        outcome = 0
        for j in range(i + 1, horizon):
            price = closes[j]
            if price >= tp:
                outcome = 1
                break
            if price <= sl:
                outcome = -1
                break
        labels.append(outcome)
    return labels


def triple_barrier_labels_adaptive(
    df: pl.DataFrame,
    cfg: TripleBarrierConfig,
) -> pl.DataFrame:
    """Generate triple barrier labels with adaptive volatility-based thresholds.

    This function extends the triple-barrier method to support adaptive thresholds
    based on rolling volatility. When fixed thresholds (take_profit, stop_loss)
    are provided, they are used directly. Otherwise, thresholds are computed as
    multiples of rolling volatility.

    Args:
        df: DataFrame with 'close' column
        cfg: Configuration specifying thresholds (fixed or volatility-based)

    Returns:
        DataFrame with additional 'label' column:
        - 1: Profit target hit first
        - -1: Stop loss hit first
        - 0: Timeout (max_holding reached without hitting barriers)

    Example:
        >>> cfg = TripleBarrierConfig(
        ...     profit_std_multiple=2.0,  # 2σ profit target
        ...     loss_std_multiple=2.0,    # 2σ stop loss
        ...     max_holding=5,
        ... )
        >>> result = triple_barrier_labels_adaptive(df, cfg)
    """
    closes = df["close"].to_list()
    n = len(closes)

    # Determine if using fixed or adaptive thresholds
    use_fixed = cfg.take_profit is not None and cfg.stop_loss is not None

    if use_fixed:
        # Use fixed thresholds (same as original function)
        labels = triple_barrier_labels(df, cfg)
    else:
        # Compute rolling volatility for adaptive thresholds
        returns = []
        for i in range(1, n):
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            returns.append(ret)
        returns = [0.0] + returns  # Pad first value

        # Compute rolling std
        window = cfg.volatility_window
        rolling_std = []
        for i in range(n):
            if i < window:
                # Not enough data for full window
                if i > 0:
                    std = _compute_std(returns[: i + 1])
                else:
                    std = 0.01  # Default volatility
            else:
                std = _compute_std(returns[i - window + 1 : i + 1])
            rolling_std.append(std if std > 0 else 0.01)

        # Generate labels with adaptive thresholds
        labels = []
        for i in range(n):
            entry = closes[i]
            vol = rolling_std[i]

            # Adaptive thresholds
            tp_pct = cfg.profit_std_multiple * vol
            sl_pct = cfg.loss_std_multiple * vol

            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)

            horizon = min(n, i + cfg.max_holding + 1)
            outcome = 0

            for j in range(i + 1, horizon):
                price = closes[j]
                if price >= tp:
                    outcome = 1
                    break
                if price <= sl:
                    outcome = -1
                    break

            labels.append(outcome)

    # Return DataFrame with label column
    return df.with_columns([pl.Series("label", labels)])


def _compute_std(values: list) -> float:
    """Compute standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5
