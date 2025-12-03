"""Microstructure proxies (Corwin-Schultz)."""

from __future__ import annotations

import math
import polars as pl


def corwin_schultz_spread(df: pl.DataFrame) -> float:
    """Estimate bid-ask spread using Corwin-Schultz estimator."""
    if df.height < 2:
        return 0.0
    log_hl = (df["high"] / df["low"]).log()
    beta = (log_hl ** 2 + log_hl.shift(1) ** 2).sum()
    gamma = (log_hl + log_hl.shift(1)) ** 2
    gamma = gamma.sum()
    beta = float(beta)
    gamma = float(gamma)
    if gamma == 0 or beta == 0:
        return 0.0
    k = (2 * math.exp(beta) - 1) / (1 + math.exp(beta))
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / (3 - 2 * math.sqrt(2))
    spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    return spread
