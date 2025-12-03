"""Regime-related features."""

from __future__ import annotations

import numpy as np


def hurst_exponent(series: list[float]) -> float:
    """Estimate Hurst exponent using rescaled range."""
    if len(series) < 20:
        return 0.5
    x = np.array(series, dtype=float)
    taus = [2, 4, 8, 16]
    rs = []
    for tau in taus:
        segments = len(x) // tau
        if segments < 1:
            continue
        reshaped = x[: segments * tau].reshape((segments, tau))
        mean_adj = reshaped - reshaped.mean(axis=1, keepdims=True)
        cumulative = mean_adj.cumsum(axis=1)
        r = cumulative.max(axis=1) - cumulative.min(axis=1)
        s = mean_adj.std(axis=1)
        valid = s > 0
        if not valid.any():
            continue
        rs.append(np.log(r[valid] / s[valid]).mean() / np.log(tau))
    if not rs:
        return 0.5
    return float(np.mean(rs))
