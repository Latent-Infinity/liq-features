"""Entropy-based features."""

from __future__ import annotations

import numpy as np


def sample_entropy(series: list[float], m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy for a series."""
    x = np.array(series, dtype=float)
    n = len(x)
    if n <= m + 1:
        return 0.0
    def _phi(m):
        count = 0
        for i in range(n - m):
            for j in range(i + 1, n - m):
                if np.all(np.abs(x[i : i + m] - x[j : j + m]) <= r * np.std(x)):
                    count += 1
        return count
    a = _phi(m)
    b = _phi(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return -np.log(b / a)
