"""Entropy-based features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def sample_entropy(series: Sequence[float | int], m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy for a series."""
    x = np.array(series, dtype=float)
    n = len(x)
    if n <= m + 1:
        return 0.0

    def _phi(window: int) -> int:
        count = 0
        for i in range(n - window):
            for j in range(i + 1, n - window):
                if np.all(np.abs(x[i : i + window] - x[j : j + window]) <= r * np.std(x)):
                    count += 1
        return count
    a = _phi(m)
    b = _phi(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return -np.log(b / a)
