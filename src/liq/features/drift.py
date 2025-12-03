"""Drift detection utilities comparing live vs. train distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DriftResult:
    feature: str
    statistic: float
    threshold: float
    drifted: bool


def ks_drift(live: Iterable[float], ref: Iterable[float], feature: str, threshold: float = 0.1) -> DriftResult:
    """Simplified KS-like drift detector using Wasserstein distance proxy."""
    live_arr = np.array(list(live), dtype=float)
    ref_arr = np.array(list(ref), dtype=float)
    if live_arr.size == 0 or ref_arr.size == 0:
        return DriftResult(feature, 0.0, threshold, False)
    stat = abs(live_arr.mean() - ref_arr.mean()) + abs(live_arr.std() - ref_arr.std())
    return DriftResult(feature, float(stat), threshold, stat > threshold)
