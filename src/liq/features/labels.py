"""Labeling utilities (triple barrier)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple

import polars as pl


@dataclass
class TripleBarrierConfig:
    take_profit: float  # e.g., 0.01 = 1%
    stop_loss: float    # e.g., 0.01 = 1%
    max_holding: int    # bars


def triple_barrier_labels(df: pl.DataFrame, cfg: TripleBarrierConfig) -> List[int]:
    """Generate +1/-1/0 labels using triple barrier."""
    closes = df["close"]
    labels: List[int] = []
    for i in range(len(closes)):
        entry = closes[i]
        tp = entry * (1 + cfg.take_profit)
        sl = entry * (1 - cfg.stop_loss)
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
