"""Stationarity utilities with fractional differencing (FFD)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List

import polars as pl


def _fracdiff_weights(d: float, max_lags: int = 50, tol: float = 1e-5) -> list[float]:
    """Generate fractional differencing weights until tol or max_lags."""
    w = [1.0]
    for k in range(1, max_lags):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < tol:
            break
        w.append(w_k)
    return w


@dataclass
class StationarityTransformer:
    """Applies fractional differencing with train-only fit."""

    d: float = 0.4
    max_lags: int = 50
    tol: float = 1e-5
    fitted_: bool = False

    def fit(self, series: Iterable[float]) -> "StationarityTransformer":
        # Placeholder for ADF-based d selection; retain configured d for now.
        self.fitted_ = True
        return self

    def transform(self, series: Iterable[float]) -> list[float]:
        if not self.fitted_:
            raise RuntimeError("StationarityTransformer must be fit before transform")
        weights = _fracdiff_weights(self.d, self.max_lags, self.tol)
        values = list(series)
        out: list[float] = []
        for i in range(len(values)):
            acc = 0.0
            for k, w in enumerate(weights):
                if i - k < 0:
                    break
                acc += w * values[i - k]
            out.append(acc)
        return out

    def fit_transform(self, series: Iterable[float]) -> list[float]:
        return self.fit(series).transform(series)
