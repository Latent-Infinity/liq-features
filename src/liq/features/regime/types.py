"""Typed result objects for regime classification."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from liq.core import RegimeId

_PROBABILITY_SUM_TOLERANCE = 1e-6


def _validate_probability(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} probabilities must be numeric")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} probabilities must be finite values in [0, 1]")
    return probability


def _validate_probability_sum(name: str, probabilities: tuple[float, ...]) -> None:
    if not math.isclose(sum(probabilities), 1.0, abs_tol=_PROBABILITY_SUM_TOLERANCE):
        raise ValueError(f"{name} probabilities must sum to 1.0")


def _validate_confidence(confidence: float | None) -> None:
    if confidence is None:
        return
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be numeric")
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")


@dataclass(frozen=True)
class RegimeFrame:
    """Vectorized regime classifier output aligned to a feature frame."""

    cluster_ids: pl.Series
    probabilities: pl.DataFrame

    def __post_init__(self) -> None:
        if self.cluster_ids.len() != self.probabilities.height:
            raise ValueError("cluster_ids and probabilities row count must match")
        n_regimes = self.probabilities.width
        if n_regimes < 2:
            raise ValueError("probability columns must contain at least two regimes")
        expected_columns = [f"regime_{index}" for index in range(n_regimes)]
        if self.probabilities.columns != expected_columns:
            raise ValueError("probability columns must be regime_0...regime_{n-1}")

        max_cluster = n_regimes - 1
        for cluster_id in self.cluster_ids:
            if not isinstance(cluster_id, int) or cluster_id < 0 or cluster_id > max_cluster:
                raise ValueError("cluster_id must reference a probability column")

        for row in self.probabilities.iter_rows():
            probabilities = tuple(_validate_probability("RegimeFrame", value) for value in row)
            _validate_probability_sum("RegimeFrame", probabilities)


@dataclass(frozen=True)
class RegimePrediction:
    """Raw classifier output for one row before semantic labeling."""

    cluster_id: int
    probabilities: tuple[float, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("probabilities must be non-empty")
        if self.cluster_id < 0 or self.cluster_id >= len(self.probabilities):
            raise ValueError("cluster_id must reference a probability index")

        probabilities = tuple(
            _validate_probability("RegimePrediction", probability)
            for probability in self.probabilities
        )
        _validate_probability_sum("RegimePrediction", probabilities)
        _validate_confidence(self.confidence)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True)
class RegimeOutput:
    """Semantic regime output after cluster labeling."""

    label: RegimeId
    cluster_id: int
    probabilities: Mapping[RegimeId, float] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        label = RegimeId(self.label)
        object.__setattr__(self, "label", label)

        if self.cluster_id < 0:
            raise ValueError("cluster_id must be non-negative")
        if self.probabilities is not None:
            if not self.probabilities:
                raise ValueError("probabilities must be non-empty when provided")
            probabilities = {
                RegimeId(regime): _validate_probability("RegimeOutput", probability)
                for regime, probability in self.probabilities.items()
            }
            if label not in probabilities:
                raise ValueError("chosen label must be present in probabilities")
            _validate_probability_sum("RegimeOutput", tuple(probabilities.values()))
            object.__setattr__(self, "probabilities", probabilities)
        _validate_confidence(self.confidence)
