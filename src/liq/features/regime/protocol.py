"""Protocols for regime classifiers and labelers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, Self

import polars as pl

from liq.core import RegimeId


class RegimeClassifier(Protocol):
    """Classifier that emits integer regime cluster IDs and probabilities."""

    @property
    def n_regimes(self) -> int:
        """Number of classifier regimes."""
        ...

    def fit(self, features: pl.DataFrame, y: pl.Series | None = None) -> Self:
        """Fit the classifier on feature rows and optional labels."""
        ...

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Predict integer cluster IDs in the same row order as features."""
        ...

    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predict probability columns named regime_0...regime_n."""
        ...


class RegimeLabeler(Protocol):
    """Maps classifier cluster IDs to semantic regime labels."""

    @property
    def mapping(self) -> Mapping[int, RegimeId]:
        """Cluster ID to semantic regime mapping."""
        ...

    def fit(self, classifier: RegimeClassifier, features: pl.DataFrame) -> Self:
        """Derive a deterministic cluster-to-label mapping."""
        _ = classifier, features
        raise NotImplementedError

    def label(self, cluster_ids: pl.Series) -> pl.Series:
        """Map integer cluster IDs to RegimeId string values."""
        ...


class Persistable(Protocol):
    """Persistence contract for classifier implementations."""

    def save(self, path: Path) -> None:
        """Persist model state to path."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load model state from path."""
        ...


class Ensemble(Protocol):
    """Voting ensemble surface for the slim SVM regime scope."""

    models: Sequence[RegimeClassifier]
    weights: Sequence[float] | None
    strategy: Literal["voting"]
