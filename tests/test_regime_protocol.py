"""Contract tests for the regime classifier protocol surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, cast

import polars as pl
import pytest

from liq.features.regime import (
    Ensemble,
    Persistable,
    RegimeClassifier,
    RegimeId,
    RegimeLabeler,
    RegimeOutput,
    RegimePrediction,
    hurst_exponent,
)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open": [float(i) for i in range(100)],
            "high": [float(i) + 1.0 for i in range(100)],
            "low": [float(i) - 1.0 for i in range(100)],
            "close": [float(i) + 0.5 for i in range(100)],
            "volume": [1_000.0 + float(i) for i in range(100)],
        }
    )


@dataclass
class StubClassifier(RegimeClassifier, Persistable):
    _n_regimes: int = 3

    @property
    def n_regimes(self) -> int:
        return self._n_regimes

    def fit(self, features: pl.DataFrame, y: pl.Series | None = None) -> Self:
        del features, y
        return self

    def predict(self, features: pl.DataFrame) -> pl.Series:
        return pl.Series("cluster_id", [index % self.n_regimes for index in range(features.height)])

    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame:
        probabilities = [
            [1.0 if regime == index % self.n_regimes else 0.0 for regime in range(self.n_regimes)]
            for index in range(features.height)
        ]
        return pl.DataFrame(
            {
                f"regime_{regime}": [row[regime] for row in probabilities]
                for regime in range(self.n_regimes)
            }
        )

    def save(self, path: Path) -> None:
        path.write_text(str(self.n_regimes), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(_n_regimes=int(path.read_text(encoding="utf-8")))


@dataclass
class StubLabeler(RegimeLabeler):
    _mapping: dict[int, RegimeId] = field(default_factory=dict)

    @property
    def mapping(self) -> Mapping[int, RegimeId]:
        return self._mapping

    def fit(self, classifier: RegimeClassifier, features: pl.DataFrame) -> Self:
        del features
        labels = (RegimeId.range, RegimeId.neutral, RegimeId.trend)
        self._mapping = {index: labels[index] for index in range(classifier.n_regimes)}
        return self

    def label(self, cluster_ids: pl.Series) -> pl.Series:
        if not self._mapping:
            raise RuntimeError("labeler must be fitted before labeling")
        return pl.Series(
            "regime", [self._mapping[int(cluster_id)].value for cluster_id in cluster_ids]
        )


@dataclass
class StubEnsemble(Ensemble):
    models: Sequence[RegimeClassifier]
    weights: Sequence[float] | None = None
    strategy: Literal["voting"] = "voting"


def test_regime_subpackage_exports_hurst_and_protocol_surface() -> None:
    assert callable(hurst_exponent)
    assert RegimeClassifier is not None
    assert RegimeLabeler is not None
    assert Persistable is not None
    assert Ensemble is not None
    assert RegimePrediction is not None
    assert RegimeOutput is not None
    assert RegimeId.trend == "trend"


def test_stub_classifier_and_labeler_execute_end_to_end() -> None:
    features = _features()
    classifier = StubClassifier().fit(features)
    labeler = StubLabeler().fit(classifier, features)

    clusters = classifier.predict(features)
    probabilities = classifier.predict_proba(features)
    labels = labeler.label(clusters)

    assert len(clusters) == features.height
    assert probabilities.shape == (features.height, classifier.n_regimes)
    assert labels.head(3).to_list() == ["range", "neutral", "trend"]
    output = RegimeOutput(label=RegimeId.range, cluster_id=0, confidence=0.9)
    assert output.cluster_id == 0
    assert output.confidence == 0.9


def test_importing_liq_features_does_not_require_regime_implementations() -> None:
    import liq.features

    assert liq.features.compute_derived_fields is not None


def test_persistable_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "classifier.txt"
    StubClassifier(_n_regimes=3).save(path)
    loaded = StubClassifier.load(path)
    assert loaded.n_regimes == 3


def test_ensemble_contract_captures_voting_strategy() -> None:
    classifier = StubClassifier()
    ensemble = StubEnsemble(models=(classifier,), weights=None)
    assert ensemble.models == (classifier,)
    assert ensemble.weights is None
    assert ensemble.strategy == "voting"


def test_regime_prediction_accepts_normalized_probabilities() -> None:
    prediction = RegimePrediction(cluster_id=1, probabilities=(0.2, 0.7, 0.1), confidence=0.7)
    assert prediction.cluster_id == 1


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: RegimePrediction(cluster_id=0, probabilities=()), "non-empty"),
        (lambda: RegimePrediction(cluster_id=-1, probabilities=(1.0,)), "cluster_id"),
        (lambda: RegimePrediction(cluster_id=1, probabilities=(1.0,)), "cluster_id"),
        (lambda: RegimePrediction(cluster_id=0, probabilities=(-0.1, 1.1)), "probabilities"),
        (
            lambda: RegimePrediction(cluster_id=0, probabilities=(float("nan"), 1.0)),
            "probabilities",
        ),
        (lambda: RegimePrediction(cluster_id=0, probabilities=(0.2, 0.2)), "sum"),
        (
            lambda: RegimePrediction(cluster_id=0, probabilities=(1.0,), confidence=1.1),
            "confidence",
        ),
    ],
)
def test_regime_prediction_rejects_invalid_values(
    factory: Callable[[], RegimePrediction], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_regime_output_accepts_label_keyed_probabilities() -> None:
    output = RegimeOutput(
        label=RegimeId.trend,
        cluster_id=2,
        probabilities={RegimeId.range: 0.2, RegimeId.neutral: 0.3, RegimeId.trend: 0.5},
        confidence=0.5,
    )
    assert output.label is RegimeId.trend


def test_regime_output_coerces_valid_label_strings_and_probability_keys() -> None:
    output = RegimeOutput(
        label=cast(Any, "trend"),
        cluster_id=2,
        probabilities=cast(Any, {"range": 0.2, "neutral": 0.3, "trend": 0.5}),
        confidence=0.5,
    )

    assert output.label is RegimeId.trend
    assert output.probabilities == {
        RegimeId.range: 0.2,
        RegimeId.neutral: 0.3,
        RegimeId.trend: 0.5,
    }


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: RegimeOutput(label=cast(Any, "invalid"), cluster_id=0), "invalid"),
        (lambda: RegimeOutput(label=RegimeId.trend, cluster_id=-1), "cluster_id"),
        (lambda: RegimeOutput(label=RegimeId.trend, cluster_id=0, probabilities={}), "non-empty"),
        (
            lambda: RegimeOutput(
                label=RegimeId.trend, cluster_id=0, probabilities={RegimeId.range: 1.0}
            ),
            "chosen label",
        ),
        (
            lambda: RegimeOutput(
                label=RegimeId.trend,
                cluster_id=0,
                probabilities={RegimeId.trend: -0.1, RegimeId.range: 1.1},
            ),
            "probabilities",
        ),
        (
            lambda: RegimeOutput(
                label=RegimeId.trend, cluster_id=0, probabilities={RegimeId.trend: float("nan")}
            ),
            "probabilities",
        ),
        (
            lambda: RegimeOutput(
                label=RegimeId.trend, cluster_id=0, probabilities={RegimeId.trend: 0.5}
            ),
            "sum",
        ),
        (lambda: RegimeOutput(label=RegimeId.trend, cluster_id=0, confidence=-0.1), "confidence"),
    ],
)
def test_regime_output_rejects_invalid_values(
    factory: Callable[[], RegimeOutput], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_classifier_output_to_labeler_round_trip_preserves_cluster_and_confidence() -> None:
    prediction = RegimePrediction(cluster_id=2, probabilities=(0.1, 0.2, 0.7), confidence=0.7)
    labeler = StubLabeler().fit(StubClassifier(), _features())
    label = RegimeId(labeler.label(pl.Series([prediction.cluster_id])).item())

    output = RegimeOutput(
        label=label, cluster_id=prediction.cluster_id, confidence=prediction.confidence
    )
    assert output.label is RegimeId.trend
    assert output.cluster_id == prediction.cluster_id
    assert output.confidence == prediction.confidence
