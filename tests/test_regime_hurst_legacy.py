"""Compatibility tests for the direct ``regime.hurst`` module surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest

import liq.features.regime.hurst as legacy


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [-3.0, -2.8, -2.6, 0.0, 0.2, 0.4, 2.6, 2.8, 3.0],
            "y": [-1.0, -0.9, -0.8, 0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "label": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        }
    )


def test_matrix_conversion_and_softmax_handle_empty_and_numeric_inputs() -> None:
    assert legacy._to_float_matrix(pl.DataFrame({"label": ["a", "b"]})).shape == (2, 0)
    matrix = legacy._to_float_matrix(_features())
    assert matrix.shape == (9, 2)
    assert matrix.dtype == np.float64

    empty = np.empty((0, 2), dtype=np.float64)
    assert legacy._stable_softmax(empty).shape == empty.shape
    probabilities = legacy._stable_softmax(np.array([[1.0, 2.0, 3.0]]))
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0])
    assert probabilities.argmax(axis=1).tolist() == [2]


def test_kmeans_label_fallbacks_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = np.empty((0, 2), dtype=np.float64)
    assert legacy._kmeans_labels(empty, 3, random_state=7).tolist() == []

    matrix = np.arange(6, dtype=np.float64).reshape(3, 2)
    assert legacy._kmeans_labels(matrix, 1, random_state=7).tolist() == [0, 0, 0]
    assert legacy._kmeans_labels(matrix[:2], 3, random_state=7).tolist() == [0, 1]

    with monkeypatch.context() as scoped:
        scoped.setattr(legacy, "KMeans", None)
        assert legacy._kmeans_labels(matrix, 2, random_state=7).tolist() == [0, 1, 0]

    with monkeypatch.context() as scoped:
        scoped.setattr(legacy, "StandardScaler", None)
        assert legacy._kmeans_labels(matrix, 2, random_state=7).tolist() == [0, 1, 0]


def test_kmeans_labels_use_sklearn_when_available() -> None:
    labels = legacy._kmeans_labels(
        legacy._to_float_matrix(_features()),
        3,
        random_state=11,
    )

    assert labels.shape == (9,)
    assert set(labels.tolist()) == {0, 1, 2}


def test_nearest_centroid_model_handles_empty_and_missing_centroids() -> None:
    values = np.array([[0.0], [3.0]], dtype=np.float64)
    empty = legacy._NearestCentroidModel(np.empty((0, 1)), n_regimes=2)
    assert empty.predict(values).tolist() == [0, 0]
    assert empty.predict_proba(values).shape == (2, 2)
    assert empty.predict_proba(values).sum() == 0.0

    incomplete = legacy._NearestCentroidModel(np.array([[0.0]]), n_regimes=2)
    probabilities = incomplete.predict_proba(values)
    assert probabilities.shape == (2, 2)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    complete = legacy._NearestCentroidModel(np.array([[0.0], [3.0]]), n_regimes=2)
    assert complete.predict(values).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"random_state": None, "n_regimes": 2}, "random_state"),
        ({"random_state": 7, "n_regimes": 1}, "n_regimes"),
    ],
)
def test_legacy_classifier_validates_constructor(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        legacy.SVMRegimeClassifier(**kwargs)


def test_legacy_classifier_handles_empty_feature_space() -> None:
    classifier = legacy.SVMRegimeClassifier(random_state=7, n_regimes=2).fit(
        pl.DataFrame({"label": ["a", "b"]})
    )

    assert classifier.predict(pl.DataFrame({"label": ["c"]})).to_list() == []
    assert classifier.predict_proba(pl.DataFrame({"label": ["c"]})).shape == (0, 2)


class _FixedBootstrap:
    name = "fixed"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        assert features.height == 9
        assert n_regimes == 3
        assert random_state == 13
        return pl.Series("cluster_id", [4, 4, 4, 9, 9, 9, 9, 9, 9])


def test_legacy_classifier_supports_bootstrap_and_noncontiguous_labels() -> None:
    classifier = legacy.SVMRegimeClassifier(
        random_state=13,
        n_regimes=3,
        bootstrap=_FixedBootstrap(),
    ).fit(_features())

    predictions = classifier.predict(_features())
    probabilities = classifier.predict_proba(_features())

    assert set(predictions.to_list()) <= {0, 1, 2}
    assert probabilities.columns == ["regime_0", "regime_1", "regime_2"]
    assert probabilities.sum_horizontal().to_list() == pytest.approx([1.0] * 9)


def test_legacy_classifier_remaps_excess_labels_and_validates_label_length() -> None:
    features = _features().head(6)
    classifier = legacy.SVMRegimeClassifier(random_state=17, n_regimes=2).fit(
        features,
        pl.Series("cluster_id", [0, 1, 2, 3, 4, 5]),
    )

    assert set(classifier.predict(features).to_list()) <= {0, 1}

    with pytest.raises(ValueError, match="row count"):
        legacy.SVMRegimeClassifier(random_state=17, n_regimes=2).fit(
            features,
            pl.Series("cluster_id", [0, 1]),
        )


def test_legacy_classifier_strict_mode_rejects_missing_clusters() -> None:
    with pytest.raises(ValueError, match="all clusters"):
        legacy.SVMRegimeClassifier(
            random_state=19,
            n_regimes=3,
            strict_cluster_sizes=True,
        ).fit(_features(), pl.Series("cluster_id", [0] * 6 + [1] * 3))


def test_legacy_classifier_requires_fit_state_for_prediction() -> None:
    classifier = legacy.SVMRegimeClassifier(random_state=23, n_regimes=2)

    with pytest.raises(RuntimeError, match="fit before predict"):
        classifier.predict(_features())
    with pytest.raises(RuntimeError, match="fit before predict_proba"):
        classifier.predict_proba(_features())

    classifier.fit(_features(), pl.Series("cluster_id", [0] * 4 + [1] * 5))
    classifier._feature_means = None
    with pytest.raises(RuntimeError, match="feature means"):
        classifier.predict(_features())
    with pytest.raises(RuntimeError, match="feature means"):
        classifier.predict_proba(_features())

    classifier._feature_means = np.zeros(2)
    classifier._feature_stds = None
    with pytest.raises(RuntimeError, match="feature scales"):
        classifier.predict(_features())
    with pytest.raises(RuntimeError, match="feature scales"):
        classifier.predict_proba(_features())


def test_legacy_classifier_persistence_round_trip_and_type_guard(tmp_path: Path) -> None:
    path = tmp_path / "legacy-regime.joblib"
    classifier = legacy.SVMRegimeClassifier(random_state=29, n_regimes=2).fit(
        _features(),
        pl.Series("cluster_id", [0] * 4 + [1] * 5),
    )

    classifier.save(path)
    loaded = legacy.SVMRegimeClassifier.load(path)
    assert loaded.predict(_features()).to_list() == classifier.predict(_features()).to_list()

    joblib.dump("not a classifier", path)
    with pytest.raises(TypeError, match="not SVMRegimeClassifier"):
        legacy.SVMRegimeClassifier.load(path)


def test_legacy_bootstraps_handle_empty_fallback_and_normal_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = pl.DataFrame({"x": []}, schema={"x": pl.Float64})
    features = _features().select("x", "y")

    assert legacy.SphericalKMeansBootstrap().fit(empty, n_regimes=2, random_state=7).is_empty()
    assert legacy.GMMBootstrap().fit(empty, n_regimes=2, random_state=7).is_empty()
    assert legacy.TemporalEncoderBootstrap().fit(empty, n_regimes=2, random_state=7).is_empty()
    assert legacy.SwAVPrototypeBootstrap().fit(empty, n_regimes=2, random_state=7).is_empty()

    for bootstrap in (
        legacy.KMeansBootstrap(),
        legacy.SphericalKMeansBootstrap(),
        legacy.GMMBootstrap(),
        legacy.TemporalEncoderBootstrap(),
        legacy.SwAVPrototypeBootstrap(),
    ):
        labels = bootstrap.fit(features, n_regimes=3, random_state=7)
        assert labels.len() == features.height

    one_column = features.select("x")
    assert (
        legacy.TemporalEncoderBootstrap().fit(one_column, n_regimes=3, random_state=7).len()
        == features.height
    )

    monkeypatch.setattr(legacy, "GaussianMixture", None)
    assert legacy.GMMBootstrap().fit(features, n_regimes=3, random_state=7).len() == 9


class _ListEncoder:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.encode_calls = 0

    def fit(self, features: pl.DataFrame) -> _ListEncoder:
        self.fit_calls += 1
        return self

    def encode_for_clustering(self, features: pl.DataFrame) -> list[list[float]]:
        self.encode_calls += 1
        return [[float(index), float(index % 2)] for index in range(features.height)]


def test_jepa_bootstrap_normalizes_encoder_outputs_and_optional_fit() -> None:
    encoder = _ListEncoder()
    labels = legacy.JEPAEmbeddingBootstrap(encoder).fit(_features(), n_regimes=3, random_state=31)
    assert labels.len() == 9
    assert encoder.fit_calls == 1
    assert encoder.encode_calls == 2

    encoder_without_fit = _ListEncoder()
    labels_without_fit = legacy.JEPAEmbeddingBootstrap(
        encoder_without_fit,
        fit_encoder=False,
    ).fit(_features(), n_regimes=3, random_state=31)
    assert labels_without_fit.len() == 9
    assert encoder_without_fit.fit_calls == 0
    assert encoder_without_fit.encode_calls == 1


def test_legacy_label_resolution_and_terminal_coverage() -> None:
    labels = pl.Series("cluster", ["trend", "range", "trend"])
    assert legacy._resolve_expected_labels(("fixed",), labels) == ("fixed",)
    assert legacy._resolve_expected_labels((), labels) == ("range", "trend")
    assert legacy.terminal_coverage(
        pl.Series("cluster", []), expected_labels=("range", "trend")
    ) == {"range": 0.0, "trend": 0.0}
    assert legacy.terminal_coverage(labels, expected_labels=("range", "trend", "other")) == {
        "range": pytest.approx(1 / 3),
        "trend": pytest.approx(2 / 3),
        "other": 0.0,
    }


def test_legacy_cluster_quality_handles_empty_degenerate_and_sampled_inputs() -> None:
    empty = legacy.cluster_quality(
        pl.Series("cluster", []),
        pl.DataFrame({"x": []}, schema={"x": pl.Float64}),
    )
    assert empty == legacy.ClusterQuality(None, None, None, 0.0, 0)

    one_cluster = legacy.cluster_quality(
        pl.Series("cluster", [0, 0, 0]),
        pl.DataFrame({"x": [0.0, 1.0, 2.0]}),
    )
    assert one_cluster.silhouette is None
    assert one_cluster.min_cluster_fraction == 1.0

    no_numeric_features = legacy.cluster_quality(
        pl.Series("cluster", [0, 1]),
        pl.DataFrame({"label": ["a", "b"]}),
    )
    assert no_numeric_features.silhouette is None

    features = _features().select("x", "y")
    labels = pl.Series("cluster", [0] * 3 + [1] * 3 + [2] * 3)
    sampled = legacy.cluster_quality(
        labels,
        features,
        silhouette_sample_size=8,
        random_state=37,
    )
    assert sampled.silhouette is not None
    assert sampled.davies_bouldin is not None
    assert sampled.calinski_harabasz is not None
    assert sampled.min_cluster_fraction == pytest.approx(1 / 3)

    singleton = legacy.cluster_quality(
        pl.Series("cluster", [0, 0, 1]),
        pl.DataFrame({"x": [0.0, 0.1, 1.0]}),
    )
    assert singleton.n_singletons == 1

    with pytest.raises(ValueError, match="row count"):
        legacy.cluster_quality(labels.head(2), features)


def test_legacy_cluster_quality_reports_missing_sklearn_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(legacy, "silhouette_score", None)

    with pytest.raises(legacy._SklearnUnavailable, match="silhouette_score"):
        legacy.cluster_quality(
            pl.Series("cluster", [0, 0, 1, 1]),
            pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]}),
        )


def test_legacy_cross_seed_stability_handles_boundary_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = pl.Series("cluster", [0, 0, 1, 1])
    second = pl.Series("cluster", [1, 1, 0, 0])
    third = pl.Series("cluster", [0, 1, 0, 1])

    assert legacy.cross_seed_stability([]) == 1.0
    assert legacy.cross_seed_stability([first]) == 1.0
    assert legacy.cross_seed_stability([first, pl.Series("empty", [])]) == 0.0
    assert legacy.cross_seed_stability([first, second, third]) == pytest.approx(0.0)

    monkeypatch.setattr(legacy, "adjusted_rand_score", None)
    with pytest.raises(legacy._SklearnUnavailable, match="adjusted_rand_score"):
        legacy.cross_seed_stability([first, second])
