"""Tests for pluggable regime bootstrap implementations."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from liq.features.regime import (
    ClusterBootstrap,
    GMMBootstrap,
    JEPAEmbeddingBootstrap,
    KMeansBootstrap,
    SphericalKMeansBootstrap,
    SVMRegimeClassifier,
    SwAVPrototypeBootstrap,
    TemporalEncoderBootstrap,
)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "momentum": [-2.0, -1.9, -2.1, -1.8, 0.0, 0.1, -0.1, 0.2, 2.0, 2.1, 1.9, 2.2],
            "volatility": [0.2, 0.3, 0.2, 0.4, 1.0, 1.1, 0.9, 1.0, 0.3, 0.2, 0.4, 0.3],
        }
    )


def test_cluster_bootstrap_protocol_surface_and_kmeans_determinism() -> None:
    bootstrap: ClusterBootstrap = KMeansBootstrap()
    features = _features()

    first = bootstrap.fit(features, n_regimes=3, random_state=7)
    second = bootstrap.fit(features, n_regimes=3, random_state=7)

    assert first.name == "regime"
    assert first.len() == features.height
    assert first.to_list() == second.to_list()
    assert set(first.to_list()) <= {0, 1, 2}


@pytest.mark.parametrize(
    "bootstrap", [KMeansBootstrap(), SphericalKMeansBootstrap(), GMMBootstrap()]
)
def test_bootstraps_reject_invalid_regime_count(bootstrap: ClusterBootstrap) -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        bootstrap.fit(_features(), n_regimes=1, random_state=7)


def test_bootstraps_reject_empty_feature_inputs() -> None:
    with pytest.raises(ValueError, match="column"):
        KMeansBootstrap().fit(pl.DataFrame(), n_regimes=2, random_state=7)

    with pytest.raises(ValueError, match="row"):
        KMeansBootstrap().fit(pl.DataFrame({"x": []}), n_regimes=2, random_state=7)


def test_gmm_bootstrap_rejects_more_regimes_than_rows() -> None:
    with pytest.raises(ValueError, match="row count"):
        GMMBootstrap().fit(pl.DataFrame({"x": [1.0, 2.0]}), n_regimes=3, random_state=7)


def test_swav_bootstrap_is_svm_default_unlabeled_behavior() -> None:
    features = _features()

    default_classifier = SVMRegimeClassifier(random_state=11).fit(features)
    explicit_classifier = SVMRegimeClassifier(
        random_state=11,
        bootstrap=SwAVPrototypeBootstrap(),
    ).fit(features)

    assert (
        explicit_classifier.predict(features).to_list()
        == default_classifier.predict(features).to_list()
    )
    assert (
        explicit_classifier.predict_proba(features).rows()
        == default_classifier.predict_proba(features).rows()
    )


def test_spherical_kmeans_matches_kmeans_on_normalized_features() -> None:
    features = _features()
    labels = SphericalKMeansBootstrap().fit(features, n_regimes=3, random_state=17)

    normalized = normalize(features.to_numpy().astype(np.float64), norm="l2")
    expected = KMeans(n_clusters=3, random_state=17, n_init="auto").fit_predict(normalized)

    assert labels.to_list() == expected.tolist()


def test_spherical_kmeans_rejects_all_zero_rows() -> None:
    with pytest.raises(ValueError, match="zero"):
        SphericalKMeansBootstrap().fit(
            pl.DataFrame({"x": [1.0, 0.0, -1.0], "y": [0.0, 0.0, 0.0]}),
            n_regimes=2,
            random_state=19,
        )


def test_gmm_bootstrap_is_deterministic_and_exposes_membership_probabilities() -> None:
    features = _features()
    bootstrap = GMMBootstrap()

    first = bootstrap.fit(features, n_regimes=3, random_state=23)
    second = bootstrap.fit(features, n_regimes=3, random_state=23)
    probabilities = bootstrap.membership_probabilities(features, n_regimes=3, random_state=23)

    assert first.to_list() == second.to_list()
    assert probabilities.columns == ["regime_0", "regime_1", "regime_2"]
    assert probabilities.shape == (features.height, 3)
    for row in probabilities.iter_rows():
        assert sum(row) == pytest.approx(1.0)


class RecordingEncoder:
    def __init__(self) -> None:
        self.fit_heights: list[int] = []

    def fit(self, features: pl.DataFrame) -> None:
        self.fit_heights.append(features.height)

    def encode_for_clustering(self, features: pl.DataFrame) -> np.ndarray:
        values = features.to_numpy().astype(np.float64)
        return np.column_stack((values[:, 0] + values[:, 1], values[:, 0] - values[:, 1]))


def test_jepa_embedding_bootstrap_uses_encoder_and_inner_bootstrap_without_leaking_test_rows() -> (
    None
):
    features = _features()
    encoder = RecordingEncoder()
    bootstrap = JEPAEmbeddingBootstrap(encoder=encoder, fit_encoder=True)

    labels = bootstrap.fit(features, n_regimes=3, random_state=29)

    assert encoder.fit_heights == [features.height]
    assert labels.len() == features.height
    assert set(labels.to_list()) <= {0, 1, 2}


def test_jepa_embedding_bootstrap_rejects_wrong_embedding_row_count() -> None:
    class BadEncoder:
        def encode_for_clustering(self, features: pl.DataFrame) -> np.ndarray:
            return np.zeros((features.height - 1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="row count"):
        JEPAEmbeddingBootstrap(encoder=BadEncoder()).fit(_features(), n_regimes=3, random_state=31)


def test_temporal_encoder_bootstrap_is_deterministic_and_fits_temporal_projection() -> None:
    features = _features()
    bootstrap = TemporalEncoderBootstrap(context_bars=4, embedding_dim=3, n_negative_samples=2)

    first = bootstrap.fit(features, n_regimes=3, random_state=37)
    second = bootstrap.fit(features, n_regimes=3, random_state=37)
    embeddings = bootstrap.encode_for_clustering(features)

    assert first.to_list() == second.to_list()
    assert first.len() == features.height
    assert embeddings.shape == (features.height, 3)
    assert bootstrap.training_loss_ is not None
    assert bootstrap.positive_similarity_ > bootstrap.negative_similarity_


def test_temporal_encoder_bootstrap_rejects_invalid_temporal_config() -> None:
    with pytest.raises(ValueError, match="context_bars"):
        TemporalEncoderBootstrap(context_bars=0)
    with pytest.raises(ValueError, match="embedding_dim"):
        TemporalEncoderBootstrap(embedding_dim=0)
    with pytest.raises(ValueError, match="negative"):
        TemporalEncoderBootstrap(n_negative_samples=0)


def test_swav_prototype_bootstrap_learns_online_prototypes_and_balances_assignments() -> None:
    features = _features()
    bootstrap = SwAVPrototypeBootstrap(n_epochs=3, temperature=0.2, sinkhorn_iterations=3)

    first = bootstrap.fit(features, n_regimes=3, random_state=41)
    second = bootstrap.fit(features, n_regimes=3, random_state=41)
    counts = first.value_counts().get_column("count").to_list()

    assert first.to_list() == second.to_list()
    assert set(first.to_list()) == {0, 1, 2}
    assert max(counts) - min(counts) <= 1
    assert bootstrap.prototypes_ is not None
    assert bootstrap.assignment_loss_ is not None
    assert bootstrap.assignment_loss_ >= 0.0


def test_swav_prototype_bootstrap_rejects_invalid_online_learning_config() -> None:
    with pytest.raises(ValueError, match="epochs"):
        SwAVPrototypeBootstrap(n_epochs=0)
    with pytest.raises(ValueError, match="temperature"):
        SwAVPrototypeBootstrap(temperature=0.0)
    with pytest.raises(ValueError, match="sinkhorn"):
        SwAVPrototypeBootstrap(sinkhorn_iterations=0)
