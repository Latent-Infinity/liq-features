"""Tests for the slim SVM regime classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib
import polars as pl
import pytest

from liq.features.regime import SVMRegimeClassifier, SVMVotingEnsemble, SwAVPrototypeBootstrap


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "momentum": [
                -2.0,
                -1.9,
                -2.1,
                -1.8,
                0.0,
                0.1,
                -0.1,
                0.2,
                2.0,
                2.1,
                1.9,
                2.2,
            ],
            "volatility": [
                0.2,
                0.3,
                0.2,
                0.4,
                1.0,
                1.1,
                0.9,
                1.0,
                0.3,
                0.2,
                0.4,
                0.3,
            ],
        }
    )


def _labels() -> pl.Series:
    return pl.Series("regime", [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


def test_svm_classifier_requires_explicit_random_state() -> None:
    with pytest.raises(TypeError):
        SVMRegimeClassifier()  # type: ignore[call-arg]


def test_svm_classifier_fits_polars_inputs_and_emits_polars_outputs() -> None:
    features = _features()
    classifier = SVMRegimeClassifier(random_state=7).fit(features, _labels())

    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    assert classifier.n_regimes == 3
    assert predictions.name == "cluster_id"
    assert predictions.len() == features.height
    assert set(predictions.to_list()) <= {0, 1, 2}
    assert probabilities.columns == ["regime_0", "regime_1", "regime_2"]
    assert probabilities.shape == (features.height, 3)
    for row in probabilities.iter_rows():
        assert sum(row) == pytest.approx(1.0)


def test_svm_classifier_can_fit_without_labels_using_default_swav_bootstrap() -> None:
    features = _features()
    classifier = SVMRegimeClassifier(random_state=11).fit(features)

    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    assert isinstance(classifier.bootstrap, SwAVPrototypeBootstrap)
    assert predictions.len() == features.height
    assert probabilities.shape == (features.height, classifier.n_regimes)


class DegenerateBootstrap:
    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        if n_regimes < 2 or random_state < 0:
            raise ValueError("invalid test bootstrap parameters")
        return pl.Series("regime", [0] * (features.height - 1) + [1])


def test_svm_classifier_accepts_custom_bootstrap_for_unlabeled_fit() -> None:
    features = _features()
    classifier = SVMRegimeClassifier(
        random_state=11,
        n_regimes=2,
        bootstrap=DegenerateBootstrap(),
    ).fit(features)

    assert classifier.predict(features).len() == features.height


def test_svm_classifier_strict_cluster_size_rejects_degenerate_bootstrap() -> None:
    features = pl.DataFrame({"x": [float(value) for value in range(200)]})

    with pytest.raises(ValueError, match="cluster size"):
        SVMRegimeClassifier(
            random_state=11,
            n_regimes=2,
            bootstrap=DegenerateBootstrap(),
            strict_cluster_sizes=True,
        ).fit(features)


def test_svm_classifier_predictions_are_deterministic_for_same_seed() -> None:
    features = _features()
    first = SVMRegimeClassifier(random_state=17).fit(features, _labels())
    second = SVMRegimeClassifier(random_state=17).fit(features, _labels())

    assert first.predict(features).to_list() == second.predict(features).to_list()
    assert first.predict_proba(features).rows() == second.predict_proba(features).rows()


def test_svm_classifier_persists_and_loads_predictions(tmp_path: Path) -> None:
    features = _features()
    path = tmp_path / "svm-regime.pkl"
    classifier = SVMRegimeClassifier(random_state=23).fit(features, _labels())

    classifier.save(path)
    loaded = SVMRegimeClassifier.load(path)

    assert loaded.n_regimes == classifier.n_regimes
    assert loaded.predict(features).to_list() == classifier.predict(features).to_list()
    assert loaded.predict_proba(features).rows() == classifier.predict_proba(features).rows()


def test_svm_classifier_rejects_prediction_before_fit() -> None:
    classifier = SVMRegimeClassifier(random_state=29)

    with pytest.raises(RuntimeError, match="fit"):
        classifier.predict(_features())

    with pytest.raises(RuntimeError, match="fit"):
        classifier.predict_proba(_features())


def test_svm_voting_ensemble_fits_models_and_averages_probabilities() -> None:
    features = _features()
    ensemble = SVMVotingEnsemble(
        models=(
            SVMRegimeClassifier(random_state=31, kernel="linear"),
            SVMRegimeClassifier(random_state=31, kernel="rbf"),
        ),
    ).fit(features, _labels())

    predictions = ensemble.predict(features)
    probabilities = ensemble.predict_proba(features)

    assert ensemble.n_regimes == 3
    assert ensemble.strategy == "voting"
    assert predictions.name == "cluster_id"
    assert predictions.len() == features.height
    assert probabilities.columns == ["regime_0", "regime_1", "regime_2"]
    for row in probabilities.iter_rows():
        assert sum(row) == pytest.approx(1.0)


def test_svm_voting_ensemble_builds_fast_default_svm_pair_from_seed() -> None:
    features = _features()
    ensemble = SVMVotingEnsemble(random_state=37).fit(features, _labels())

    kernels = [cast(SVMRegimeClassifier, model).kernel for model in ensemble.models]
    assert kernels == ["rbf", "linear"]
    assert ensemble.predict(features).len() == features.height


def test_svm_voting_ensemble_uses_one_bootstrap_for_unlabeled_fit() -> None:
    features = _features()
    first_model = SVMRegimeClassifier(random_state=7, kernel="linear")
    second_model = SVMRegimeClassifier(random_state=23, kernel="rbf")
    independently_bootstrapped_second = SVMRegimeClassifier(random_state=23, kernel="rbf").fit(
        features
    )

    ensemble = SVMVotingEnsemble(models=(first_model, second_model)).fit(features)

    assert (
        independently_bootstrapped_second.predict(features).to_list()
        != first_model.predict(features).to_list()
    )
    assert second_model.predict(features).to_list() == first_model.predict(features).to_list()
    assert ensemble.predict(features).to_list() == first_model.predict(features).to_list()


def test_svm_classifier_validates_inputs_and_label_range() -> None:
    classifier = SVMRegimeClassifier(random_state=41)

    with pytest.raises(ValueError, match="row"):
        classifier.fit(pl.DataFrame({"feature": []}))

    with pytest.raises(ValueError, match="column"):
        classifier.fit(pl.DataFrame())

    with pytest.raises(ValueError, match="length"):
        classifier.fit(_features(), pl.Series("regime", [0, 1]))

    with pytest.raises(ValueError, match="regime range"):
        classifier.fit(_features(), pl.Series("regime", [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3]))


@pytest.mark.parametrize("n_regimes", [0, 1])
def test_svm_classifier_rejects_invalid_regime_count(n_regimes: int) -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        SVMRegimeClassifier(random_state=43, n_regimes=n_regimes)


def test_svm_classifier_rejects_saving_before_fit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        SVMRegimeClassifier(random_state=43).save(tmp_path / "unfit-svm.npz")


def test_svm_classifier_load_rejects_invalid_state_file(tmp_path: Path) -> None:
    path = tmp_path / "not-svm.npz"
    path.write_text("not a classifier state", encoding="utf-8")

    with pytest.raises(ValueError):
        SVMRegimeClassifier.load(path)


@pytest.mark.parametrize(
    "state, message",
    [
        (["not", "a", "state"], "state format"),
        ({"format": "liq.features.regime.svm.v1", "params": {}}, "missing required fields"),
    ],
)
def test_svm_classifier_load_rejects_malformed_state_shapes(
    tmp_path: Path,
    state: object,
    message: str,
) -> None:
    path = tmp_path / "malformed-svm.pkl"
    joblib.dump(state, path)

    with pytest.raises(ValueError, match=message):
        SVMRegimeClassifier.load(path)


def test_svm_voting_ensemble_uses_weights_when_provided() -> None:
    features = _features()
    ensemble = SVMVotingEnsemble(
        models=(
            SVMRegimeClassifier(random_state=47, kernel="linear"),
            SVMRegimeClassifier(random_state=47, kernel="rbf"),
        ),
        weights=(0.75, 0.25),
    ).fit(features, _labels())

    assert ensemble.weights == (0.75, 0.25)
    assert ensemble.predict_proba(features).shape == (features.height, ensemble.n_regimes)


def test_svm_voting_ensemble_rejects_prediction_before_fit() -> None:
    ensemble = SVMVotingEnsemble(random_state=53)

    with pytest.raises(RuntimeError, match="fit"):
        ensemble.predict(_features())

    with pytest.raises(RuntimeError, match="fit"):
        ensemble.predict_proba(_features())


@pytest.mark.parametrize("n_regimes", [0, 1])
def test_svm_voting_ensemble_rejects_invalid_regime_count(n_regimes: int) -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        SVMVotingEnsemble(random_state=59, n_regimes=n_regimes)


def test_svm_voting_ensemble_requires_seed_for_default_models() -> None:
    with pytest.raises(ValueError, match="random_state"):
        SVMVotingEnsemble()


def test_svm_voting_ensemble_rejects_empty_models() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SVMVotingEnsemble(models=())


def test_svm_voting_ensemble_validates_weights() -> None:
    models = (
        SVMRegimeClassifier(random_state=61),
        SVMRegimeClassifier(random_state=61, kernel="linear"),
    )

    with pytest.raises(ValueError, match="weights"):
        SVMVotingEnsemble(models=models, weights=(1.0,))

    with pytest.raises(ValueError, match="positive"):
        SVMVotingEnsemble(models=models, weights=(0.0, 0.0))


def test_svm_voting_ensemble_rejects_mismatched_model_regime_counts() -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        SVMVotingEnsemble(
            models=(
                SVMRegimeClassifier(random_state=67, n_regimes=3),
                SVMRegimeClassifier(random_state=67, n_regimes=4),
            ),
        )
