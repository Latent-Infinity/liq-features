"""Support Vector Machine regime classifiers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, cast

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from liq.features.regime.bootstrap import ClusterBootstrap, SwAVPrototypeBootstrap
from liq.features.regime.protocol import Ensemble, Persistable, RegimeClassifier

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

SVMKernel = Literal["linear", "poly", "rbf", "sigmoid"]
SVMClassWeight = Literal["balanced"] | dict[int, float] | None

_PERSISTENCE_FORMAT = "liq.features.regime.svm.v1"
logger = logging.getLogger(__name__)


def _to_feature_array(features: pl.DataFrame) -> FloatArray:
    if features.width == 0:
        raise ValueError("features must contain at least one column")
    if features.is_empty():
        raise ValueError("features must contain at least one row")
    return features.to_numpy().astype(np.float64, copy=False)


def _to_label_array(labels: pl.Series, expected_length: int, n_regimes: int) -> IntArray:
    if labels.len() != expected_length:
        raise ValueError("labels length must match feature row count")
    label_values = labels.to_numpy().astype(np.int_, copy=False)
    if np.any((label_values < 0) | (label_values >= n_regimes)):
        raise ValueError("labels must be in the configured regime range")
    return label_values


def _probability_columns(n_regimes: int) -> list[str]:
    return [f"regime_{regime}" for regime in range(n_regimes)]


def _normalize_probabilities(probabilities: FloatArray) -> FloatArray:
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities,
        row_sums,
        out=np.zeros_like(probabilities),
        where=row_sums != 0.0,
    )


def _bootstrap_label_series(
    features: pl.DataFrame,
    *,
    n_regimes: int,
    random_state: int,
    bootstrap: ClusterBootstrap | None = None,
) -> pl.Series:
    return (bootstrap or SwAVPrototypeBootstrap()).fit(
        features,
        n_regimes=n_regimes,
        random_state=random_state,
    )


def _min_cluster_fraction(labels: pl.Series) -> float:
    counts = [int(count) for count in labels.value_counts().get_column("count").to_list()]
    return min(counts) / labels.len()


class SVMRegimeClassifier(RegimeClassifier, Persistable):
    """Polars-native regime classifier backed by sklearn's ``SVC``."""

    def __init__(
        self,
        *,
        random_state: int,
        n_regimes: int = 3,
        kernel: SVMKernel = "rbf",
        c: float = 1.0,
        gamma: str = "scale",
        degree: int = 3,
        class_weight: SVMClassWeight = None,
        bootstrap: ClusterBootstrap | None = None,
        strict_cluster_sizes: bool = False,
    ) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")
        self.random_state = random_state
        self._n_regimes = n_regimes
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight
        self.bootstrap = bootstrap or SwAVPrototypeBootstrap()
        self.strict_cluster_sizes = strict_cluster_sizes
        self._scaler: StandardScaler | None = None
        self._model: SVC | None = None

    @property
    def n_regimes(self) -> int:
        """Number of classifier regimes."""
        return self._n_regimes

    def fit(self, features: pl.DataFrame, y: pl.Series | None = None) -> Self:
        """Fit the classifier from Polars feature rows and optional labels."""
        values = _to_feature_array(features)
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
        label_series = (
            y
            if y is not None
            else _bootstrap_label_series(
                features,
                n_regimes=self.n_regimes,
                random_state=self.random_state,
                bootstrap=self.bootstrap,
            )
        )
        if y is None:
            min_fraction = _min_cluster_fraction(label_series)
            if min_fraction < 0.01:
                message = f"cluster size below 1% threshold: {min_fraction:.4f}"
                if self.strict_cluster_sizes:
                    raise ValueError(message)
                logger.warning(message)
        labels = _to_label_array(label_series, features.height, self.n_regimes)

        model = SVC(
            kernel=self.kernel,
            C=self.c,
            gamma=self.gamma,
            degree=self.degree,
            probability=True,
            class_weight=self.class_weight,
            random_state=self.random_state,
            decision_function_shape="ovr",
        )
        model.fit(scaled_values, labels)
        self._scaler = scaler
        self._model = model
        return self

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Predict integer cluster IDs in feature row order."""
        model, scaled_values = self._model_and_scaled_values(features)
        predictions = model.predict(scaled_values).astype(np.int_, copy=False)
        return pl.Series("cluster_id", predictions.tolist())

    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predict columns named ``regime_0`` through ``regime_n``."""
        model, scaled_values = self._model_and_scaled_values(features)
        raw_probabilities = model.predict_proba(scaled_values).astype(np.float64, copy=False)
        probabilities = np.zeros((features.height, self.n_regimes), dtype=np.float64)
        for source_index, class_label in enumerate(model.classes_):
            probabilities[:, int(class_label)] = raw_probabilities[:, source_index]
        probabilities = _normalize_probabilities(probabilities)
        return pl.DataFrame(
            {
                column: probabilities[:, index].tolist()
                for index, column in enumerate(_probability_columns(self.n_regimes))
            }
        )

    def save(self, path: Path) -> None:
        """Persist the fitted classifier via joblib.

        Security: joblib uses pickle internally — only load files from trusted
        sources. The persisted file is compatible with the same major version
        of scikit-learn it was saved under; cross-version loads are not guaranteed.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("classifier must be fit before saving")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": _PERSISTENCE_FORMAT,
                "params": {
                    "random_state": self.random_state,
                    "n_regimes": self.n_regimes,
                    "kernel": self.kernel,
                    "c": self.c,
                    "gamma": self.gamma,
                    "degree": self.degree,
                    "class_weight": self.class_weight,
                    "strict_cluster_sizes": self.strict_cluster_sizes,
                },
                "scaler": self._scaler,
                "model": self._model,
            },
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a persisted SVM classifier saved by :meth:`save`.

        Security: joblib uses pickle internally — only load files from trusted
        sources. Raises ``ValueError`` if the file is missing the expected
        format tag, has the wrong shape, or cannot be deserialized.
        """
        try:
            state: Any = joblib.load(path)
        except Exception as error:
            raise ValueError("file is not a valid SVMRegimeClassifier state") from error

        if not isinstance(state, dict) or state.get("format") != _PERSISTENCE_FORMAT:
            raise ValueError("unsupported SVMRegimeClassifier state format")
        params = state.get("params")
        scaler = state.get("scaler")
        model = state.get("model")
        if not isinstance(params, dict) or scaler is None or model is None:
            raise ValueError("SVMRegimeClassifier state is missing required fields")

        loaded = cls(**params)
        loaded._scaler = scaler
        loaded._model = model
        return loaded

    def _model_and_scaled_values(self, features: pl.DataFrame) -> tuple[SVC, FloatArray]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("classifier must be fit before prediction")
        values = _to_feature_array(features)
        return self._model, np.asarray(self._scaler.transform(values), dtype=np.float64)


class SVMVotingEnsemble(RegimeClassifier, Ensemble):
    """Voting ensemble for the slim SVM regime scope.

    ``random_state`` is used only when constructing the default RBF + linear
    model pair. When explicit ``models`` are supplied, unlabeled bootstrap
    alignment follows the first model's ``random_state`` so every supplied model
    receives the same derived labels.
    """

    strategy: Literal["voting"] = "voting"

    def __init__(
        self,
        *,
        random_state: int | None = None,
        models: Sequence[SVMRegimeClassifier] | None = None,
        weights: Sequence[float] | None = None,
        n_regimes: int = 3,
    ) -> None:
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")
        if models is None:
            if random_state is None:
                raise ValueError("random_state is required when models are not provided")
            models = (
                SVMRegimeClassifier(random_state=random_state, n_regimes=n_regimes, kernel="rbf"),
                SVMRegimeClassifier(
                    random_state=random_state, n_regimes=n_regimes, kernel="linear"
                ),
            )
        if not models:
            raise ValueError("models must be non-empty")
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("weights length must match models length")
            if sum(weights) <= 0.0:
                raise ValueError("weights must have a positive sum")
        model_regime_counts = {model.n_regimes for model in models}
        if model_regime_counts != {n_regimes}:
            raise ValueError("all models must use the ensemble n_regimes")

        self.models = tuple(models)
        self.weights = tuple(weights) if weights is not None else None
        self._n_regimes = n_regimes
        self._is_fitted = False

    @property
    def n_regimes(self) -> int:
        """Number of classifier regimes."""
        return self._n_regimes

    def fit(self, features: pl.DataFrame, y: pl.Series | None = None) -> Self:
        """Fit every model in the ensemble with aligned regime IDs."""
        labels = (
            y
            if y is not None
            else _bootstrap_label_series(
                features,
                n_regimes=self.n_regimes,
                random_state=cast(SVMRegimeClassifier, self.models[0]).random_state,
                bootstrap=cast(SVMRegimeClassifier, self.models[0]).bootstrap,
            )
        )
        for model in self.models:
            model.fit(features, labels)
        self._is_fitted = True
        return self

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Predict cluster IDs by highest averaged regime probability."""
        probabilities = self._predict_probability_array(features)
        predictions = np.argmax(probabilities, axis=1).astype(np.int_, copy=False)
        return pl.Series("cluster_id", predictions.tolist())

    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame:
        """Average model probabilities into regime columns."""
        probabilities = self._predict_probability_array(features)
        return pl.DataFrame(
            {
                column: probabilities[:, index].tolist()
                for index, column in enumerate(_probability_columns(self.n_regimes))
            }
        )

    def _predict_probability_array(self, features: pl.DataFrame) -> FloatArray:
        if not self._is_fitted:
            raise RuntimeError("ensemble must be fit before prediction")
        model_probabilities = [model.predict_proba(features).to_numpy() for model in self.models]
        stacked = np.stack(model_probabilities).astype(np.float64, copy=False)
        if self.weights is None:
            weights = np.ones(len(self.models), dtype=np.float64)
        else:
            weights = np.array(self.weights, dtype=np.float64)
        averaged = np.average(stacked, axis=0, weights=weights)
        return _normalize_probabilities(averaged.astype(np.float64, copy=False))
