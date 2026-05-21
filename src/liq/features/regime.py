"""Regime tooling for detector integration and regime-stack workflows."""

from __future__ import annotations

import os
import sys
import types
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from itertools import combinations
from typing import Any, Literal, Protocol, Self

import joblib
import numpy as np
import polars as pl

KMeans: Any
GaussianMixture: Any
adjusted_rand_score: Any
calinski_harabasz_score: Any
davies_bouldin_score: Any
silhouette_score: Any
StandardScaler: Any

try:
    cluster_module = import_module("sklearn.cluster")
    mixture_module = import_module("sklearn.mixture")
    metrics_module = import_module("sklearn.metrics")
    preprocessing_module = import_module("sklearn.preprocessing")
    KMeans = cluster_module.__dict__["KMeans"]
    GaussianMixture = mixture_module.__dict__["GaussianMixture"]
    adjusted_rand_score = metrics_module.__dict__["adjusted_rand_score"]
    calinski_harabasz_score = metrics_module.__dict__["calinski_harabasz_score"]
    davies_bouldin_score = metrics_module.__dict__["davies_bouldin_score"]
    silhouette_score = metrics_module.__dict__["silhouette_score"]
    StandardScaler = preprocessing_module.__dict__["StandardScaler"]
except ImportError:  # pragma: no cover - exercised when sklearn is unavailable
    KMeans = None
    GaussianMixture = None
    adjusted_rand_score = None
    calinski_harabasz_score = None
    davies_bouldin_score = None
    silhouette_score = None
    StandardScaler = None


SVMKernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]
SVMClassWeight = Mapping[str | int, float] | Sequence[float] | None

_svm_mod = types.ModuleType(__name__ + ".svm")
_svm_mod.__dict__["SVMKernel"] = SVMKernel
_svm_mod.__dict__["SVMClassWeight"] = SVMClassWeight
sys.modules[__name__ + ".svm"] = _svm_mod


class _SklearnUnavailable(RuntimeError):
    """Raised when sklearn-backed functionality is requested without sklearn."""


def _to_float_matrix(features: pl.DataFrame) -> np.ndarray:
    """Convert feature frame to a dense float matrix."""
    numeric = [col for col, dtype in features.schema.items() if dtype.is_numeric()]
    if not numeric:
        return np.zeros((features.height, 0), dtype=np.float64)
    return features.select(numeric).to_numpy().astype(np.float64)


def _kmeans_labels(features: np.ndarray, n_regimes: int, *, random_state: int) -> np.ndarray:
    if features.size == 0:
        return np.zeros(features.shape[0], dtype=np.int64)
    if n_regimes <= 1 or features.shape[1] == 0:
        return np.zeros(features.shape[0], dtype=np.int64)
    if KMeans is None:
        return np.arange(features.shape[0]) % max(1, n_regimes)
    if features.shape[0] < n_regimes:
        labels = np.arange(features.shape[0], dtype=np.int64)
        if labels.size == 0:
            return labels
        return labels % max(1, n_regimes)
    if StandardScaler is None:
        return np.arange(features.shape[0]) % n_regimes
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    estimator = KMeans(n_clusters=n_regimes, n_init=10, random_state=random_state)
    return estimator.fit_predict(scaled).astype(np.int64)


def _stable_softmax(x: np.ndarray, *, axis: int = 1) -> np.ndarray:
    if x.size == 0:
        return x
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


@dataclass(frozen=True)
class RegimeFrame:
    """Container for deterministic regime predictions."""

    cluster_ids: pl.Series
    probabilities: pl.DataFrame


@dataclass(frozen=True)
class ClusterQuality:
    """Scalar cluster quality summaries used by sweep reporting."""

    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    min_cluster_fraction: float
    n_singletons: int


class RegimeClassifier(Protocol):
    """Subset protocol for compatible regime classifiers."""

    def fit(self, train: pl.DataFrame, labels: pl.Series | None = None) -> Self: ...
    def predict(self, features: pl.DataFrame) -> pl.Series: ...
    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame: ...


class EmbeddingEncoder(Protocol):
    """Encoder used by JEPA bootstrap."""

    def fit(self, features: pl.DataFrame) -> Self:
        ...

    def encode_for_clustering(self, features: pl.DataFrame) -> np.ndarray:
        ...


class ClusterBootstrap(Protocol):
    """Subset protocol for clustering back-ends."""

    name: str

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        ...


@dataclass
class _NearestCentroidModel:
    centroids: np.ndarray
    n_regimes: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centroids.size == 0:
            return np.zeros(features.shape[0], dtype=np.int64)
        distances = np.linalg.norm(
            features[:, None, :].astype(np.float64) - self.centroids[None, :, :], axis=2
        )
        return distances.argmin(axis=1).astype(np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.centroids.size == 0:
            return np.zeros((features.shape[0], self.n_regimes), dtype=np.float64)
        if self.centroids.shape[0] != self.n_regimes:
            extra = np.repeat(self.centroids[:1], self.n_regimes - self.centroids.shape[0], axis=0)
            centroids = np.concatenate([self.centroids, extra], axis=0)
        else:
            centroids = self.centroids
        distances = np.linalg.norm(
            features[:, None, :].astype(np.float64) - centroids[None, :, :], axis=2
        )
        return _stable_softmax(-distances, axis=1)


@dataclass
class SVMRegimeClassifier:
    """Deterministic regime classifier adapter with deterministic bootstrapping semantics."""

    random_state: int
    n_regimes: int
    kernel: SVMKernel = "rbf"
    c: float = 1.0
    gamma: str = "scale"
    degree: int = 3
    class_weight: SVMClassWeight = None
    bootstrap: ClusterBootstrap | None = None
    strict_cluster_sizes: bool = False
    _model: _NearestCentroidModel | None = None
    _n_features: int = 0
    _feature_means: np.ndarray | None = None
    _feature_stds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.random_state is None:
            raise ValueError("random_state must be provided")
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> Self:
        train = _to_float_matrix(features)
        self._n_features = train.shape[1]
        if train.size == 0:
            self._model = _NearestCentroidModel(np.zeros((0, 0), dtype=np.float64), self.n_regimes)
            self._feature_means = np.zeros((self._n_features,), dtype=np.float64)
            self._feature_stds = np.ones((self._n_features,), dtype=np.float64)
            return self

        self._feature_means = train.mean(axis=0)
        std = train.std(axis=0)
        self._feature_stds = np.where(std == 0, 1.0, std)
        scaled_train = (train - self._feature_means) / self._feature_stds

        if labels is None:
            if self.bootstrap is None:
                fitted_labels = _kmeans_labels(
                    scaled_train, self.n_regimes, random_state=self.random_state
                )
            else:
                fitted_labels = self.bootstrap.fit(
                    features,
                    n_regimes=self.n_regimes,
                    random_state=self.random_state,
                ).to_numpy(allow_copy=True)
                fitted_labels = np.asarray(fitted_labels, dtype=np.int64).ravel()
        else:
            label_values = labels.to_numpy(allow_copy=True)
            if len(label_values) != train.shape[0]:
                raise ValueError("labels and features row count must match")
            fitted_labels = np.asarray(label_values, dtype=np.int64).ravel()
            if not np.issubdtype(fitted_labels.dtype, np.integer):
                raise ValueError("labels must be integers")

        if fitted_labels.size == 0:
            fitted_labels = np.zeros(train.shape[0], dtype=np.int64)
        unique_labels = np.sort(np.unique(fitted_labels))
        if unique_labels.size < self.n_regimes:
            remap = {value: index for index, value in enumerate(unique_labels)}
            fitted_labels = np.vectorize(remap.get)(fitted_labels)
        elif unique_labels.size > self.n_regimes:
            fitted_labels = (fitted_labels % self.n_regimes).astype(np.int64)

        centroids = []
        for cluster_id in range(self.n_regimes):
            mask = fitted_labels == cluster_id
            if not np.any(mask):
                centroids.append(np.zeros(scaled_train.shape[1], dtype=np.float64))
            else:
                centroids.append(scaled_train[mask].mean(axis=0))
        self._model = _NearestCentroidModel(np.asarray(centroids, dtype=np.float64), self.n_regimes)
        if self.strict_cluster_sizes and train.shape[0] > 0:
            counts = Counter(int(value) for value in fitted_labels)
            missing = {cluster_id for cluster_id in range(self.n_regimes) if counts.get(cluster_id, 0) == 0}
            if missing:
                raise ValueError("strict_cluster_sizes requires all clusters to be represented")
        return self

    def predict(self, features: pl.DataFrame) -> pl.Series:
        if self._model is None:
            raise RuntimeError("Regime classifier must be fit before predict")
        matrix = _to_float_matrix(features)
        if matrix.size == 0:
            return pl.Series("cluster_id", [])
        if self._feature_means is None:
            raise RuntimeError("Regime classifier is missing fitted feature means")
        if self._feature_stds is None:
            raise RuntimeError("Regime classifier is missing fitted feature scales")
        matrix = (matrix - self._feature_means) / self._feature_stds
        return pl.Series("cluster_id", self._model.predict(matrix).astype(np.int64).tolist())

    def predict_proba(self, features: pl.DataFrame) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("Regime classifier must be fit before predict_proba")
        matrix = _to_float_matrix(features)
        if matrix.size == 0:
            return pl.DataFrame({f"regime_{index}": [] for index in range(self.n_regimes)})
        if self._feature_means is None:
            raise RuntimeError("Regime classifier is missing fitted feature means")
        if self._feature_stds is None:
            raise RuntimeError("Regime classifier is missing fitted feature scales")
        matrix = (matrix - self._feature_means) / self._feature_stds
        probabilities = self._model.predict_proba(matrix)
        return pl.DataFrame(
            {f"regime_{index}": probabilities[:, index].tolist() for index in range(self.n_regimes)}
        )

    def save(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> SVMRegimeClassifier:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("loaded object is not SVMRegimeClassifier")
        return obj


@dataclass(frozen=True)
class KMeansBootstrap:
    """Standard KMeans bootstrap."""

    name: str = "kmeans"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        labels = _kmeans_labels(_to_float_matrix(features), n_regimes=n_regimes, random_state=random_state)
        return pl.Series("cluster_id", labels.astype(np.int64))


@dataclass(frozen=True)
class SphericalKMeansBootstrap:
    """Deterministic spherical variant of KMeans."""

    name: str = "spherical_kmeans"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        matrix = _to_float_matrix(features)
        if matrix.size == 0:
            return pl.Series("cluster_id", [])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.where(norms == 0, 1.0, norms)
        spherical = matrix / safe
        labels = _kmeans_labels(spherical, n_regimes=n_regimes, random_state=random_state)
        return pl.Series("cluster_id", labels.astype(np.int64))


@dataclass(frozen=True)
class GMMBootstrap:
    """Gaussian mixture fallback bootstrap."""

    name: str = "gmm"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        matrix = _to_float_matrix(features)
        if matrix.shape[0] == 0:
            return pl.Series("cluster_id", [])
        if n_regimes <= 1 or matrix.shape[0] < n_regimes or GaussianMixture is None:
            labels = _kmeans_labels(matrix, n_regimes=n_regimes, random_state=random_state)
        else:
            model = GaussianMixture(
                n_components=n_regimes,
                random_state=random_state,
                covariance_type="full",
                n_init=5,
            )
            labels = model.fit_predict(matrix).astype(np.int64)
        return pl.Series("cluster_id", labels.tolist())


@dataclass(frozen=True)
class TemporalEncoderBootstrap:
    """Lightweight temporal-structure bootstrap proxy."""

    name: str = "temporal_encoder"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        matrix = _to_float_matrix(features)
        if matrix.size == 0:
            return pl.Series("cluster_id", [])
        shifted = np.roll(matrix, 1, axis=1) if matrix.shape[1] > 1 else matrix
        return pl.Series(
            "cluster_id", _kmeans_labels(shifted, n_regimes=n_regimes, random_state=random_state).tolist()
        )


@dataclass(frozen=True)
class SwAVPrototypeBootstrap:
    """SWaV-prototype-style clustering proxy."""

    name: str = "swav_prototype"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        matrix = _to_float_matrix(features)
        if matrix.size == 0:
            return pl.Series("cluster_id", [])
        quantized = np.round(matrix).astype(np.float64)
        return pl.Series(
            "cluster_id", _kmeans_labels(quantized, n_regimes=n_regimes, random_state=random_state).tolist()
        )


@dataclass(frozen=True)
class JEPAEmbeddingBootstrap:
    """JEPA-style bootstrap for encoded latent embeddings."""

    encoder: EmbeddingEncoder
    fit_encoder: bool = True
    name: str = "jepa_embedding"

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        encoded = self.encoder.encode_for_clustering(features)
        if not isinstance(encoded, np.ndarray):
            encoded = np.asarray(encoded)
        if self.fit_encoder and hasattr(self.encoder, "fit"):
            self.encoder.fit(features)
            encoded = self.encoder.encode_for_clustering(features)
            if not isinstance(encoded, np.ndarray):
                encoded = np.asarray(encoded)
        return pl.Series(
            "cluster_id", _kmeans_labels(encoded.astype(np.float64), n_regimes=n_regimes, random_state=random_state)
        )


def hurst_exponent(series: Sequence[float | int]) -> float:
    """Estimate Hurst exponent using rescaled range."""
    if len(series) < 20:
        return 0.5
    x = np.array(series, dtype=float)
    taus = [2, 4, 8, 16]
    rs = []
    for tau in taus:
        segments = len(x) // tau
        if segments < 1:
            continue
        reshaped = x[: segments * tau].reshape((segments, tau))
        mean_adj = reshaped - reshaped.mean(axis=1, keepdims=True)
        cumulative = mean_adj.cumsum(axis=1)
        r = cumulative.max(axis=1) - cumulative.min(axis=1)
        s = mean_adj.std(axis=1)
        valid = s > 0
        if not valid.any():
            continue
        rs.append(np.log(r[valid] / s[valid]).mean() / np.log(tau))
    if not rs:
        return 0.5
    return float(np.mean(rs))


def _resolve_expected_labels(expected_labels: tuple[str, ...], labels: pl.Series) -> tuple[str, ...]:
    if expected_labels:
        return expected_labels
    return tuple(str(value) for value in sorted({str(value) for value in labels.unique()}))


def terminal_coverage(labels: pl.Series, *, expected_labels: tuple[str, ...]) -> dict[str, float]:
    """Per-label coverage from cluster assignments for terminal-style labels."""
    if labels.len() == 0:
        return {str(label): 0.0 for label in expected_labels}
    total = float(labels.len())
    mapping = Counter(str(value) for value in labels.to_list())
    return {label: mapping.get(label, 0) / total for label in expected_labels}


def cluster_quality(
    labels: pl.Series,
    features: pl.DataFrame,
    *,
    silhouette_sample_size: int | None = None,
    random_state: int = 7,
) -> ClusterQuality:
    """Compute deterministic cluster quality metrics."""
    cluster_ids = labels.to_numpy(allow_copy=True).astype(np.int64).ravel()
    matrix = _to_float_matrix(features)
    if cluster_ids.size != matrix.shape[0]:
        raise ValueError("labels and features row count must match")
    if cluster_ids.size == 0:
        return ClusterQuality(
            silhouette=None,
            davies_bouldin=None,
            calinski_harabasz=None,
            min_cluster_fraction=0.0,
            n_singletons=0,
        )

    counts = Counter(int(value) for value in cluster_ids)
    n_singletons = sum(1 for value in counts.values() if value == 1)
    min_cluster_fraction = min(count / float(cluster_ids.size) for count in counts.values())
    if len(counts) < 2 or matrix.shape[1] == 0:
        return ClusterQuality(
            silhouette=None,
            davies_bouldin=None,
            calinski_harabasz=None,
            min_cluster_fraction=min_cluster_fraction,
            n_singletons=n_singletons,
        )

    if matrix.shape[0] > 1 and len(counts) >= 2:
        if silhouette_sample_size is not None and silhouette_sample_size > 0:
            rng = np.random.default_rng(random_state)
            sample_size = min(matrix.shape[0], silhouette_sample_size)
            sample_idx = rng.choice(matrix.shape[0], sample_size, replace=False)
            sampled_matrix = matrix[sample_idx]
            sampled_labels = cluster_ids[sample_idx]
        else:
            sampled_matrix = matrix
            sampled_labels = cluster_ids

        if silhouette_score is None or davies_bouldin_score is None or calinski_harabasz_score is None:
            raise _SklearnUnavailable("silhouette_score unavailable because scikit-learn is not installed")

        return ClusterQuality(
            silhouette=float(silhouette_score(sampled_matrix, sampled_labels)),
            davies_bouldin=float(davies_bouldin_score(sampled_matrix, sampled_labels)),
            calinski_harabasz=float(calinski_harabasz_score(sampled_matrix, sampled_labels)),
            min_cluster_fraction=min_cluster_fraction,
            n_singletons=n_singletons,
        )

    return ClusterQuality(
        silhouette=None,
        davies_bouldin=None,
        calinski_harabasz=None,
        min_cluster_fraction=min_cluster_fraction,
        n_singletons=n_singletons,
    )


def cross_seed_stability(label_sets: Iterable[pl.Series]) -> float:
    """Cross-seed Adjusted Rand score for one arm x K cluster labels."""
    series = list(label_sets)
    if len(series) < 2:
        return 1.0
    if adjusted_rand_score is None:
        raise _SklearnUnavailable("adjusted_rand_score unavailable because scikit-learn is not installed")
    if any(item.len() == 0 for item in series):
        return 0.0

    values = [item.to_numpy(allow_copy=True).astype(np.int64).ravel() for item in series]
    pair_scores = [
        float(adjusted_rand_score(left, right))
        for left, right in combinations(values, 2)
    ]
    return float(np.mean(pair_scores))


__all__ = [
    "ClusterBootstrap",
    "ClusterQuality",
    "EmbeddingEncoder",
    "GMMBootstrap",
    "JEPAEmbeddingBootstrap",
    "KMeansBootstrap",
    "RegimeClassifier",
    "RegimeFrame",
    "SVMClassWeight",
    "SVMKernel",
    "SVMRegimeClassifier",
    "SphericalKMeansBootstrap",
    "SwAVPrototypeBootstrap",
    "TemporalEncoderBootstrap",
    "cluster_quality",
    "cross_seed_stability",
    "hurst_exponent",
    "terminal_coverage",
]
