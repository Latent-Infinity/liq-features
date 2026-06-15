"""Pluggable bootstrap strategies for regime cluster labels."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

FloatArray = NDArray[np.float64]


def _to_feature_array(features: pl.DataFrame) -> FloatArray:
    if features.width == 0:
        raise ValueError("features must contain at least one column")
    if features.is_empty():
        raise ValueError("features must contain at least one row")
    return features.to_numpy().astype(np.float64, copy=False)


def _validate_n_regimes(n_regimes: int) -> None:
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2")


def _label_series(labels: NDArray[np.integer]) -> pl.Series:
    return pl.Series("regime", labels.astype(np.int_, copy=False).tolist())


@runtime_checkable
class ClusterBootstrap(Protocol):
    """Strategy that derives integer cluster labels from feature rows."""

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        """Return one integer label per feature row."""
        ...  # pragma: no cover


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Encoder that can produce fixed-length embeddings for clustering."""

    def encode_for_clustering(self, features: pl.DataFrame) -> FloatArray:
        """Return one embedding row per feature row."""
        ...  # pragma: no cover


class KMeansBootstrap:
    """Standard-scaled KMeans bootstrap matching the original SVM behavior."""

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        _validate_n_regimes(n_regimes)
        values = _to_feature_array(features)
        scaled_values = StandardScaler().fit_transform(values)
        labels = KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto").fit_predict(
            scaled_values
        )
        return _label_series(labels)


class SphericalKMeansBootstrap:
    """KMeans over L2-normalized feature directions."""

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        _validate_n_regimes(n_regimes)
        values = _to_feature_array(features)
        row_norms = np.linalg.norm(values, axis=1)
        if np.any(row_norms == 0.0):
            raise ValueError("spherical k-means cannot normalize all-zero feature rows")
        normalized = normalize(values, norm="l2")
        labels = KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto").fit_predict(
            normalized
        )
        return _label_series(labels)


class GMMBootstrap:
    """Gaussian-mixture bootstrap with hard cluster assignments."""

    def __init__(self, *, reg_covar: float = 1e-6) -> None:
        self.reg_covar = reg_covar

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        _validate_n_regimes(n_regimes)
        labels = self._fit_model(features, n_regimes=n_regimes, random_state=random_state).predict(
            _to_feature_array(features)
        )
        return _label_series(labels)

    def membership_probabilities(
        self,
        features: pl.DataFrame,
        *,
        n_regimes: int,
        random_state: int,
    ) -> pl.DataFrame:
        """Return posterior component probabilities for each feature row."""
        _validate_n_regimes(n_regimes)
        values = _to_feature_array(features)
        probabilities = self._fit_model(
            features,
            n_regimes=n_regimes,
            random_state=random_state,
        ).predict_proba(values)
        return pl.DataFrame(
            {f"regime_{index}": probabilities[:, index].tolist() for index in range(n_regimes)}
        )

    def _fit_model(
        self,
        features: pl.DataFrame,
        *,
        n_regimes: int,
        random_state: int,
    ) -> GaussianMixture:
        values = _to_feature_array(features)
        if values.shape[0] < n_regimes:
            raise ValueError("n_regimes cannot exceed feature row count")
        model = GaussianMixture(
            n_components=n_regimes,
            random_state=random_state,
            reg_covar=self.reg_covar,
        )
        model.fit(values)
        return model


class JEPAEmbeddingBootstrap:
    """Bootstrap labels by clustering L2-normalized encoder embeddings.

    The encoder is intentionally duck-typed to avoid adding a hard `liq-models`
    runtime dependency to `liq-features`. Objects such as `ForexJEPA` can be
    passed when that package is installed.
    """

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        inner_bootstrap: ClusterBootstrap | None = None,
        fit_encoder: bool = False,
    ) -> None:
        self.encoder = encoder
        self.inner_bootstrap = inner_bootstrap or SphericalKMeansBootstrap()
        self.fit_encoder = fit_encoder

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        _validate_n_regimes(n_regimes)
        fit = getattr(self.encoder, "fit", None)
        if self.fit_encoder and callable(fit):
            fit(features)
        embeddings: FloatArray = np.asarray(
            self.encoder.encode_for_clustering(features),
            dtype=np.float64,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != features.height:
            raise ValueError("encoder embeddings and features row count must match")
        return self.inner_bootstrap.fit(
            _embedding_frame(embeddings),
            n_regimes=n_regimes,
            random_state=random_state,
        )


class TemporalEncoderBootstrap:
    """TS2Vec/TNC-style temporal contrastive encoder bootstrap.

    This is a lightweight in-tree encoder: each row is represented by a
    left-padded temporal context window, a deterministic projection is fitted
    from positive adjacent-window differences against sampled negative-window
    differences, and the resulting normalized embeddings are clustered. It keeps
    the online-fit-per-fold contract without adding a heavy dependency.
    """

    def __init__(
        self,
        *,
        context_bars: int = 8,
        embedding_dim: int = 8,
        n_negative_samples: int = 8,
    ) -> None:
        if context_bars < 1:
            raise ValueError("context_bars must be at least 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if n_negative_samples < 1:
            raise ValueError("n_negative_samples must be at least 1")
        self.context_bars = context_bars
        self.embedding_dim = embedding_dim
        self.n_negative_samples = n_negative_samples
        self._projection: FloatArray | None = None
        self.training_loss_: float | None = None
        self.positive_similarity_: float = float("nan")
        self.negative_similarity_: float = float("nan")

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        _validate_n_regimes(n_regimes)
        contexts = _temporal_context_matrix(_to_feature_array(features), self.context_bars)
        self._fit_projection(contexts, random_state=random_state)
        return SphericalKMeansBootstrap().fit(
            _embedding_frame(self._encode_contexts(contexts)),
            n_regimes=n_regimes,
            random_state=random_state,
        )

    def encode_for_clustering(self, features: pl.DataFrame) -> FloatArray:
        if self._projection is None:
            raise RuntimeError("temporal encoder must be fit before encoding")
        contexts = _temporal_context_matrix(_to_feature_array(features), self.context_bars)
        return self._encode_contexts(contexts)

    def _fit_projection(self, contexts: FloatArray, *, random_state: int) -> None:
        if contexts.shape[0] < 2:
            raise ValueError("temporal encoder requires at least two rows")
        centered = StandardScaler().fit_transform(contexts)
        positives = centered[1:] - centered[:-1]
        rng = np.random.default_rng(random_state)
        negative_indices = rng.integers(
            0, centered.shape[0], size=(positives.shape[0], self.n_negative_samples)
        )
        anchors = centered[:-1]
        negatives = centered[negative_indices] - np.expand_dims(anchors, axis=1)
        negative_mean = negatives.reshape(-1, centered.shape[1]).mean(axis=0)
        contrast_direction = positives.mean(axis=0) - negative_mean
        if np.allclose(contrast_direction, 0.0):
            contrast_direction = centered.std(axis=0)
        components = [contrast_direction]
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        for component in vh:
            if len(components) >= self.embedding_dim:
                break
            components.append(component)
        while len(components) < self.embedding_dim:
            basis = np.zeros(centered.shape[1], dtype=np.float64)
            basis[(len(components) - 1) % centered.shape[1]] = 1.0
            components.append(basis)
        projection = np.vstack(components[: self.embedding_dim]).T
        projection = normalize(projection, norm="l2", axis=0)
        self._projection = np.asarray(projection, dtype=np.float64)
        embeddings = np.asarray(normalize(centered @ self._projection, norm="l2"), dtype=np.float64)
        positive_pairs = np.sum(embeddings[1:] * embeddings[:-1], axis=1)
        negative_embeddings = embeddings[negative_indices]
        negative_pairs = np.sum(
            np.expand_dims(embeddings[:-1], axis=1) * negative_embeddings,
            axis=2,
        ).reshape(-1)
        self.positive_similarity_ = float(np.mean(positive_pairs))
        self.negative_similarity_ = float(np.mean(negative_pairs))
        self.training_loss_ = float(
            max(0.0, 1.0 - self.positive_similarity_ + self.negative_similarity_)
        )

    def _encode_contexts(self, contexts: FloatArray) -> FloatArray:
        if self._projection is None:
            raise RuntimeError("temporal encoder must be fit before encoding")
        centered = StandardScaler().fit_transform(contexts)
        return np.asarray(normalize(centered @ self._projection, norm="l2"), dtype=np.float64)


def _temporal_context_matrix(values: FloatArray, context_bars: int) -> FloatArray:
    first = values[:1]
    padded = np.vstack((np.repeat(first, context_bars - 1, axis=0), values))
    contexts = [
        padded[index : index + context_bars].reshape(-1) for index in range(values.shape[0])
    ]
    return np.asarray(contexts, dtype=np.float64)


class SwAVPrototypeBootstrap:
    """SwAV-style online prototype learner with balanced assignments."""

    def __init__(
        self,
        *,
        n_epochs: int = 5,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
    ) -> None:
        if n_epochs < 1:
            raise ValueError("epochs must be at least 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be at least 1")
        self.n_epochs = n_epochs
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.prototypes_: FloatArray | None = None
        self.assignment_loss_: float | None = None

    def fit(self, features: pl.DataFrame, *, n_regimes: int, random_state: int) -> pl.Series:
        _validate_n_regimes(n_regimes)
        values = _to_feature_array(features)
        if values.shape[0] < n_regimes:
            raise ValueError("n_regimes cannot exceed feature row count")
        embeddings: FloatArray = np.asarray(normalize(values, norm="l2"), dtype=np.float64)
        prototypes = self._initial_prototypes(
            embeddings, n_regimes=n_regimes, random_state=random_state
        )
        assignments = np.zeros((embeddings.shape[0], n_regimes), dtype=np.float64)
        for _ in range(self.n_epochs):
            logits = embeddings @ prototypes.T / self.temperature
            assignments = _sinkhorn_assignments(logits, self.sinkhorn_iterations)
            prototypes = assignments.T @ embeddings
            prototype_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
            prototypes = np.divide(
                prototypes,
                prototype_norms,
                out=np.zeros_like(prototypes),
                where=prototype_norms != 0.0,
            )
        self.prototypes_ = np.asarray(prototypes, dtype=np.float64)
        hard_scores = embeddings @ self.prototypes_.T
        distances = 1.0 - hard_scores
        labels = _balanced_assignment(np.asarray(distances, dtype=np.float64))
        row_scores = hard_scores[np.arange(embeddings.shape[0]), labels]
        self.assignment_loss_ = float(np.mean(1.0 - row_scores))
        return _label_series(labels)

    def _initial_prototypes(
        self,
        embeddings: FloatArray,
        *,
        n_regimes: int,
        random_state: int,
    ) -> FloatArray:
        centers = (
            KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto")
            .fit(embeddings)
            .cluster_centers_
        )
        return np.asarray(normalize(centers, norm="l2"), dtype=np.float64)


def _sinkhorn_assignments(logits: FloatArray, iterations: int) -> FloatArray:
    scores = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores = scores + 1e-12
    assignment = scores / scores.sum()
    n_samples, n_prototypes = assignment.shape
    row_target = 1.0 / n_samples
    column_target = 1.0 / n_prototypes
    for _ in range(iterations):
        assignment = assignment * (row_target / assignment.sum(axis=1, keepdims=True))
        assignment = assignment * (column_target / assignment.sum(axis=0, keepdims=True))
    return assignment / assignment.sum(axis=1, keepdims=True)


def _embedding_frame(embeddings: FloatArray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            f"embedding_{index}": embeddings[:, index].tolist()
            for index in range(embeddings.shape[1])
        }
    )


def _balanced_assignment(distances: FloatArray) -> NDArray[np.int_]:
    n_samples, n_regimes = distances.shape
    base_quota = n_samples // n_regimes
    remainder = n_samples % n_regimes
    quotas = [base_quota + (1 if regime < remainder else 0) for regime in range(n_regimes)]
    labels = np.full(n_samples, -1, dtype=np.int_)
    for sample_index in np.argsort(distances.min(axis=1)):
        ranked_regimes = np.argsort(distances[sample_index])
        for regime in ranked_regimes:
            if quotas[int(regime)] > 0:
                labels[sample_index] = int(regime)
                quotas[int(regime)] -= 1
                break
    return labels
