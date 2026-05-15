"""Regime classification contracts and utilities."""

from liq.core import RegimeId
from liq.features.regime.bootstrap import (
    ClusterBootstrap,
    EmbeddingEncoder,
    GMMBootstrap,
    JEPAEmbeddingBootstrap,
    KMeansBootstrap,
    SphericalKMeansBootstrap,
    SwAVPrototypeBootstrap,
    TemporalEncoderBootstrap,
)
from liq.features.regime.hurst import hurst_exponent
from liq.features.regime.protocol import Ensemble, Persistable, RegimeClassifier, RegimeLabeler
from liq.features.regime.quality import (
    ClusterQuality,
    cluster_quality,
    cross_seed_stability,
    terminal_coverage,
)
from liq.features.regime.svm import SVMRegimeClassifier, SVMVotingEnsemble
from liq.features.regime.types import RegimeFrame, RegimeOutput, RegimePrediction

__all__ = [
    "ClusterBootstrap",
    "ClusterQuality",
    "EmbeddingEncoder",
    "Ensemble",
    "GMMBootstrap",
    "JEPAEmbeddingBootstrap",
    "KMeansBootstrap",
    "Persistable",
    "RegimeClassifier",
    "RegimeId",
    "RegimeLabeler",
    "RegimeFrame",
    "RegimeOutput",
    "RegimePrediction",
    "SphericalKMeansBootstrap",
    "SwAVPrototypeBootstrap",
    "SVMRegimeClassifier",
    "SVMVotingEnsemble",
    "cluster_quality",
    "cross_seed_stability",
    "hurst_exponent",
    "terminal_coverage",
    "TemporalEncoderBootstrap",
]
