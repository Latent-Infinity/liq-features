"""Canonical risk-variance estimator for ``liq-features``.

The package ships the contract types (``VolEstimatorSpec`` and its
policies, ``VolEstimate`` and its components), the exception hierarchy,
and the public entry point ``estimate_variance``. The formula registry
behind ``estimate_variance`` is implemented for the daily OHLC
estimators and wired through data-quality, fallback, and structured
logging hooks.

The legacy ``yang_zhang`` / ``garman_klass`` functions remain importable
from this package (``from liq.features.volatility import yang_zhang``)
during the transition; they will be retired once downstream consumers
migrate to ``estimate_variance``.

See the research plan ``[PHASE0_CONTRACT]`` for the canonical-scalar
decision and ``[CANONICALIZATION]`` for the single-path commitment.
"""

from __future__ import annotations

from ._legacy import garman_klass, yang_zhang
from .contracts import (
    RVSpec,
    TimingPolicy,
    VolCalendarPolicy,
    VolComponent,
    VolEstimate,
    VolEstimatorSpec,
    VolQualityPolicy,
)
from .estimate import estimate_variance
from .exceptions import (
    VolDataQualityError,
    VolFeatureError,
    VolPITViolationError,
    VolSpecError,
    VolUnavailableError,
)

__all__ = [
    "RVSpec",
    "TimingPolicy",
    "VolCalendarPolicy",
    "VolComponent",
    "VolDataQualityError",
    "VolEstimate",
    "VolEstimatorSpec",
    "VolFeatureError",
    "VolPITViolationError",
    "VolQualityPolicy",
    "VolSpecError",
    "VolUnavailableError",
    "estimate_variance",
    "garman_klass",
    "yang_zhang",
]
