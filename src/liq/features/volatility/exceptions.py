"""Exception hierarchy for the canonical risk-variance estimator.

All errors that originate inside ``liq.features.volatility`` derive from
``VolFeatureError`` so callers can catch the family with a single
``except`` and still distinguish leaf causes by class.

See the canonical risk-variance research plan ``[PHASE0_CONTRACT]`` for
the contract these exceptions defend.
"""

from __future__ import annotations


class VolFeatureError(Exception):
    """Base class for every error raised by ``liq.features.volatility``.

    Catch this when a caller wants any failure in the volatility module
    to take a single fallback path; catch a leaf subclass when the
    response depends on the cause.
    """


class VolSpecError(VolFeatureError):
    """A ``VolEstimatorSpec`` (or one of its component policies) is
    inconsistent — for example, ``estimator='rv'`` was requested without
    a matching ``rv_spec``, or ``target='quadratic_variation'`` was set
    against a non-minute estimator.

    Raised at validation time, before any bar is consumed.
    """


class VolDataQualityError(VolFeatureError):
    """Input bars failed a data-quality rule and no fallback estimator
    was eligible per the active ``VolQualityPolicy`` /
    ``VolEstimatorSpec``. Examples: missing close on a non-skippable
    bar; high < low after sanitization; data-quality failure rate
    exceeded over the window (Gate 3 enforcement).
    """


class VolPITViolationError(VolFeatureError):
    """The set of input rows feeding an estimate at time ``t`` included
    a row whose ``valid_from`` is strictly greater than ``t``.

    Indicates a point-in-time integrity violation; the estimator never
    returns a value when this is detected. The detection rule is
    ``max(input.valid_from) <= t`` per research plan §3.5.
    """


class VolUnavailableError(VolFeatureError):
    """The requested ``VolEstimate`` component cannot be produced in
    the active data mode — typically requesting an intraday RV / BPV /
    jump component when minute data are not available in the degraded
    daily-OHLC-only fallback mode (research plan §3.6 mode-availability
    table).
    """


__all__ = [
    "VolDataQualityError",
    "VolFeatureError",
    "VolPITViolationError",
    "VolSpecError",
    "VolUnavailableError",
]
