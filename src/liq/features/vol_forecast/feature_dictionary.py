"""Versioned feature dictionary for forecast features.

The dictionary is a data artifact: one row per feature field with
field-level metadata that can be consumed by downstream validation,
feature manifests, and audit tooling.
"""

from __future__ import annotations

import json
from hashlib import sha256

from .contracts import (
    FEATURE_DICTIONARY_VERSION,
    FeatureDictionary,
    feature_dictionary_id,
)

forecast_feature_dictionary: FeatureDictionary = {
    "daily_var": {
        "unit": "variance",
        "transform": "identity",
        "annualization_status": "interval",
        "additive": False,
        "availability_rule": "EOD/Pre-open only from completed sessions",
        "null_policy": "nullable",
    },
    "weekly_avg_var": {
        "unit": "variance",
        "transform": "mean_rolling",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "post EOD close; requires at least 5 sessions",
        "null_policy": "nullable",
    },
    "monthly_avg_var": {
        "unit": "variance",
        "transform": "mean_rolling",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "post EOD close; requires at least 22 sessions",
        "null_policy": "nullable",
    },
    "var_slope": {
        "unit": "slope",
        "transform": "least_squares",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "post EOD close; rolling window coverage",
        "null_policy": "nullable",
    },
    "log_daily_var": {
        "unit": "log-variance",
        "transform": "log",
        "annualization_status": "interval",
        "additive": False,
        "availability_rule": "aligned with daily_var",
        "null_policy": "nullable",
    },
    "log_var_slope": {
        "unit": "dimensionless",
        "transform": "difference",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "aligned with var_slope",
        "null_policy": "nullable",
    },
    "downside_rv": {
        "unit": "variance",
        "transform": "semivariance",
        "annualization_status": "interval",
        "additive": False,
        "availability_rule": "session-aligned",
        "null_policy": "nullable",
    },
    "upside_rv": {
        "unit": "variance",
        "transform": "semivariance",
        "annualization_status": "interval",
        "additive": False,
        "availability_rule": "session-aligned",
        "null_policy": "nullable",
    },
    "log_semivar_skew_long": {
        "unit": "dimensionless",
        "transform": "log_ratio",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "session-aligned",
        "null_policy": "nullable",
    },
    "log_semivar_skew_short": {
        "unit": "dimensionless",
        "transform": "log_ratio",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "session-aligned",
        "null_policy": "nullable",
    },
    "estimator_uncertainty": {
        "unit": "dimensionless",
        "transform": "identity",
        "annualization_status": "session-equivalent",
        "additive": False,
        "availability_rule": "post input assembly",
        "null_policy": "nullable",
    },
    "feature_coverage": {
        "unit": "coverage-metadata",
        "transform": "identity",
        "annualization_status": "n/a",
        "additive": False,
        "availability_rule": "same as source feature payload",
        "null_policy": "empty",
    },
    "quality_flags": {
        "unit": "flags",
        "transform": "identity",
        "annualization_status": "n/a",
        "additive": False,
        "availability_rule": "same as source feature payload",
        "null_policy": "empty-list",
    },
    "regime_labels": {
        "unit": "labels",
        "transform": "identity",
        "annualization_status": "n/a",
        "additive": False,
        "availability_rule": "same as source feature payload",
        "null_policy": "empty-set",
    },
}


def build_feature_dictionary_signature() -> str:
    """Return SHA256 over canonical JSON serialization.

    This hash is used by the F0 contracts freeze artifact so
    downstream pipelines can detect dictionary-level drift.
    """

    payload = {
        "feature_dictionary_id": feature_dictionary_id,
        "feature_dictionary_version": FEATURE_DICTIONARY_VERSION,
        "entries": forecast_feature_dictionary,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


__all__ = [
    "forecast_feature_dictionary",
    "build_feature_dictionary_signature",
]

