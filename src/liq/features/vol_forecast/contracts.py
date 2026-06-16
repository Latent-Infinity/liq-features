"""Forecast contracts consumed by the volatility forecast workstream.

This module contains only F0 contract surfaces. Runtime feature
engineering and model objects are intentionally not implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

ForecastOriginType = Literal["EOD", "PRE_OPEN", "INTRADAY"]
ForecastTargetIntervalType = Literal[
    "one_session",
    "multi_session_cumulative",
    "multi_session_average",
    "intraday_interval",
    "event_to_exit",
]
ReasonCodeStage = Literal["feature", "forecast", "sizing", "fallback", "execution"]
ReasonCodeSeverity = Literal["info", "warning", "critical"]
ForecastVolUnit = Literal["per_session_volatility", "session_volatility", "volatility"]


class ForecastingModelInput(TypedDict, total=False):
    """Model-only payload used by forecasting consumers.

    The type lives here to keep payload keys centralized while the
    actual model implementation remains F1+.
    """

    feature_row_id: str
    symbol: str
    forecast_origin_type: ForecastOriginType


@dataclass(frozen=True)
class ReasonCode:
    """Machine-readable reason code for deterministic post-mortems."""

    code: str
    stage: ReasonCodeStage
    severity: ReasonCodeSeverity
    source_object_id: str | None
    details: dict[str, object]


@dataclass(frozen=True)
class VolForecastFeatures:
    """Row-level feature payload for forecast learners.

    Fields are intentionally conservative and align with research
    contract names and units.
    """

    feature_row_id: str
    symbol: str
    forecast_origin_type: ForecastOriginType
    forecast_origin_ts: datetime
    observation_start_ts: datetime
    observation_end_ts: datetime
    availability_ts: datetime
    valid_from: datetime
    valid_until: datetime
    feature_dictionary_id: str
    feature_schema_version: str
    daily_var: float | None
    weekly_avg_var: float | None
    monthly_avg_var: float | None
    var_slope: float | None
    log_daily_var: float | None
    log_var_slope: float | None
    downside_rv: float | None
    upside_rv: float | None
    log_semivar_skew_long: float | None
    log_semivar_skew_short: float | None
    regime_labels: set[str]
    jump_flag: bool
    estimator_uncertainty: float | None
    feature_coverage: dict[str, object]
    quality_flags: list[str]
    estimator_version: str


@dataclass(frozen=True)
class ForecastTarget:
    """Forecast contract emitted by model stage."""

    forecast_id: str
    symbol: str
    forecast_origin_type: ForecastOriginType
    forecast_origin_ts: datetime
    forecast_generated_ts: datetime
    forecast_available_ts: datetime
    valid_from: datetime
    valid_until: datetime
    target_id: str
    target_start_ts: datetime
    target_end_ts: datetime
    target_interval_type: ForecastTargetIntervalType
    horizon_sessions_equiv: float
    target_definition: str
    forecast_var_interval: float
    forecast_var_per_session_equiv: float
    forecast_vol: float
    forecast_var_unit: str
    forecast_vol_unit: ForecastVolUnit
    annualization_factor: int | None
    forecast_uncertainty: float
    forecast_uncertainty_unit: str
    prediction_interval_low: float | None
    prediction_interval_high: float | None
    prediction_interval_unit: str
    model_id: str
    model_version: str
    feature_set_id: str
    training_data_start: datetime
    training_data_end: datetime
    training_cutoff: datetime
    construction_version: str
    fallback_used: bool
    reason_codes: list[ReasonCode]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class SizeVolInput:
    """Sizing input contract consumed by liq-risk."""

    symbol: str
    sizing_decision_ts: datetime
    forecast_id: str | None
    forecast_available_ts: datetime | None
    source_valid_from: datetime
    measurement_id: str | None
    policy_id: str
    policy_version: str
    raw_vol: float
    raw_vol_source: str
    source_measurement_vol: float | None
    source_forecast_var: float | None
    source_forecast_vol: float | None
    previous_size_vol: float | None
    size_vol: float
    sizing_unit: str
    uncertainty_multiplier: float
    floor_applied: bool
    emergency_floor_applied: bool
    cap_applied: bool
    clip_applied: bool
    smoothing_applied: bool
    shock_override_applied: bool
    regime_labels: set[str]
    reason_codes: list[ReasonCode]
    valid_until: datetime


FeatureDictionary = dict[str, dict[str, str | bool | int | float | None]]
FeatureDictionaryMetadata = dict[str, object]

feature_dictionary_id: str = "vol_forecast_features_v1"
FEATURE_DICTIONARY_VERSION: str = "v1.0.0"


feature_dictionary_metadata: FeatureDictionaryMetadata = {
    "feature_dictionary_id": feature_dictionary_id,
    "feature_dictionary_version": FEATURE_DICTIONARY_VERSION,
    "owner": "liq-features",
    "scope": "forecast-only",
}


__all__: Iterable[str] = [
    "ForecastingModelInput",
    "ForecastTarget",
    "ForecastOriginType",
    "ForecastTargetIntervalType",
    "ForecastVolUnit",
    "ReasonCode",
    "ReasonCodeSeverity",
    "ReasonCodeStage",
    "SizeVolInput",
    "VolForecastFeatures",
    "FeatureDictionary",
    "FeatureDictionaryMetadata",
    "feature_dictionary_id",
    "feature_dictionary_metadata",
    "FEATURE_DICTIONARY_VERSION",
]

