"""Forecast-feature contracts for the volatility forecast workstream."""

from .contracts import (
    FEATURE_DICTIONARY_VERSION,
    LOAD_BEARING_TIMESTAMP_FIELDS,
    ForecastingModelInput,
    ForecastOriginType,
    ForecastTarget,
    ForecastTargetIntervalType,
    ForecastVolUnit,
    ReasonCode,
    ReasonCodeSeverity,
    ReasonCodeStage,
    SizeVolInput,
    VolForecastFeatures,
    feature_dictionary_id,
    feature_dictionary_metadata,
)
from .feature_dictionary import build_feature_dictionary_signature, forecast_feature_dictionary
from .multiscale import MultiscaleVolFeatures, build_multiscale_features
from .serving import (
    assert_feature_forecast_clock,
    assert_forecast_size_clock,
    assert_forecast_target_clock,
    assert_no_straddle,
    assert_sizing_order_clock,
)
from .targets import TargetRvTotal, build_target_rv_total
from .universes import (
    UNIVERSE_DEFINITION_ID,
    BaselineEligibility,
    UniverseMembership,
    compute_universe_membership,
)

__all__ = [
    "ForecastOriginType",
    "ForecastTargetIntervalType",
    "ForecastVolUnit",
    "ForecastingModelInput",
    "ReasonCode",
    "ReasonCodeSeverity",
    "ReasonCodeStage",
    "VolForecastFeatures",
    "ForecastTarget",
    "feature_dictionary_id",
    "feature_dictionary_metadata",
    "FEATURE_DICTIONARY_VERSION",
    "LOAD_BEARING_TIMESTAMP_FIELDS",
    "SizeVolInput",
    "assert_feature_forecast_clock",
    "assert_forecast_target_clock",
    "assert_no_straddle",
    "assert_forecast_size_clock",
    "assert_sizing_order_clock",
    "forecast_feature_dictionary",
    "build_feature_dictionary_signature",
    "TargetRvTotal",
    "build_target_rv_total",
    "MultiscaleVolFeatures",
    "build_multiscale_features",
    "BaselineEligibility",
    "UniverseMembership",
    "UNIVERSE_DEFINITION_ID",
    "compute_universe_membership",
]
