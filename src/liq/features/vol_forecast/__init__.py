"""Forecast-feature contracts for the volatility forecast workstream."""

from .contracts import (
    ForecastTarget,
    ForecastingModelInput,
    FEATURE_DICTIONARY_VERSION,
    ForecastTargetIntervalType,
    ForecastOriginType,
    ForecastVolUnit,
    SizeVolInput,
    ReasonCode,
    ReasonCodeSeverity,
    ReasonCodeStage,
    VolForecastFeatures,
    feature_dictionary_metadata,
    feature_dictionary_id,
)
from .feature_dictionary import forecast_feature_dictionary, build_feature_dictionary_signature
from .serving import (
    assert_feature_forecast_clock,
    assert_forecast_target_clock,
    assert_no_straddle,
    assert_forecast_size_clock,
    assert_sizing_order_clock,
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
    "SizeVolInput",
    "assert_feature_forecast_clock",
    "assert_forecast_target_clock",
    "assert_no_straddle",
    "assert_forecast_size_clock",
    "assert_sizing_order_clock",
    "forecast_feature_dictionary",
    "build_feature_dictionary_signature",
]
