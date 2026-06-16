"""Contract tests for the volatility forecast contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from dataclasses import FrozenInstanceError

import pytest

from liq.features.vol_forecast import (
    ForecastTarget,
    VolForecastFeatures,
    ReasonCode,
    SizeVolInput,
    assert_no_straddle,
    assert_feature_forecast_clock,
    assert_forecast_size_clock,
    assert_forecast_target_clock,
    assert_sizing_order_clock,
)
from liq.features.vol_forecast.contracts import (
    FEATURE_DICTIONARY_VERSION,
    feature_dictionary_id,
    feature_dictionary_metadata,
)
from liq.features.vol_forecast.feature_dictionary import forecast_feature_dictionary


def _base_features() -> VolForecastFeatures:
    now = datetime(2024, 1, 2, 20, 0, 0, tzinfo=UTC)
    return VolForecastFeatures(
        feature_row_id="row_1",
        symbol="AAPL",
        forecast_origin_type="EOD",
        forecast_origin_ts=now,
        observation_start_ts=now,
        observation_end_ts=now,
        availability_ts=now - timedelta(hours=1),
        valid_from=now,
        valid_until=now + timedelta(days=7),
        feature_dictionary_id=feature_dictionary_id,
        feature_schema_version=FEATURE_DICTIONARY_VERSION,
        daily_var=0.001,
        weekly_avg_var=0.0012,
        monthly_avg_var=0.0014,
        var_slope=0.01,
        log_daily_var=-6.9,
        log_var_slope=-7.1,
        downside_rv=0.0004,
        upside_rv=0.0006,
        log_semivar_skew_long=0.02,
        log_semivar_skew_short=-0.01,
        regime_labels={"UNKNOWN"},
        jump_flag=False,
        estimator_uncertainty=0.001,
        feature_coverage={"weekly_count": 5, "has_full_weekly_window": True},
        quality_flags=[],
        estimator_version="v0.1",
    )


def _base_forecast() -> ForecastTarget:
    now = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    return ForecastTarget(
        forecast_id="fc_1",
        symbol="AAPL",
        forecast_origin_type="EOD",
        forecast_origin_ts=now,
        forecast_generated_ts=now,
        forecast_available_ts=now + timedelta(minutes=1),
        valid_from=now,
        valid_until=now + timedelta(days=2),
        target_id="tgt_1",
        target_start_ts=now + timedelta(minutes=1),
        target_end_ts=now + timedelta(days=1),
        target_interval_type="one_session",
        horizon_sessions_equiv=1.0,
        target_definition="target_rv_total",
        forecast_var_interval=0.002,
        forecast_var_per_session_equiv=0.002,
        forecast_vol=0.0447,
        forecast_var_unit="per_session_variance",
        forecast_vol_unit="per_session_volatility",
        annualization_factor=252,
        forecast_uncertainty=0.05,
        forecast_uncertainty_unit="dimensionless",
        prediction_interval_low=0.0015,
        prediction_interval_high=0.0025,
        prediction_interval_unit="per_session_variance",
        model_id="model_a",
        model_version="v1",
        feature_set_id="fs_1",
        training_data_start=now - timedelta(days=365),
        training_data_end=now - timedelta(days=1),
        training_cutoff=now - timedelta(hours=1),
        construction_version="v1",
        fallback_used=False,
        reason_codes=[
            ReasonCode(
                code="NONE",
                stage="forecast",
                severity="info",
                source_object_id=None,
                details={},
            )
        ],
        diagnostics={"coverage": 1.0},
    )


def _base_size_input() -> SizeVolInput:
    now = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    return SizeVolInput(
        symbol="AAPL",
        sizing_decision_ts=now + timedelta(minutes=5),
        forecast_id="fc_1",
        forecast_available_ts=now + timedelta(minutes=1),
        source_valid_from=now,
        measurement_id="m_1",
        policy_id="default",
        policy_version="v1",
        raw_vol=0.01,
        raw_vol_source="forecast_vol",
        source_measurement_vol=None,
        source_forecast_var=0.002,
        source_forecast_vol=0.0447,
        previous_size_vol=0.008,
        size_vol=0.009,
        sizing_unit="volatility",
        uncertainty_multiplier=1.0,
        floor_applied=False,
        emergency_floor_applied=False,
        cap_applied=False,
        clip_applied=False,
        smoothing_applied=False,
        shock_override_applied=False,
        regime_labels={"UNKNOWN"},
        reason_codes=[
            ReasonCode(
                code="NONE",
                stage="sizing",
                severity="info",
                source_object_id=None,
                details={},
            )
        ],
        valid_until=now + timedelta(hours=8),
    )


def test_contracts_are_frozen() -> None:
    """Public contracts remain immutable by construction."""
    feature = _base_features()
    forecast = _base_forecast()
    size_input = _base_size_input()

    for value in (feature, forecast, size_input):
        with pytest.raises(FrozenInstanceError):
            setattr(value, next(iter(value.__dict__)), "mutated")


def test_dictionary_contains_expected_fields() -> None:
    """Feature dictionary includes all forecast features and has metadata."""

    assert feature_dictionary_id
    assert FEATURE_DICTIONARY_VERSION
    assert feature_dictionary_metadata["feature_dictionary_id"] == feature_dictionary_id
    required = {
        "daily_var",
        "weekly_avg_var",
        "monthly_avg_var",
        "var_slope",
        "log_daily_var",
        "log_var_slope",
        "downside_rv",
        "upside_rv",
        "log_semivar_skew_long",
        "log_semivar_skew_short",
        "estimator_uncertainty",
        "feature_coverage",
        "quality_flags",
        "regime_labels",
    }
    assert required.issubset(forecast_feature_dictionary.keys())

    for key in required:
        field = forecast_feature_dictionary[key]
        for field_key in (
            "unit",
            "transform",
            "annualization_status",
            "additive",
            "availability_rule",
            "null_policy",
        ):
            assert field_key in field


def test_serving_invariants_are_enforced() -> None:
    feature = _base_features()
    forecast = _base_forecast()
    size_input = _base_size_input()

    assert_feature_forecast_clock(feature=feature, forecast=forecast)
    assert_forecast_target_clock(forecast=forecast)
    assert_forecast_size_clock(forecast=forecast, size_input=size_input)
    assert_sizing_order_clock(size_input=size_input, order_decision_ts=forecast.forecast_origin_ts)


def test_no_straddle_invariant() -> None:
    feature = _base_features()
    forecast = _base_forecast()
    assert_no_straddle(feature=feature, forecast=forecast)
