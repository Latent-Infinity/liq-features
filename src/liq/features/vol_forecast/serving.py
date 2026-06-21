"""Serving-clock invariants for forecast contracts."""

from __future__ import annotations

from datetime import datetime

from .contracts import ForecastTarget, SizeVolInput, VolForecastFeatures


def assert_feature_forecast_clock(
    *, feature: VolForecastFeatures, forecast: ForecastTarget
) -> None:
    """Feature → forecast invariant.

    ``availability_ts`` must be at or before the forecast origin.
    """

    assert feature.availability_ts <= forecast.forecast_origin_ts


def assert_forecast_target_clock(*, forecast: ForecastTarget) -> None:
    """Forecast target-time ordering invariant.

    ``forecast_origin_ts`` must be strictly before target start,
    and target window must be non-empty.
    """

    assert forecast.forecast_origin_ts < forecast.target_start_ts
    assert forecast.target_start_ts <= forecast.target_end_ts
    assert forecast.forecast_generated_ts <= forecast.forecast_available_ts


def assert_forecast_size_clock(*, forecast: ForecastTarget, size_input: SizeVolInput) -> None:
    """Forecast → sizing invariant.

    If a forecast exists, sizing should consume after forecast
    availability.
    """

    if size_input.forecast_available_ts is not None:
        assert forecast.forecast_available_ts <= size_input.sizing_decision_ts
    assert size_input.source_valid_from <= size_input.sizing_decision_ts
    assert size_input.source_valid_from <= size_input.valid_until


def assert_sizing_order_clock(*, size_input: SizeVolInput, order_decision_ts: datetime) -> None:
    """Sizing → order invariant with half-open right bound."""

    assert size_input.source_valid_from <= order_decision_ts
    assert order_decision_ts < size_input.valid_until


def assert_no_straddle(*, feature: VolForecastFeatures, forecast: ForecastTarget) -> None:
    """No feature straddling invariant.

    Feature observations must not include timestamps past forecast
    origin for non-partial rows.
    """

    assert feature.observation_end_ts <= forecast.forecast_origin_ts


__all__ = [
    "assert_feature_forecast_clock",
    "assert_forecast_target_clock",
    "assert_forecast_size_clock",
    "assert_sizing_order_clock",
    "assert_no_straddle",
]
