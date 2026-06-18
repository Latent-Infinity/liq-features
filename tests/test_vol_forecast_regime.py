from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from liq.features.vol_forecast import (
    GAP_DOMINATED_VOL,
    INTRADAY_RANGE_DOMINATED_VOL,
    JUMP_DAY,
    compute_asymmetry_regression,
    compute_semivariance,
    derive_gap_jump_labels,
    resolve_multi_label,
)

_RETURNS_STRATEGY = st.lists(
    st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=64,
)
_PROPERTY_SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def test_semivariance_components_add_to_total_realized_variance() -> None:
    estimate = compute_semivariance([-0.02, 0.01, -0.03, 0.04], window=4)

    assert estimate.downside_rv + estimate.upside_rv == pytest.approx(estimate.bar_rv)
    assert estimate.downside_rv == pytest.approx(0.0013)
    assert estimate.upside_rv == pytest.approx(0.0017)
    assert estimate.coverage["has_full_window"] is True


@_PROPERTY_SETTINGS
@given(returns=_RETURNS_STRATEGY, window=st.integers(min_value=1, max_value=64))
def test_semivariance_decomposition_property(returns: list[float], window: int) -> None:
    estimate = compute_semivariance(returns, window=window)

    assert estimate.downside_rv >= 0.0
    assert estimate.upside_rv >= 0.0
    assert estimate.downside_rv + estimate.upside_rv == pytest.approx(estimate.bar_rv, abs=1e-12)
    expected_bar = float(np.sum(np.asarray(returns[-window:], dtype=float) ** 2))
    assert estimate.bar_rv == pytest.approx(expected_bar, abs=1e-12)


def test_semivariance_partial_window_coverage_is_explicit() -> None:
    estimate = compute_semivariance([0.01, -0.02], window=5)

    assert estimate.coverage["observed_count"] == 2
    assert estimate.coverage["has_full_window"] is False


def test_asymmetry_regression_symmetric_series_has_near_zero_interaction() -> None:
    result = compute_asymmetry_regression([-0.02, 0.02] * 20)

    assert result.observations > 0
    assert result.b3 == pytest.approx(0.0, abs=1e-12)


def test_gap_jump_labels_derive_diagnostics_from_estimate_fields() -> None:
    labels = derive_gap_jump_labels(
        {
            "overnight_gap_var": 0.006,
            "intraday_range_var": 0.005,
            "jump_var": 0.004,
            "total_var": 0.01,
        },
        threshold_gap=0.5,
        threshold_range=0.4,
        threshold_jump=0.3,
    )

    assert labels == {GAP_DOMINATED_VOL, INTRADAY_RANGE_DOMINATED_VOL, JUMP_DAY}


def test_gap_jump_labels_estimator_flag_forces_jump_day() -> None:
    labels = derive_gap_jump_labels(
        {
            "overnight_gap_var": 0.001,
            "intraday_range_var": 0.001,
            "jump_var": 0.001,
            "total_var": 0.01,
            "jump_flag": True,
        },
        threshold_gap=0.5,
        threshold_range=0.5,
        threshold_jump=0.5,
    )

    assert JUMP_DAY in labels
    assert GAP_DOMINATED_VOL not in labels
    assert INTRADAY_RANGE_DOMINATED_VOL not in labels


def test_gap_jump_labels_thresholds_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="threshold_gap"):
        derive_gap_jump_labels(
            {"overnight_gap_var": 0.0, "intraday_range_var": 0.0, "jump_var": 0.0, "total_var": 1.0},
            threshold_gap=-0.1,
            threshold_range=0.5,
            threshold_jump=0.5,
        )


def test_multi_label_resolution_uses_max_multiplier_not_product() -> None:
    resolved = resolve_multi_label(
        {GAP_DOMINATED_VOL, JUMP_DAY, INTRADAY_RANGE_DOMINATED_VOL},
        {
            GAP_DOMINATED_VOL: 1.2,
            JUMP_DAY: 1.5,
            INTRADAY_RANGE_DOMINATED_VOL: 1.3,
        },
    )

    assert resolved == {JUMP_DAY}


def test_multi_label_resolution_ties_keep_all_winners() -> None:
    resolved = resolve_multi_label(
        {GAP_DOMINATED_VOL, JUMP_DAY},
        {GAP_DOMINATED_VOL: 1.4, JUMP_DAY: 1.4},
    )

    assert resolved == {GAP_DOMINATED_VOL, JUMP_DAY}
