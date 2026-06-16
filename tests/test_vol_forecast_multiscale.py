from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from liq.features.vol_forecast import build_multiscale_features


def test_multiscale_windows_include_current_session() -> None:
    values = [float(idx) for idx in range(1, 23)]
    sessions = [date(2024, 1, 1) + timedelta(days=idx) for idx in range(22)]

    features = build_multiscale_features(values, sessions=sessions)

    assert features.daily_var == 22.0
    assert features.weekly_avg_var == pytest.approx(sum(range(18, 23)) / 5)
    assert features.monthly_avg_var == pytest.approx(sum(range(1, 23)) / 22)
    assert features.var_slope == pytest.approx(1.0)
    assert features.log_daily_var == pytest.approx(math.log(22.0))
    assert features.feature_coverage["weekly_count"] == 5
    assert features.feature_coverage["has_full_monthly_window"] is True


def test_multiscale_partial_coverage_is_explicit() -> None:
    features = build_multiscale_features([0.01, 0.02, 0.03])

    assert features.daily_var == 0.03
    assert features.weekly_avg_var is None
    assert features.monthly_avg_var is None
    assert features.feature_coverage["weekly_count"] == 3
    assert features.feature_coverage["has_full_weekly_window"] is False
