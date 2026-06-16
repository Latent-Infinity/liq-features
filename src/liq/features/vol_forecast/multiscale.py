"""Multiscale realized-variance feature construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class MultiscaleVolFeatures:
    """Daily, rolling, and log-space variance features."""

    asof_session: date | None
    daily_var: float | None
    weekly_avg_var: float | None
    monthly_avg_var: float | None
    var_slope: float | None
    log_daily_var: float | None
    log_var_slope: float | None
    feature_coverage: dict[str, object]


def _positive_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _mean(values: Sequence[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return float(np.mean(np.asarray(values[-window:], dtype=float)))


def _slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    centered_x = x - float(np.mean(x))
    denom = float(np.dot(centered_x, centered_x))
    if denom == 0.0:
        return None
    centered_y = y - float(np.mean(y))
    return float(np.dot(centered_x, centered_y) / denom)


def build_multiscale_features(
    realized_variance: Sequence[float],
    *,
    sessions: Sequence[date] | None = None,
    weekly_window: int = 5,
    monthly_window: int = 22,
    floor: float = 1e-12,
) -> MultiscaleVolFeatures:
    """Build daily, weekly, monthly, and slope features from observed variance."""

    if weekly_window <= 0 or monthly_window <= 0:
        raise ValueError("rolling windows must be positive")
    if sessions is not None and len(sessions) != len(realized_variance):
        raise ValueError("sessions and realized_variance must have the same length")

    clean = [float(value) for value in realized_variance if math.isfinite(float(value))]
    if not clean:
        return MultiscaleVolFeatures(
            asof_session=sessions[-1] if sessions else None,
            daily_var=None,
            weekly_avg_var=None,
            monthly_avg_var=None,
            var_slope=None,
            log_daily_var=None,
            log_var_slope=None,
            feature_coverage={
                "observed_count": 0,
                "weekly_count": 0,
                "monthly_count": 0,
                "has_full_weekly_window": False,
                "has_full_monthly_window": False,
            },
        )

    daily_var = _positive_or_none(clean[-1])
    weekly_avg_var = _positive_or_none(_mean(clean, weekly_window))
    monthly_avg_var = _positive_or_none(_mean(clean, monthly_window))
    slope_window = min(len(clean), monthly_window)
    var_slope = _slope(clean[-slope_window:])
    log_values = [math.log(max(value, floor)) for value in clean]
    log_daily_var = math.log(max(clean[-1], floor))
    log_var_slope = _slope(log_values[-slope_window:])

    observed_count = len(clean)
    weekly_count = min(observed_count, weekly_window)
    monthly_count = min(observed_count, monthly_window)
    return MultiscaleVolFeatures(
        asof_session=sessions[-1] if sessions else None,
        daily_var=daily_var,
        weekly_avg_var=weekly_avg_var,
        monthly_avg_var=monthly_avg_var,
        var_slope=var_slope,
        log_daily_var=log_daily_var,
        log_var_slope=log_var_slope,
        feature_coverage={
            "observed_count": observed_count,
            "weekly_count": weekly_count,
            "monthly_count": monthly_count,
            "has_full_weekly_window": observed_count >= weekly_window,
            "has_full_monthly_window": observed_count >= monthly_window,
        },
    )


__all__ = ["MultiscaleVolFeatures", "build_multiscale_features"]
