"""Contract tests anchoring the canonical risk-variance estimator.

Originally three of these were ``pytest.mark.xfail(strict=True)`` —
the scaffolded ``estimate_variance`` raised ``NotImplementedError`` and
the strict-xfail discipline guaranteed those tests flipped green only
when the formula registry satisfied the contract. With the formula
registry now in place, the xfail markers are gone and these tests run
as plain contract checks. The filename retains the ``_xfail`` suffix
out of stage-only respect — the operator chooses the rename ceremony.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from liq.features import volatility as volpkg
from liq.features.volatility import (
    RVSpec,
    TimingPolicy,
    VolCalendarPolicy,
    VolEstimate,
    VolEstimatorSpec,
    VolFeatureError,
    VolPITViolationError,
    VolQualityPolicy,
    VolSpecError,
    estimate_variance,
)


def _baseline_spec() -> VolEstimatorSpec:
    """Smallest valid ``VolEstimatorSpec`` for contract tests."""
    return VolEstimatorSpec(
        estimator="yang_zhang",
        target="close_to_close_risk_variance",
        window=5,
        min_periods=5,
        output_unit="per_bar_variance",
        price_basis="point_in_time_split_adjusted",
        calendar_policy=VolCalendarPolicy(),
        timing_policy=TimingPolicy(
            bar_timestamp_semantics="end_time",
            data_latency_policy="close_plus_0",
            corporate_action_policy="point_in_time_adjusted",
        ),
        quality_policy=VolQualityPolicy(
            high_low_outlier_method="past_rolling_z",
            high_low_outlier_threshold=4.0,
            rv_noise_ratio_threshold=2.0,
            estimator_dispersion_threshold=1.5,
            min_nonzero_volume_fraction=0.95,
        ),
        rv_spec=None,
    )


def _minimal_bars(n: int = 10) -> pl.DataFrame:
    base = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": 100.0 + i,
                "high": 101.5 + i,
                "low": 99.5 + i,
                "close": 100.5 + i,
            }
        )
    return pl.DataFrame(rows)


class TestPublicSurfaceExists:
    """Scaffold-era guarantees — the import surface is stable. These
    pin the contract types, exception hierarchy, and entry-point
    name."""

    def test_estimate_variance_is_callable(self) -> None:
        assert callable(estimate_variance)

    def test_contract_types_present(self) -> None:
        assert volpkg.VolEstimatorSpec is VolEstimatorSpec
        assert volpkg.VolCalendarPolicy is VolCalendarPolicy
        assert volpkg.TimingPolicy is TimingPolicy
        assert volpkg.VolQualityPolicy is VolQualityPolicy
        assert volpkg.RVSpec is RVSpec
        assert hasattr(volpkg, "VolComponent")
        assert hasattr(volpkg, "VolEstimate")

    def test_exception_hierarchy(self) -> None:
        assert issubclass(VolSpecError, VolFeatureError)
        assert issubclass(VolPITViolationError, VolFeatureError)
        assert issubclass(volpkg.VolDataQualityError, VolFeatureError)
        assert issubclass(volpkg.VolUnavailableError, VolFeatureError)

    def test_legacy_helpers_still_importable(self) -> None:
        """``yang_zhang`` and ``garman_klass`` remain importable through
        the new package so existing callers keep working until the
        ATR-bridge retires them."""
        assert callable(volpkg.yang_zhang)
        assert callable(volpkg.garman_klass)

    def test_spec_is_frozen(self) -> None:
        spec = _baseline_spec()
        with pytest.raises(FrozenInstanceError):
            spec.window = 42  # type: ignore[misc]


class TestImplementationContract:
    """Contract obligations from research plan §3.4 + §3.6 — the
    formula registry must satisfy them. These were the original
    strict-xfail scaffold pins; with the registry landed, they run
    green."""

    def test_returns_vol_estimate_with_required_fields(self) -> None:
        result = estimate_variance(_minimal_bars(), _baseline_spec())
        assert isinstance(result, VolEstimate)
        assert hasattr(result, "var_per_bar")
        assert hasattr(result, "vol_per_bar")
        assert hasattr(result, "components")
        assert hasattr(result, "quality_flags")
        assert hasattr(result, "estimator_used")
        assert hasattr(result, "valid_from")

    def test_spec_validation_raises_vol_spec_error_for_missing_rv_spec(self) -> None:
        rv_only_spec = VolEstimatorSpec(
            estimator="rv",
            target="quadratic_variation",
            window=21,
            min_periods=21,
            output_unit="per_bar_variance",
            price_basis="raw",
            calendar_policy=VolCalendarPolicy(),
            timing_policy=TimingPolicy(
                bar_timestamp_semantics="end_time",
                data_latency_policy="close_plus_0",
                corporate_action_policy="raw_for_live",
            ),
            quality_policy=VolQualityPolicy(
                high_low_outlier_method="none",
                high_low_outlier_threshold=4.0,
                rv_noise_ratio_threshold=2.0,
                estimator_dispersion_threshold=1.5,
                min_nonzero_volume_fraction=0.95,
            ),
            rv_spec=None,
        )
        with pytest.raises(VolSpecError):
            estimate_variance(_minimal_bars(), rv_only_spec)

    def test_pit_violation_raises(self) -> None:
        """A row with ``valid_from > asof`` triggers
        ``VolPITViolationError``. The PIT gate enforces research plan
        §3.5's hard rule on every emission."""
        bars = _minimal_bars()
        valid_from = bars.get_column("timestamp").to_list()
        # Force the third row to be future-stamped past the last bar.
        valid_from[2] = valid_from[-1] + timedelta(days=2)
        bars = bars.with_columns(pl.Series("valid_from", valid_from))
        with pytest.raises(VolPITViolationError):
            estimate_variance(bars, _baseline_spec())
