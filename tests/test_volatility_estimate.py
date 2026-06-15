"""Integration tests for ``estimate_variance`` — the public entry-point.

These tests cover the contract obligations the scaffold pinned
as ``xfail(strict=True)``: returning a ``VolEstimate`` with the
canonical fields, raising ``VolSpecError`` on inconsistent specs, and
raising ``VolPITViolationError`` on future-stamped input rows. They
also exercise:

- spec dispatch across all six estimators
- scale-invariance over the registered set
- PIT-property over a Hypothesis-generated PIT-violating frame
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings

from liq.features.volatility import (
    RVSpec,
    TimingPolicy,
    VolCalendarPolicy,
    VolEstimate,
    VolEstimatorSpec,
    VolPITViolationError,
    VolQualityPolicy,
    VolSpecError,
    estimate_variance,
)
from liq.validation.volatility import (
    expected_invariant_tolerance as _tol,
)
from liq.validation.volatility import (
    pit_violating_frame,
    scale_factor,
    valid_ohlc_frame,
)

_HSETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _bars(rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    """Build a polars frame with deterministic daily timestamps."""
    base = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "timestamp": base + timedelta(days=i),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
            }
            for i, (o, h, lo, c) in enumerate(rows)
        ]
    )


def _spec(
    estimator: str = "yang_zhang",
    *,
    target: str = "close_to_close_risk_variance",
    window: int = 5,
    min_periods: int = 5,
    rv_spec: RVSpec | None = None,
) -> VolEstimatorSpec:
    return VolEstimatorSpec(
        estimator=estimator,  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        window=window,
        min_periods=min_periods,
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
        rv_spec=rv_spec,
    )


_FIXTURE_ROWS = [
    (100.0, 102.0, 99.5, 101.0),
    (101.8, 102.0, 100.5, 100.6),
    (99.5, 101.5, 99.4, 101.3),
    (101.3, 101.6, 101.0, 101.35),
    (101.35, 101.4, 98.0, 98.2),
    (98.2, 98.8, 98.0, 98.5),
    (97.0, 97.5, 96.5, 97.0),
    (97.0, 99.0, 96.8, 98.9),
    (99.5, 100.0, 99.4, 99.8),
    (99.8, 100.3, 99.6, 100.1),
]


class TestReturnsVolEstimate:
    def test_returns_vol_estimate_with_required_fields(self) -> None:
        result = estimate_variance(_bars(_FIXTURE_ROWS), _spec())
        assert isinstance(result, VolEstimate)
        assert isinstance(result.var_per_bar, pl.Series)
        assert isinstance(result.vol_per_bar, pl.Series)
        assert result.var_annualized is None  # per_bar_variance output_unit
        assert result.vol_annualized is None
        assert "cont" in result.components
        assert "overnight_gap" in result.components

    def test_var_per_bar_length_matches_input(self) -> None:
        bars = _bars(_FIXTURE_ROWS)
        result = estimate_variance(bars, _spec())
        assert result.var_per_bar.len() == bars.height
        assert result.estimator_used.len() == bars.height
        assert result.valid_from.len() == bars.height

    def test_vol_per_bar_is_sqrt_of_var_per_bar(self) -> None:
        result = estimate_variance(_bars(_FIXTURE_ROWS), _spec())
        for v, s in zip(
            result.var_per_bar.to_list(),
            result.vol_per_bar.to_list(),
            strict=True,
        ):
            if v is None or math.isnan(v):
                assert s is None or math.isnan(s)
            else:
                assert s == pytest.approx(math.sqrt(v))


class TestSpecValidation:
    def test_rv_estimator_without_rv_spec_raises(self) -> None:
        with pytest.raises(VolSpecError, match="rv_spec is required"):
            estimate_variance(_bars(_FIXTURE_ROWS), _spec(estimator="rv"))

    def test_quadratic_variation_target_without_rv_spec_raises(self) -> None:
        with pytest.raises(VolSpecError, match="rv_spec is required"):
            estimate_variance(
                _bars(_FIXTURE_ROWS),
                _spec(estimator="parkinson", target="quadratic_variation"),
            )

    def test_min_periods_greater_than_window_raises(self) -> None:
        with pytest.raises(VolSpecError, match="cannot exceed window"):
            estimate_variance(_bars(_FIXTURE_ROWS), _spec(window=5, min_periods=10))

    def test_missing_columns_raises(self) -> None:
        bare = pl.DataFrame({"timestamp": [datetime(2024, 6, 3, tzinfo=UTC)]})
        with pytest.raises(VolSpecError, match="missing required columns"):
            estimate_variance(bare, _spec())


class TestSpecDispatch:
    @pytest.mark.parametrize(
        "estimator",
        ["ctc", "parkinson", "garman_klass", "rogers_satchell", "gk_yang_zhang", "yang_zhang"],
    )
    def test_dispatches_per_spec(self, estimator: str) -> None:
        target = (
            "close_to_close_risk_variance"
            if estimator in {"ctc", "gk_yang_zhang", "yang_zhang"}
            else "continuous_intraday_variance"
        )
        result = estimate_variance(
            _bars(_FIXTURE_ROWS),
            _spec(estimator=estimator, target=target, window=5, min_periods=5),
        )
        assert result.estimator_used.unique().to_list() == [estimator]


class TestFallbackIntegration:
    def test_missing_open_anywhere_in_window_falls_back_to_parkinson(self) -> None:
        bars = _bars(_FIXTURE_ROWS).with_columns(
            pl.when(pl.arange(0, pl.len()) == 2).then(None).otherwise(pl.col("open")).alias("open")
        )

        result = estimate_variance(bars, _spec(window=5, min_periods=5))

        assert result.estimator_used.unique().to_list() == ["parkinson"]
        assert "MISSING_OPEN" in result.quality_flags.to_list()[2]

    def test_suspect_high_low_anywhere_in_window_falls_back_to_ctc(self) -> None:
        bars = _bars(_FIXTURE_ROWS).with_columns(
            pl.when(pl.arange(0, pl.len()) == 3)
            .then(pl.col("close") - 0.1)
            .otherwise(pl.col("high"))
            .alias("high")
        )

        result = estimate_variance(bars, _spec(window=5, min_periods=5))

        assert result.estimator_used.unique().to_list() == ["ctc"]
        assert "DATA_QUALITY_FAILURE" in result.quality_flags.to_list()[3]


class TestPitViolation:
    def test_future_bar_raises(self) -> None:
        bars = _bars(_FIXTURE_ROWS)
        last_ts = bars.get_column("timestamp")[-1]
        valid_from = bars.get_column("timestamp").to_list()
        # Inject a future-stamped row in the middle.
        valid_from[3] = last_ts + timedelta(days=1)
        bars = bars.with_columns(pl.Series("valid_from", valid_from))
        with pytest.raises(VolPITViolationError, match="PIT violation"):
            estimate_variance(bars, _spec())

    @_HSETTINGS
    @given(pit_violating_frame(n_bars=12))  # type: ignore[missing-argument]
    def test_property_pit_gate_always_fires(self, payload) -> None:
        frame, asof = payload
        with pytest.raises(VolPITViolationError):
            estimate_variance(frame, _spec(window=5, min_periods=5), asof=asof)


class TestScaleInvariance:
    @_HSETTINGS
    @given(valid_ohlc_frame(n_bars=12), scale_factor())  # type: ignore[missing-argument]
    def test_yang_zhang_is_scale_invariant(self, frame, k: float) -> None:
        spec = _spec("yang_zhang", window=5, min_periods=5)
        base = estimate_variance(frame, spec)
        scaled = frame.with_columns(
            (pl.col("open") * k).alias("open"),
            (pl.col("high") * k).alias("high"),
            (pl.col("low") * k).alias("low"),
            (pl.col("close") * k).alias("close"),
        )
        scaled_result = estimate_variance(scaled, spec)
        tolerance = _tol("yang_zhang")
        for a, b in zip(
            base.var_per_bar.to_list(),
            scaled_result.var_per_bar.to_list(),
            strict=True,
        ):
            if a is None or math.isnan(a):
                assert b is None or math.isnan(b)
            else:
                assert abs(a - b) < tolerance, f"YZ not scale-invariant: {a} vs {b}"

    @_HSETTINGS
    @given(valid_ohlc_frame(n_bars=12), scale_factor())  # type: ignore[missing-argument]
    def test_parkinson_is_scale_invariant(self, frame, k: float) -> None:
        spec = _spec("parkinson", target="continuous_intraday_variance", window=5, min_periods=5)
        base = estimate_variance(frame, spec)
        scaled = frame.with_columns(
            (pl.col("open") * k).alias("open"),
            (pl.col("high") * k).alias("high"),
            (pl.col("low") * k).alias("low"),
            (pl.col("close") * k).alias("close"),
        )
        scaled_result = estimate_variance(scaled, spec)
        tolerance = _tol("parkinson")
        for a, b in zip(
            base.var_per_bar.to_list(),
            scaled_result.var_per_bar.to_list(),
            strict=True,
        ):
            if a is None or math.isnan(a):
                assert b is None or math.isnan(b)
            else:
                assert abs(a - b) < tolerance


class TestAnnualizedOutput:
    def test_annualized_variance_requires_periods_per_year(self) -> None:
        spec = VolEstimatorSpec(
            estimator="yang_zhang",
            target="close_to_close_risk_variance",
            window=5,
            min_periods=5,
            output_unit="annualized_variance",
            price_basis="point_in_time_split_adjusted",
            calendar_policy=VolCalendarPolicy(periods_per_year=None),
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
        with pytest.raises(VolSpecError, match="periods_per_year"):
            estimate_variance(_bars(_FIXTURE_ROWS), spec)

    def test_annualized_variance_with_periods_per_year(self) -> None:
        spec = VolEstimatorSpec(
            estimator="yang_zhang",
            target="close_to_close_risk_variance",
            window=5,
            min_periods=5,
            output_unit="annualized_variance",
            price_basis="point_in_time_split_adjusted",
            calendar_policy=VolCalendarPolicy(periods_per_year=252),
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
        result = estimate_variance(_bars(_FIXTURE_ROWS), spec)
        assert result.var_annualized is not None
        assert result.vol_annualized is not None
        # Per-bar = annualized / 252 on every non-NaN bar.
        for pb, ann in zip(
            result.var_per_bar.to_list(),
            result.var_annualized.to_list(),
            strict=True,
        ):
            if pb is None or math.isnan(pb):
                continue
            assert ann == pytest.approx(pb * 252, rel=1e-12)
