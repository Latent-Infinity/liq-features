"""Contract tests for the ``volatility_signature`` VolComponent emission.

Closes the impl plan's microstructure-iteration deliverable:

    Volatility-signature plot data emitted as a ``VolComponent`` named
    ``volatility_signature`` when minute data are available.

The signature is the per-bar realized variance computed at the
``rv_noise_gate``-resolved interval (5m when the gate fires, 1m
otherwise). The component's ``source`` is ``"minute_rv"`` when minute
data is provided.

Behavior pinned:

- No minute data → no ``volatility_signature`` key in ``components``.
- Minute data provided → ``volatility_signature`` carries one RV value
  per bar; bars without minute data are NaN.
- Gate fires for a bar → emitter logs ``rv_noise_gate_fired`` once for
  that bar and the signature value uses the 5m fallback.
- Gate stays down → the 1m RV is used.
- 5m / 15m RV are computed from grouped minute returns, not strided
  one-minute returns.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

import polars as pl

from liq.features.volatility import (
    TimingPolicy,
    VolCalendarPolicy,
    VolEstimatorSpec,
    VolQualityPolicy,
    estimate_variance,
)
from liq.features.volatility.logging import LOGGER_NAME
from liq.features.volatility.quality import FLAG_NOISY_RV_TARGET


def _spec(*, rv_noise_ratio_threshold: float = 2.0) -> VolEstimatorSpec:
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
            rv_noise_ratio_threshold=rv_noise_ratio_threshold,
            estimator_dispersion_threshold=1.5,
            min_nonzero_volume_fraction=0.95,
        ),
        rv_spec=None,
    )


def _bars() -> pl.DataFrame:
    base = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "timestamp": base + timedelta(days=i),
                "open": 100.0 + i,
                "high": 101.5 + i,
                "low": 99.5 + i,
                "close": 100.5 + i,
            }
            for i in range(10)
        ]
    )


def _calm_minute_returns(seed: int, n_minutes: int = 390) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    sigma_per_minute = 0.0126 / math.sqrt(n_minutes)
    return rng.normal(0.0, sigma_per_minute, size=n_minutes).tolist()


def _noisy_minute_returns(seed: int, n_minutes: int = 390) -> list[float]:
    """Returns whose 1m sampling RV is inflated relative to 5m/15m so
    the §5.3 noise gate fires."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sigma_per_minute = 0.0126 / math.sqrt(n_minutes)
    log_path = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, sigma_per_minute, size=n_minutes))])
    # Add observation noise so adjacent diffs have noise variance
    # comparable to the true per-minute variance.
    noise = rng.normal(0.0, 0.002, size=log_path.shape)
    log_path = log_path + noise
    return list(np.diff(log_path))


def _alternating_noise_returns() -> list[float]:
    group = [0.01, -0.01, 0.01, -0.01]
    returns: list[float] = []
    for group_idx in range(6):
        remainder = 0.0005 if group_idx % 2 == 0 else -0.0005
        returns.extend([*group, remainder])
    return returns


def _grouped_rv(returns: list[float], stride: int) -> float:
    grouped = [sum(returns[i : i + stride]) for i in range(0, len(returns), stride)]
    return sum(r * r for r in grouped)


class TestNoMinuteData:
    def test_volatility_signature_absent_when_no_minute_data(self) -> None:
        result = estimate_variance(_bars(), _spec())
        assert "volatility_signature" not in result.components


class TestMinuteDataProvided:
    def test_volatility_signature_emitted_with_minute_rv_source(self) -> None:
        intra_bar_returns = {i: _calm_minute_returns(seed=1000 + i) for i in range(10)}
        result = estimate_variance(_bars(), _spec(), intra_bar_returns=intra_bar_returns)
        assert "volatility_signature" in result.components
        sig = result.components["volatility_signature"]
        assert sig.source == "minute_rv"
        assert sig.value.len() == _bars().height
        # All bars covered → no NaN values.
        for v in sig.value.to_list():
            assert v is not None and not math.isnan(v)

    def test_volatility_signature_is_nan_for_uncovered_bars(self) -> None:
        # Only bars 2, 4 have minute data; the rest emit NaN.
        intra_bar_returns = {
            2: _calm_minute_returns(seed=2000),
            4: _calm_minute_returns(seed=2001),
        }
        result = estimate_variance(_bars(), _spec(), intra_bar_returns=intra_bar_returns)
        values = result.components["volatility_signature"].value.to_list()
        assert not math.isnan(values[2])
        assert not math.isnan(values[4])
        assert math.isnan(values[0])
        assert math.isnan(values[9])

    def test_noise_gate_fires_for_inflated_bar_and_logs(self, caplog) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        # Bar 3 has microstructure-inflated minute data; bar 5 is calm.
        intra_bar_returns = {
            3: _noisy_minute_returns(seed=3000),
            5: _calm_minute_returns(seed=3001),
        }
        result = estimate_variance(_bars(), _spec(), intra_bar_returns=intra_bar_returns)
        gate_events = [
            r
            for r in caplog.records
            if getattr(r, "structured", {}).get("event") == "rv_noise_gate_fired"
        ]
        # At least one bar tripped the gate.
        assert gate_events, "expected at least one rv_noise_gate_fired event"
        # The first such event carries the canonical fields.
        payload = getattr(gate_events[0], "structured", {})
        for required in ("rv_1m", "rv_5m", "rv_15m", "price_movement", "bar_index"):
            assert required in payload, f"missing field {required}: {payload}"
        flagged_bar = int(payload["bar_index"])
        assert FLAG_NOISY_RV_TARGET in result.quality_flags.to_list()[flagged_bar]
        assert (
            FLAG_NOISY_RV_TARGET
            in result.components["volatility_signature"].quality_flags.to_list()[flagged_bar]
        )

    def test_gate_fallback_uses_grouped_five_minute_returns(self, caplog) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        returns = _alternating_noise_returns()
        result = estimate_variance(_bars(), _spec(), intra_bar_returns={3: returns})
        signature = result.components["volatility_signature"].value.to_list()[3]
        assert signature == _grouped_rv(returns, stride=5)

    def test_noise_threshold_comes_from_quality_policy(self, caplog) -> None:
        caplog.set_level(logging.INFO, logger=LOGGER_NAME)
        returns = _alternating_noise_returns()
        result = estimate_variance(
            _bars(),
            _spec(rv_noise_ratio_threshold=10_000.0),
            intra_bar_returns={3: returns},
        )
        signature = result.components["volatility_signature"].value.to_list()[3]
        assert signature == sum(r * r for r in returns)
        gate_events = [
            r
            for r in caplog.records
            if getattr(r, "structured", {}).get("event") == "rv_noise_gate_fired"
        ]
        assert not gate_events
