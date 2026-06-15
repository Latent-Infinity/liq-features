"""Contract tests for the minute-mode RV / BPV / JV helpers + RV-noise gate.

Anchors research plan §5.3 (RV-noise gate) and ``[APPENDIX_FORMULAS]``
(RV / BPV definitions). Each test pins one row of the contract:

- ``compute_rv`` = ``Σ r_i²`` (sum of squared log-returns).
- ``compute_bpv`` = ``(π/2) · Σ |r_{i-1}| · |r_i|`` (bipower variation,
  jump-robust continuous-variance proxy).
- ``compute_jv`` = ``max(RV - BPV, 0)``.
- ``rv_noise_gate`` fires when ``RV_1m`` materially exceeds
  ``RV_5m / RV_15m`` WITHOUT corresponding price movement; that is the
  hard noise gate from research plan §5.3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from liq.features.volatility.rv import (
    compute_bpv,
    compute_jv,
    compute_rv,
    rv_noise_gate,
)


class TestComputeRv:
    def test_zero_returns_yield_zero_rv(self) -> None:
        assert compute_rv([0.0, 0.0, 0.0]) == 0.0

    def test_sum_of_squared_returns(self) -> None:
        returns = [0.01, -0.02, 0.005, -0.015]
        expected = sum(r * r for r in returns)
        assert compute_rv(returns) == pytest.approx(expected, rel=1e-12)

    def test_unbiased_on_gbm(self) -> None:
        """For 5000 i.i.d. Normal(0, σ_per_step) returns, RV should
        approximate the integrated true variance ``n_steps * σ²``. With
        n=5000, the standard error is ≈ sqrt(2/n) ≈ 2% of σ², so a 6%
        tolerance bounds it at ~3σ comfortably."""
        rng = np.random.default_rng(20240615)
        sigma = 0.001
        n_steps = 5000
        returns = rng.normal(0.0, sigma, size=n_steps).tolist()
        rv = compute_rv(returns)
        true_iv = n_steps * sigma * sigma
        assert abs(rv - true_iv) / true_iv < 0.06

    def test_empty_returns_zero(self) -> None:
        assert compute_rv([]) == 0.0


class TestComputeBpv:
    def test_zero_returns_yield_zero_bpv(self) -> None:
        assert compute_bpv([0.0, 0.0, 0.0]) == 0.0

    def test_constant_returns_scale_correctly(self) -> None:
        """For ``r_i = c`` for all i: |r_{i-1}| * |r_i| = c² for each pair,
        so BPV = (π/2) * (n-1) * c²."""
        c = 0.01
        returns = [c] * 5
        expected = (math.pi / 2.0) * (len(returns) - 1) * c * c
        assert compute_bpv(returns) == pytest.approx(expected, rel=1e-12)

    def test_single_return_yields_zero_bpv(self) -> None:
        """BPV requires at least 2 returns to form a pair."""
        assert compute_bpv([0.01]) == 0.0

    def test_bpv_approximately_equal_to_rv_under_gbm(self) -> None:
        """Under pure GBM with no jumps, BPV converges to RV (both are
        unbiased for integrated variance). With n=2000 samples the
        relative gap should be modest."""
        rng = np.random.default_rng(20240615)
        sigma = 0.001
        n_steps = 2000
        returns = rng.normal(0.0, sigma, size=n_steps).tolist()
        rv = compute_rv(returns)
        bpv = compute_bpv(returns)
        # BPV and RV should agree to within ~10% under no-jump GBM.
        assert abs(bpv - rv) / rv < 0.15


class TestComputeJv:
    def test_jv_is_max_rv_minus_bpv_zero(self) -> None:
        returns = [0.01, -0.02, 0.005, -0.015]
        rv = compute_rv(returns)
        bpv = compute_bpv(returns)
        jv = compute_jv(returns)
        assert jv == pytest.approx(max(rv - bpv, 0.0), rel=1e-12)

    def test_jv_isolates_a_jump(self) -> None:
        """Inject a single 5% jump into an otherwise calm return series;
        JV should capture the jump variance, BPV should stay near the
        continuous part."""
        rng = np.random.default_rng(20240615)
        sigma = 0.001
        n_steps = 500
        returns = list(rng.normal(0.0, sigma, size=n_steps))
        returns[250] = 0.05  # 5% jump
        rv = compute_rv(returns)
        bpv = compute_bpv(returns)
        jv = compute_jv(returns)
        # The jump's squared return is 0.0025. RV picks it up; BPV largely
        # discounts it because the |r_{i-1}| * |r_i| pair around it is
        # small * 0.05 + 0.05 * small.
        assert jv > 0.001  # captured most of the jump variance
        assert rv > bpv  # RV > BPV by construction

    def test_jv_nonnegative_when_no_jump(self) -> None:
        rng = np.random.default_rng(20240615)
        returns = list(rng.normal(0.0, 0.001, size=200))
        assert compute_jv(returns) >= 0.0


class TestRvNoiseGate:
    def test_gate_fires_on_inflated_one_minute_rv(self) -> None:
        """The canonical noise scenario: ``RV_1m`` is ~3× RV_5m and
        RV_15m while price barely moved. The gate must fire."""
        rv_by_interval = {"1m": 0.0030, "5m": 0.0010, "15m": 0.0008}
        # Price moved 0.1% over the session — small.
        assert rv_noise_gate(rv_by_interval, price_movement=0.001) is True

    def test_gate_does_not_fire_when_rv_ratios_match(self) -> None:
        rv_by_interval = {"1m": 0.0011, "5m": 0.0010, "15m": 0.0010}
        assert rv_noise_gate(rv_by_interval, price_movement=0.001) is False

    def test_gate_does_not_fire_when_price_moved_a_lot(self) -> None:
        """Even if RV_1m is inflated, a large concurrent price movement
        is a legit volatility shock — the noise gate stays down."""
        rv_by_interval = {"1m": 0.0030, "5m": 0.0010, "15m": 0.0008}
        # Price moved 5%; the inflated RV_1m is consistent with the
        # actual move, not microstructure noise.
        assert rv_noise_gate(rv_by_interval, price_movement=0.05) is False

    def test_gate_missing_intervals_raises(self) -> None:
        with pytest.raises(KeyError):
            rv_noise_gate({"1m": 0.001}, price_movement=0.001)
