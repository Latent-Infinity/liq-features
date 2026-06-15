"""Golden + property tests for the windowed-aggregation layer.

The simple estimators (CtC, Parkinson, GK, RS, GK-YZ) aggregate by a
trailing mean; Yang-Zhang composes three component means with the
canonical ``k_yz`` weighting. Reference values come from the
arbitrary-precision Decimal computation in
``liq.validation.volatility.golden_fixtures``.

Tolerance: per research plan §10.1 gate-parameter table, ``1e-10`` for
windowed output (vs the ``1e-12`` per-bar floor) — accumulated float
error in the windowed sum is bounded by this.
"""

from __future__ import annotations

import math
from typing import Literal

import pytest

from liq.features.volatility.estimators.formulas import (
    ctc_var_contribution,
    garman_klass_var_contribution,
    gk_yang_zhang_var_contribution,
    parkinson_var_contribution,
    rogers_satchell_var_contribution,
    yz_open_close_term,
    yz_overnight_term,
    yz_rs_term,
)
from liq.features.volatility.estimators.windowed import (
    NAN,
    trailing_mean,
    yang_zhang_var,
    yz_k_weight,
)
from liq.validation.volatility import load_ohlc_fixture, windowed_reference

# Impl plan §10.1 windowed tolerance.
TOL_WINDOWED = 1e-10

_WindowedName = Literal[
    "ctc",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "gk_yang_zhang",
    "yang_zhang",
]


def _log_bars():
    fixture = load_ohlc_fixture()
    c_prev = math.log(float(fixture["prev_close"]))
    for bar in fixture["bars"]:
        o = math.log(float(bar["open"]))
        h = math.log(float(bar["high"]))
        lo = math.log(float(bar["low"]))
        c = math.log(float(bar["close"]))
        yield o, h, lo, c, c_prev
        c_prev = c


def _per_bar_series(name: str) -> list[float]:
    bars = list(_log_bars())
    if name == "ctc":
        return [ctc_var_contribution(c, c_prev) for _o, _h, _lo, c, c_prev in bars]
    if name == "parkinson":
        return [parkinson_var_contribution(h, lo) for _o, h, lo, _c, _cp in bars]
    if name == "garman_klass":
        return [garman_klass_var_contribution(o, h, lo, c) for o, h, lo, c, _cp in bars]
    if name == "rogers_satchell":
        return [rogers_satchell_var_contribution(o, h, lo, c) for o, h, lo, c, _cp in bars]
    if name == "gk_yang_zhang":
        return [gk_yang_zhang_var_contribution(o, h, lo, c, c_prev) for o, h, lo, c, c_prev in bars]
    if name == "yz_overnight_term":
        return [yz_overnight_term(o, c_prev) for o, _h, _lo, _c, c_prev in bars]
    if name == "yz_open_close_term":
        return [yz_open_close_term(o, c) for o, _h, _lo, c, _cp in bars]
    if name == "yz_rs_term":
        return [yz_rs_term(o, h, lo, c) for o, h, lo, c, _cp in bars]
    raise ValueError(f"unknown series: {name}")


def _compare_with_reference(name: _WindowedName, window: int, ours: list[float]) -> None:
    ref = windowed_reference(name, window)
    assert len(ours) == len(ref)
    for i, (got, want) in enumerate(zip(ours, ref, strict=True)):
        if want is None:
            assert math.isnan(got), f"index {i}: expected NaN, got {got}"
        else:
            assert not math.isnan(got), f"index {i}: expected {want}, got NaN"
            assert abs(got - want) < TOL_WINDOWED, (
                f"index {i}: {got} vs {want}, diff {abs(got - want):.3e}"
            )


class TestTrailingMeanContract:
    def test_emits_nan_until_min_periods(self) -> None:
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = trailing_mean(series, window=3, min_periods=3)
        assert math.isnan(out[0]) and math.isnan(out[1])
        assert out[2] == pytest.approx(2.0)
        assert out[3] == pytest.approx(3.0)
        assert out[4] == pytest.approx(4.0)

    def test_min_periods_less_than_window(self) -> None:
        series = [1.0, 2.0, 3.0]
        out = trailing_mean(series, window=5, min_periods=2)
        assert math.isnan(out[0])
        assert out[1] == pytest.approx(1.5)
        assert out[2] == pytest.approx(2.0)

    def test_nan_inputs_are_skipped(self) -> None:
        # With window=3, min_periods=2, NaN samples don't count toward
        # the min_periods gate — so the gate doesn't unlock until
        # there are 2 non-NaN samples in the trailing window.
        series = [NAN, 2.0, 4.0, 6.0]
        out = trailing_mean(series, window=3, min_periods=2)
        assert math.isnan(out[0])  # window=[NaN]; 0 non-NaN
        assert math.isnan(out[1])  # window=[NaN, 2.0]; 1 non-NaN < 2
        assert out[2] == pytest.approx(3.0)  # window=[NaN, 2.0, 4.0]; mean(2,4)=3
        assert out[3] == pytest.approx(4.0)  # window=[2.0, 4.0, 6.0]; mean=4

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            trailing_mean([1.0], window=0, min_periods=1)
        with pytest.raises(ValueError, match="cannot exceed window"):
            trailing_mean([1.0], window=2, min_periods=3)


class TestYzKWeightMatchesFormula:
    @pytest.mark.parametrize("window", [2, 3, 10, 21, 63, 252])
    def test_closed_form(self, window: int) -> None:
        expected = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        assert yz_k_weight(window) == pytest.approx(expected, abs=1e-15)

    def test_window_less_than_two_raises(self) -> None:
        with pytest.raises(ValueError, match="window >= 2"):
            yz_k_weight(1)


class TestCtcWindowedGolden:
    @pytest.mark.parametrize("window", [5, 10, 15])
    def test_matches_decimal_reference(self, window: int) -> None:
        ours = trailing_mean(_per_bar_series("ctc"), window=window, min_periods=window)
        _compare_with_reference("ctc", window, ours)


class TestParkinsonWindowedGolden:
    @pytest.mark.parametrize("window", [5, 10, 15])
    def test_matches_decimal_reference(self, window: int) -> None:
        ours = trailing_mean(_per_bar_series("parkinson"), window=window, min_periods=window)
        _compare_with_reference("parkinson", window, ours)


class TestGarmanKlassWindowedGolden:
    @pytest.mark.parametrize("window", [5, 10, 15])
    def test_matches_decimal_reference(self, window: int) -> None:
        ours = trailing_mean(_per_bar_series("garman_klass"), window=window, min_periods=window)
        _compare_with_reference("garman_klass", window, ours)


class TestRogersSatchellWindowedGolden:
    @pytest.mark.parametrize("window", [5, 10, 15])
    def test_matches_decimal_reference(self, window: int) -> None:
        ours = trailing_mean(_per_bar_series("rogers_satchell"), window=window, min_periods=window)
        _compare_with_reference("rogers_satchell", window, ours)


class TestGkYzWindowedGolden:
    @pytest.mark.parametrize("window", [5, 10, 15])
    def test_matches_decimal_reference(self, window: int) -> None:
        ours = trailing_mean(_per_bar_series("gk_yang_zhang"), window=window, min_periods=window)
        _compare_with_reference("gk_yang_zhang", window, ours)


class TestYangZhangWindowedGolden:
    @pytest.mark.parametrize("window", [5, 10, 15])
    def test_matches_decimal_reference(self, window: int) -> None:
        overnight = _per_bar_series("yz_overnight_term")
        open_close = _per_bar_series("yz_open_close_term")
        rs = _per_bar_series("yz_rs_term")
        ours = yang_zhang_var(
            overnight_terms=overnight,
            open_close_terms=open_close,
            rs_terms=rs,
            window=window,
            min_periods=window,
        )
        _compare_with_reference("yang_zhang", window, ours)
