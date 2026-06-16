"""Golden-fixture tests for the per-bar variance contribution registry.

Each formula's per-bar series is computed against the 16-bar OHLC
fixture shipped from ``liq-validation`` and compared to the
arbitrary-precision reference values at tolerance ``1e-12``.

The tests use the LOG-PRICE convention from research plan
``[APPENDIX_FORMULAS]`` — every formula takes ``log(price)``, never
raw prices. The fixture loader exposes ``Decimal`` OHLC values; the
tests convert to ``float`` log-prices and call the registry.
"""

from __future__ import annotations

import math

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
from liq.validation.volatility import load_ohlc_fixture, per_bar_reference

# Impl plan §10.1 deterministic tolerance.
TOL_PER_BAR = 1e-12


def _log_bars():
    """Yield per-bar (o, h, lo, c, c_prev) log-price tuples."""
    fixture = load_ohlc_fixture()
    c_prev = math.log(float(fixture["prev_close"]))
    for bar in fixture["bars"]:
        o = math.log(float(bar["open"]))
        h = math.log(float(bar["high"]))
        lo = math.log(float(bar["low"]))
        c = math.log(float(bar["close"]))
        yield o, h, lo, c, c_prev
        c_prev = c


class TestCtcVarContribution:
    def test_golden_examples(self) -> None:
        reference = per_bar_reference("ctc")
        actual = [ctc_var_contribution(c, c_prev) for _o, _h, _lo, c, c_prev in _log_bars()]
        assert len(actual) == len(reference)
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR


class TestParkinsonVarContribution:
    def test_golden_examples(self) -> None:
        reference = per_bar_reference("parkinson")
        actual = [parkinson_var_contribution(h, lo) for _o, h, lo, _c, _cp in _log_bars()]
        assert len(actual) == len(reference)
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR


class TestGarmanKlassVarContribution:
    def test_golden_examples(self) -> None:
        reference = per_bar_reference("garman_klass")
        actual = [garman_klass_var_contribution(o, h, lo, c) for o, h, lo, c, _cp in _log_bars()]
        assert len(actual) == len(reference)
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR


class TestRogersSatchellVarContribution:
    def test_golden_examples(self) -> None:
        reference = per_bar_reference("rogers_satchell")
        actual = [rogers_satchell_var_contribution(o, h, lo, c) for o, h, lo, c, _cp in _log_bars()]
        assert len(actual) == len(reference)
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR

    def test_handles_drift(self) -> None:
        """Rogers-Satchell's defining property: a pure trending day
        (open at low, close at high, no reversal) produces a finite
        non-negative variance contribution. GBM-with-drift assumptions
        do not break it."""
        # log prices for an open=low=100, close=high=105 day
        o = math.log(100.0)
        lo = math.log(100.0)
        c = math.log(105.0)
        h = math.log(105.0)
        v = rogers_satchell_var_contribution(o, h, lo, c)
        assert v >= 0
        # u = log(1.05), d = 0, k = log(1.05) → u*(u-k) = 0, d*(d-k) = 0
        assert v == pytest.approx(0.0, abs=TOL_PER_BAR)


class TestGkYangZhangVarContribution:
    def test_golden_examples(self) -> None:
        reference = per_bar_reference("gk_yang_zhang")
        actual = [
            gk_yang_zhang_var_contribution(o, h, lo, c, c_prev)
            for o, h, lo, c, c_prev in _log_bars()
        ]
        assert len(actual) == len(reference)
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR

    def test_opening_jump_term_isolated(self) -> None:
        """If H == O == C == L (a doji bar at the open), the GK part
        vanishes and only the opening-jump term remains."""
        c_prev = math.log(99.40)
        o = math.log(100.00)
        h = lo = c = o
        v = gk_yang_zhang_var_contribution(o, h, lo, c, c_prev)
        # Only (o - c_prev)^2 contributes
        expected = (o - c_prev) ** 2
        assert v == pytest.approx(expected, abs=TOL_PER_BAR)


class TestYangZhangPerBarTerms:
    def test_overnight_term_golden(self) -> None:
        reference = per_bar_reference("yz_overnight_term")
        actual = [yz_overnight_term(o, c_prev) for o, _h, _lo, _c, c_prev in _log_bars()]
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR

    def test_open_close_term_golden(self) -> None:
        reference = per_bar_reference("yz_open_close_term")
        actual = [yz_open_close_term(o, c) for o, _h, _lo, c, _cp in _log_bars()]
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR

    def test_rs_term_matches_rogers_satchell(self) -> None:
        reference = per_bar_reference("yz_rs_term")
        actual = [yz_rs_term(o, h, lo, c) for o, h, lo, c, _cp in _log_bars()]
        for got, want in zip(actual, reference, strict=True):
            assert abs(got - want) < TOL_PER_BAR
        # And by construction matches RS exactly.
        rs_actual = [
            rogers_satchell_var_contribution(o, h, lo, c) for o, h, lo, c, _cp in _log_bars()
        ]
        assert actual == rs_actual
