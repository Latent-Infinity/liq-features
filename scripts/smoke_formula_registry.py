"""Smoke check for the canonical variance formula registry.

Runs the registry over the 16-bar golden OHLC fixture shipped from
``liq-validation``, compares per-bar and windowed outputs against the
arbitrary-precision Decimal reference, and emits a structured JSON
report to **stdout**. Logs (warnings, info) go to **stderr** per the
workspace CLI convention.

Acceptance contract (impl plan):

- Every per-bar abs-error <= 1e-12 against the Decimal reference.
- Every windowed abs-error <= 1e-10 (window=15 over 16 bars => last
  two windowed values per estimator).

The script exits 0 when all gates pass and 1 when any abs-error
exceeds its gate. Useful for the operator to spot-check the registry
after a formula edit; not part of the verify-gate.

Run: ``uv run python scripts/smoke_formula_registry.py``
"""

from __future__ import annotations

import json
import math
import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

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
)
from liq.validation.volatility import (
    load_ohlc_fixture,
    per_bar_reference,
    windowed_reference,
)

PER_BAR_GATE = 1e-12
WINDOWED_GATE = 1e-10
WINDOW = 15
MIN_PERIODS = 15

_PerBarName = Literal[
    "ctc",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "gk_yang_zhang",
    "yz_overnight_term",
    "yz_open_close_term",
    "yz_rs_term",
]
_WindowedName = Literal[
    "ctc",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "gk_yang_zhang",
    "yang_zhang",
]


def _log_bars() -> list[tuple[float, float, float, float, float | None]]:
    fixture = load_ohlc_fixture()
    c_prev: float | None = math.log(float(fixture["prev_close"]))
    rows: list[tuple[float, float, float, float, float | None]] = []
    for bar in fixture["bars"]:
        o = math.log(float(bar["open"]))
        h = math.log(float(bar["high"]))
        lo = math.log(float(bar["low"]))
        c = math.log(float(bar["close"]))
        rows.append((o, h, lo, c, c_prev))
        c_prev = c
    return rows


def _max_abs_diff(actual: Iterable[float], reference: Iterable[float | None]) -> float:
    worst = 0.0
    for got, want in zip(actual, reference, strict=True):
        if want is None:
            if not math.isnan(got):
                worst = max(worst, float("inf"))
            continue
        if math.isnan(got):
            worst = max(worst, float("inf"))
            continue
        worst = max(worst, abs(got - want))
    return worst


def _per_bar_actual(
    name: _PerBarName, bars: list[tuple[float, float, float, float, float | None]]
) -> list[float]:
    out: list[float] = []
    for o, h, lo, c, c_prev in bars:
        cp = c_prev
        if name == "ctc":
            out.append(NAN if cp is None else ctc_var_contribution(c, cp))
        elif name == "parkinson":
            out.append(parkinson_var_contribution(h, lo))
        elif name == "garman_klass":
            out.append(garman_klass_var_contribution(o, h, lo, c))
        elif name == "rogers_satchell":
            out.append(rogers_satchell_var_contribution(o, h, lo, c))
        elif name == "gk_yang_zhang":
            out.append(NAN if cp is None else gk_yang_zhang_var_contribution(o, h, lo, c, cp))
        elif name == "yz_overnight_term":
            out.append(NAN if cp is None else yz_overnight_term(o, cp))
        elif name == "yz_open_close_term":
            out.append(yz_open_close_term(o, c))
        elif name == "yz_rs_term":
            out.append(yz_rs_term(o, h, lo, c))
        else:
            raise ValueError(f"unknown estimator name {name!r}")
    return out


def _windowed_actual(
    name: _WindowedName, bars: list[tuple[float, float, float, float, float | None]]
) -> list[float]:
    if name == "yang_zhang":
        ov = _per_bar_actual("yz_overnight_term", bars)
        oc = _per_bar_actual("yz_open_close_term", bars)
        rs = _per_bar_actual("yz_rs_term", bars)
        return yang_zhang_var(
            overnight_terms=ov,
            open_close_terms=oc,
            rs_terms=rs,
            window=WINDOW,
            min_periods=MIN_PERIODS,
        )
    per_bar = _per_bar_actual(name, bars)
    return trailing_mean(per_bar, window=WINDOW, min_periods=MIN_PERIODS)


def main() -> int:
    bars = _log_bars()
    estimators_per_bar: tuple[_PerBarName, ...] = (
        "ctc",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "gk_yang_zhang",
        "yz_overnight_term",
        "yz_open_close_term",
        "yz_rs_term",
    )
    estimators_windowed: tuple[_WindowedName, ...] = (
        "ctc",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "gk_yang_zhang",
        "yang_zhang",
    )
    per_bar_results = {}
    failures: list[str] = []
    for name in estimators_per_bar:
        actual = _per_bar_actual(name, bars)
        reference = per_bar_reference(name)
        worst = _max_abs_diff(actual, reference)
        per_bar_results[name] = {
            "max_abs_diff": worst,
            "gate": PER_BAR_GATE,
            "pass": worst <= PER_BAR_GATE,
        }
        if worst > PER_BAR_GATE:
            failures.append(f"per_bar:{name}:{worst:.3e}")

    windowed_results = {}
    for name in estimators_windowed:
        actual = _windowed_actual(name, bars)
        reference = windowed_reference(name, WINDOW)
        worst = _max_abs_diff(actual, reference)
        windowed_results[name] = {
            "window": WINDOW,
            "max_abs_diff": worst,
            "gate": WINDOWED_GATE,
            "pass": worst <= WINDOWED_GATE,
        }
        if worst > WINDOWED_GATE:
            failures.append(f"windowed:{name}:{worst:.3e}")

    report = {
        "fixture": "ohlc_16bar",
        "per_bar": per_bar_results,
        "windowed": windowed_results,
        "all_passed": len(failures) == 0,
        "failures": failures,
    }
    json.dump(report, sys.stdout, indent=2, default=float)
    sys.stdout.write("\n")
    if failures:
        print(f"FAIL: {len(failures)} gates breached: {failures}", file=sys.stderr)
        return 1
    print(
        f"OK: per-bar <= {PER_BAR_GATE:.0e}, windowed <= {WINDOWED_GATE:.0e}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
