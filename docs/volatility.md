# Canonical risk-variance estimator (`liq.features.volatility`)

> **Status.** Contract types, exception hierarchy, public entry-point,
> and the per-bar / windowed formula registry are in place;
> `estimate_variance` dispatches on the spec and returns a
> `VolEstimate` populated with `var_per_bar`, `vol_per_bar`, optional
> annualized series, and the `cont` / `overnight_gap` components when
> the spec targets close-to-close risk variance. PIT enforcement is
> active. Data-quality / fallback / structured logging hookups land in
> the hardening iteration. See
> [research plan](../../liq-docs/plans/liq-features-canonical-risk-variance-plan.md)
> §3.4–3.6 + `[APPENDIX_FORMULAS]` for the load-bearing detail and the
> [implementation plan](../../liq-docs/plans/liq-features-canonical-risk-variance-impl-plan.md)
> for the per-iteration ledger.

## Why this exists

A single point-in-time trailing risk-variance estimator that replaces
the ad-hoc range / ATR / rolling-vol paths previously scattered across
`liq-features`, `liq-risk`, and `liq-sim`. The canonical scalar is
**close-to-close risk variance** (`risk_var_t`), stored under the
generic name `var_t`; `vol_t = sqrt(var_t)` is a derived convenience.

See research plan `[PHASE0_CONTRACT]` for the canonical-scalar choice
(Option B), `[CANONICALIZATION]` for the single-path commitment, and
`[DECISIONS]` for the production defaults.

## Public surface

```python
from liq.features.volatility import (
    VolEstimatorSpec, VolCalendarPolicy, TimingPolicy, VolQualityPolicy,
    RVSpec, VolEstimate, VolComponent,
    VolFeatureError, VolSpecError, VolDataQualityError,
    VolPITViolationError, VolUnavailableError,
    estimate_variance,
)
```

| Section | Status | Anchor |
| --- | --- | --- |
| Frozen spec dataclasses (`VolEstimatorSpec`, `VolCalendarPolicy`, `TimingPolicy`, `VolQualityPolicy`, `RVSpec`) | ✅ defined | research plan §3.4, `[PHASE0_CONTRACT]` |
| Output object (`VolEstimate`, `VolComponent`) | ✅ defined | research plan §3.6 |
| Exception hierarchy | ✅ defined | impl plan foundation |
| `estimate_variance(bars, spec, *, asof=None) -> VolEstimate` | ✅ implemented (CtC, Parkinson, GK, RS, GK-YZ, YZ; PIT enforcement; spec validation; annualization) | research plan §3.4 |
| Formula registry (`ctc`, `parkinson`, `garman_klass`, `rogers_satchell`, `gk_yang_zhang`, `yang_zhang`) | ✅ implemented + golden-tested at 1e-12 (per-bar) / 1e-10 (windowed) | research plan `[APPENDIX_FORMULAS]` |
| RV / BPV / minute-mode estimators | landing with the RV-noise gate | research plan §5.3 + `[APPENDIX_FORMULAS]` |
| Data-quality enforcement + fallback chain | landing with `[DESIGN_DATA_QUALITY]` | research plan §4.2 + `[DESIGN_DATA_QUALITY]` |
| Variance decomposition (`cont`, `overnight_gap`, `jump`, `intraday_range`) | YZ + close-to-close composes `cont` + `overnight_gap`; jump components land with minute mode | research plan `[DESIGN_DECOMPOSITION]` |
| Refinement (`6A` arbitration, `6B` calibration) | conditional, gated | research plan `[PHASE25_REFINEMENT]` |
| Mode availability (minute-enabled / degraded-fallback) | landing with data-quality | research plan §3.6 |

## Formula registry

The per-bar variance contributions in
`liq.features.volatility.estimators.formulas` implement research plan
`[APPENDIX_FORMULAS]` verbatim. Every function takes **log prices** —
the caller (in `estimate.py`) takes `log(price)` before dispatching.
This keeps the closed-form expressions exactly as written and makes
scale invariance free (every formula is a polynomial in differences of
logs).

```text
Notation: o, h, l, c = log(open), log(high), log(low), log(close)
          c_prev    = log(prev close)
          u = h - o,  d = l - o,  k = c - o
```

| Function | Inputs | Estimates | Per-bar contribution |
| --- | --- | --- | --- |
| `ctc_var_contribution(c, c_prev)` | C, C_prev | close-to-close | `(c - c_prev)^2` |
| `parkinson_var_contribution(h, low)` | H, L | intraday | `(h - l)^2 / (4 ln 2)` |
| `garman_klass_var_contribution(o, h, low, c)` | O, H, L, C | intraday | `0.5 (h-l)^2 - (2 ln 2 - 1)(c-o)^2` |
| `rogers_satchell_var_contribution(o, h, low, c)` | O, H, L, C | intraday, drift-aware | `u(u-k) + d(d-k)` |
| `gk_yang_zhang_var_contribution(o, h, low, c, c_prev)` | O, H, L, C, C_prev | close-to-close | `(o - c_prev)^2 + garman_klass` |
| `yz_overnight_term(o, c_prev)` | O, C_prev | YZ per-bar component | `(o - c_prev)^2` |
| `yz_open_close_term(o, c)` | O, C | YZ per-bar component | `(c - o)^2` |
| `yz_rs_term(o, h, low, c)` | O, H, L, C | YZ per-bar component (≡ Rogers-Satchell) | `u(u-k) + d(d-k)` |

The windowed layer in `estimators.windowed` exposes:

- `trailing_mean(values, *, window, min_periods)` — NaN-skipping
  rolling mean for the simple estimators.
- `yz_k_weight(window)` — the canonical
  `k_yz = 0.34 / (1.34 + (n+1)/(n-1))` weighting.
- `yang_zhang_var(overnight_terms, open_close_terms, rs_terms, *, window, min_periods)`
  — composes `mean(overnight) + k_yz * mean(open_close) + (1 - k_yz) * mean(rs)`.

Reference values for the registry are computed in
`liq-validation`'s `golden_fixtures` at 50-digit `Decimal` precision
and stored as JSON in `liq-validation/tests/fixtures/volatility/`. The
contract tests in `liq-features/tests/test_volatility_formulas.py` and
`tests/test_volatility_windowed.py` pin every estimator to those
reference values at the tolerances called out in §10.1 of the research
plan. The operator can spot-check the registry after a formula edit
via `uv run python scripts/smoke_formula_registry.py`; the script
emits a JSON report to stdout and exits non-zero on any tolerance
breach.

## Minute-mode RV / BPV / JV + RV-noise gate

Implementations of research plan §5.3 ship in `liq.features.volatility.rv`.

| Function | Purpose | Formula |
| --- | --- | --- |
| `compute_rv(returns)` | Realized variance — sum of squared log-returns. | `Σ r_i²` |
| `compute_bpv(returns)` | Bipower variation — jump-robust continuous-variance proxy. | `(π/2) · Σ_{i≥1} \|r_{i-1}\| · \|r_i\|` |
| `compute_jv(returns)` | Jump variation. | `max(RV - BPV, 0)` |
| `rv_noise_gate(rv_by_interval, price_movement)` | §5.3 hard gate. | See below. |

The Barndorff-Nielsen / Shephard decomposition `RV = BPV + JV` holds
*exactly* whenever `JV > 0`; the `max` clip in `compute_jv` keeps the
jump component non-negative in finite samples. Tested in
`tests/test_volatility_rv.py` (per-bar) and
`liq-experiments/tests/vol/test_bpv_jv_jump_isolation.py`
(integration — single-jump injection isolates to JV while BPV stays
near the continuous floor).

### RV-noise gate

The gate fires when both of the following hold:

1. `RV_1m ≥ ratio_threshold · RV_5m` AND `RV_1m ≥ ratio_threshold · RV_15m` —
   the finer-sampling-inflation pattern (default `ratio_threshold = 2.0`).
2. `price_movement² < unexplained_threshold · RV_5m` — the directional
   price movement does not explain the inflated 1m RV (default
   `unexplained_threshold = 0.5`). `RV_5m` is the low-noise reference for
   the session's true integrated variance.

When the gate fires, the caller falls back to `RV_5m` (or a realized
kernel) and adds `NOISY_RV_TARGET` to the bar's quality flags.

A canonical worked example lives in
`liq-experiments/scripts/eval_simulation_minute.py`; the acceptance
criterion is

```bash
uv run python scripts/eval_simulation_minute.py --noise micro --output json \
    | jq '.gate_fired'
```

returning `true`. The companion `--noise none` run leaves the gate
down.

## Estimator dispersion + §5.4 quality flags

The decomposition layer in `liq.features.volatility.decomposition`
turns a `{estimator_name: variance}` dict into:

- A scalar `estimator_dispersion = stdev_pop(estimates.values())` — the
  §5.4 "disagreement is information" signal.
- A tuple of derived quality flags following the §5.4 table:

  | Pattern | Flag |
  | --- | --- |
  | Parkinson low + CtC high (`ctc ≥ 4× parkinson`) | `GAP_DOMINATED_VOL` |
  | RS high + CtC low (`rs ≥ 4× ctc`) | `INTRADAY_RANGE_DOMINATED_VOL` |
  | `estimator_dispersion > VolQualityPolicy.estimator_dispersion_threshold` | `HIGH_ESTIMATOR_DISAGREEMENT` |
  | CtC and the range-mean differ by ≥ 4× in either direction | `CTC_DISAGREES_WITH_RANGE` |

Missing estimator keys are tolerated — a degraded-fallback caller may
pass only `{ctc, parkinson}` without crashing the derivation. The
patterns are computed independently and may co-occur.

## Spec is authoritative

Every field that affects the emitted number lives on `VolEstimatorSpec`;
the prose in this document and in the research plan *describes* the
fields, it does not add new ones. Two runs with the same
`VolEstimatorSpec` on the same bars must produce identical output —
that's the reproducibility contract the Gate-1 golden tests enforce.

## Legacy helpers

The previous single-file `volatility.py` module (35 lines:
`yang_zhang` and `garman_klass` returning a scalar) lives at
`volatility/_legacy.py` for the duration of the transition; both
functions are re-exported from the package so existing callers keep
working:

```python
from liq.features.volatility import yang_zhang, garman_klass
```

These are retired once downstream consumers migrate to
`estimate_variance` (research plan §13 + impl plan ATR bridge writeup).

## Calendar consumption

The volatility decomposition uses `liq.data.calendar` for closed-market
gap classification — `closed_hours_between(c_prev, o)` and
`classify_gap(c_prev, o)`. See [`liq-data/docs/calendar.md`](../../liq-data/docs/calendar.md)
for the helpers; their outputs feed `gap_class_t`, `closed_hours_t`,
and the per-closed-hour diagnostic on `VolEstimate.components` per
research plan §3.1a.
