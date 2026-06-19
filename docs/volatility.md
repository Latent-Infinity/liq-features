# Canonical risk-variance estimator (`liq.features.volatility`)

> **Status.** Consolidated for production. The full surface is live:
> contract types, exception hierarchy, formula registry (CtC,
> Parkinson, Garman-Klass, Rogers-Satchell, GK-YZ, Yang-Zhang),
> windowed estimators, PIT enforcement, data-quality + fallback chain,
> structured logging (research plan §1.9), minute-mode RV / BPV / JV
> + RV-noise gate (§5.3), estimator dispersion + derived quality flags
> (§5.4), and the `volatility_signature` VolComponent. The canonical
> estimator is **`gk_yang_zhang`** (decisions registry §5 — MCS_B
> winner under Option B; selection evidence is recorded in the
> decisions registry). See
> [research plan](../../liq-docs/plans/liq-features-canonical-risk-variance-plan.md)
> §3.4–3.6 + `[APPENDIX_FORMULAS]` for the load-bearing detail and the
> [decisions registry](../../liq-docs/plans/liq-features-canonical-risk-variance-decisions.md)
> for every frozen decision.

## Why this exists

A single point-in-time trailing risk-variance estimator that replaces
the ad-hoc range / ATR / rolling-vol paths previously scattered across
the feature, sizing, and simulation surfaces. The canonical scalar is
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
| RV / BPV / minute-mode estimators + RV-noise gate | ✅ implemented in `liq.features.volatility.rv` | research plan §5.3 + `[APPENDIX_FORMULAS]` |
| Data-quality enforcement + fallback chain | ✅ implemented in `liq.features.volatility.quality` + `estimators.fallback` | research plan §4.2 + `[DESIGN_DATA_QUALITY]` |
| Variance decomposition (`cont`, `overnight_gap`, `jump`, `intraday_range`, `volatility_signature`) | ✅ implemented in `liq.features.volatility.decomposition` and `estimate.py` | research plan `[DESIGN_DECOMPOSITION]` + §5.3 |
| Estimator dispersion + §5.4 quality-flag derivation | ✅ implemented | research plan §5.4 |
| Refinement (`6A` arbitration, `6B` calibration) | SKIPPED — both triggers `trigger-not-fired` (decisions registry §6) | research plan `[PHASE25_REFINEMENT]` |
| Mode availability (minute-enabled / degraded-fallback) | ✅ tracked per-component via `VolComponent.source` (see table below) | research plan §3.6 |

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

Reference values for the registry are computed at 50-digit `Decimal`
precision and shipped as JSON golden fixtures consumed by this repo's
contract tests. The contract tests in
`tests/test_volatility_formulas.py` and
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
jump component non-negative in finite samples. Per-bar behavior is
covered by `tests/test_volatility_rv.py`; integration-level
single-jump injection (jump isolates to JV while BPV stays near the
continuous floor) is owned by the downstream experiments harness.

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

The gate is end-to-end exercised by the downstream simulation harness
under a microstructure-noise injection; its acceptance criterion is
that `gate_fired` is `true` on the noisy run and `false` on the clean
run.

returning `true`. The companion `--noise none` run leaves the gate
down.

### `volatility_signature` VolComponent

When the caller passes `intra_bar_returns={bar_index: [log_returns]}`
to `estimate_variance(...)`, a `volatility_signature` entry is added to
`VolEstimate.components` carrying one realized-variance value per bar:

- Bars without minute data → `NaN`.
- Bars whose 1m sampling RV is consistent with the 5m / 15m readings
  → 1m RV is used.
- Bars whose 1m sampling RV trips the §5.3 noise gate → 5m RV is used,
  and the emitter logs a `rv_noise_gate_fired` event for that bar.

The component's `source` is `"minute_rv"` (vs the daily-OHLC sources
on the `cont` / `overnight_gap` components); its `unit` is
`per_bar_variance`. Consumers driving the §5.3 volatility-signature
plot or downstream regime/sizing code can pull the per-bar value
without re-computing RV themselves.

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

## Production / degraded-fallback mode availability

Per research plan §3.6, every `VolComponent` carries a `source` field
identifying the data path that produced it. The table below summarizes
which components are available in each mode; a request for an
`unavailable` component raises :class:`VolUnavailableError`.

| Component | Production (minute-enabled) | Degraded-fallback (daily OHLC only) | `VolComponent.source` |
| --- | --- | --- | --- |
| `cont` (continuous intraday variance) | available | available (window-aggregated range estimator) | `derived` |
| `overnight_gap` | available | available | `daily_ohlc` |
| `jump` | available (`RV - BPV`) | unavailable | `minute_rv` |
| `intraday_range` | available | available | `daily_ohlc` |
| `volatility_signature` (per-bar RV) | available (passes `intra_bar_returns`) | unavailable | `minute_rv` |
| `gap_var_per_closed_hour` diagnostic | available | available | `derived` |

When degraded-fallback fires, structured-log events
`estimator_fallback_applied` and (if a flag is set)
`quality_flag_set` accompany the emitted estimate so consumers can
detect the mode shift without inspecting the source field directly.

## Full quality-flag vocabulary

Per research plan §5.4 + `[DESIGN_DATA_QUALITY]`, the canonical flag
set on `VolEstimate.quality_flags` (and per-bar via the
`quality_flag_set` event) is:

| Flag | When set | Source |
| --- | --- | --- |
| `HIGH_LOW_OUTLIER` | A bar's high/low fails the PIT outlier check against past-available neighbors | `quality.py` |
| `MISSING_OPEN` | A bar's open is missing; YZ / RS fall back to Parkinson or CtC | `quality.py` |
| `MISSING_PREV_CLOSE` | The prior close is missing; close-to-close and YZ overnight terms fall back | `quality.py` |
| `HALTED_OR_ZERO_VOLUME` | Bar flagged as halted / zero-volume; treated per `VolQualityPolicy.halt_policy` | `quality.py` |
| `PARTIAL_SESSION` | Calendar-aware annualization or exclusion engaged for a partial trading day | `calendar.py` |
| `GAP_DOMINATED_VOL` | Parkinson low + CtC high (`ctc ≥ 4× parkinson`) — variance is dominated by the overnight gap | `decomposition.py` |
| `INTRADAY_RANGE_DOMINATED_VOL` | RS high + CtC low (`rs ≥ 4× ctc`) — variance is dominated by intraday range | `decomposition.py` |
| `HIGH_ESTIMATOR_DISAGREEMENT` | `estimator_dispersion > VolQualityPolicy.estimator_dispersion_threshold` | `decomposition.py` |
| `CTC_DISAGREES_WITH_RANGE` | CtC and the range-mean differ by ≥ 4× in either direction | `decomposition.py` |
| `NOISY_RV_TARGET` | The §5.3 RV-noise gate fired — `RV_1m` was rejected in favor of `RV_5m` / realized kernel | `rv.py` + `estimate.py` |

Flags compose: a single bar may carry multiple flags at once. The
`quality_flag_set` structured-log event is debounced per
`(estimate_call, bar_index)` and carries the union of flags set on
that bar.

## Calendar consumption

The volatility decomposition uses `liq.data.calendar` for closed-market
gap classification — `closed_hours_between(c_prev, o)` and
`classify_gap(c_prev, o)`. See [`liq-data/docs/calendar.md`](../../liq-data/docs/calendar.md)
for the helpers; their outputs feed `gap_class_t`, `closed_hours_t`,
and the per-closed-hour diagnostic on `VolEstimate.components` per
research plan §3.1a.

### Worked example — weeknight gap vs weekend gap

A close on Friday at 16:00 ET → open on Monday at 09:30 ET spans
roughly 65 closed-market hours (research plan §3.1a). A close on
Tuesday at 16:00 ET → open on Wednesday at 09:30 ET spans roughly
17.5 closed-market hours. Both gaps contribute to `risk_var_t` at
their raw magnitudes (`VolCalendarPolicy.overnight_basis = "separate"`,
the Phase-0 default). The `gap_class_t` field labels each gap as
`weeknight` / `weekend` / `holiday`; the `gap_var_per_closed_hour_t`
diagnostic divides the gap's squared log-return by `closed_hours_t`
so consumers can compare gap magnitudes per unit of closed time
without rescaling the canonical scalar.

## Canonical `risk_var_t` as the sizing-input replacement

The canonical `risk_var_t` (and its derived `vol_t = sqrt(var_t)`) is the
sizing-input replacement for legacy ATR-based scalars (`ATR_t`,
`natr_t = ATR_t / close_t`). Head-to-head non-inferiority evidence on the
Gate 4 panel is recorded in this repo's volatility artifacts; consumers
that previously read ATR-derived inputs migrate to `estimate_variance(...)`
without rescaling the canonical scalar. The retirement of any specific
downstream ATR consumer is owned by that consumer's own migration plan.
