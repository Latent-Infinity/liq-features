# Volatility forecast features (`liq.features.vol_forecast`)

The forecast / regime feature layer that consumes the canonical
`risk_var_t` measurement and exposes frozen contracts, target builders,
multiscale features, and regime labels. Every export below is shipped
under a `feature_dictionary_id` so downstream consumers can detect
contract drift, and every load-bearing field carries the timestamps the
leakage CI gate reads.

> **Discovery.** Each surface in this doc is testable from the test files
> alongside the source modules. Decisions that pinned a parameter or
> picked a label name are recorded in the workspace decisions registry.

## Forecast contracts

The feature-row contract is a frozen dataclass + a versioned feature
dictionary:

- `VolForecastFeatures` — frozen dataclass with the load-bearing
  timestamp fields (`availability_ts`, `valid_from`) and the regime /
  multiscale field set.
- `ForecastTarget` — frozen target row consumed by both the forecast
  evaluation harness and the sizing pipeline.
- `feature_dictionary_id` — version sigil + SHA256 fingerprint over the
  per-field `(unit, transform, annualization_status, additive,
  availability_rule, null_policy)` tuple. Any change to the dictionary
  changes the id.
- `LOAD_BEARING_TIMESTAMP_FIELDS` — the canonical tuple
  (`availability_ts`, `valid_from`) the leakage assertions read by
  attribute. Renaming or dropping a load-bearing field is an explicit
  break.

Three reason-code stages mark every fallback / data-quality event:
`feature`, `forecast`, and `sizing`. The reason vocabulary is shared
across the workstream and consumed by the leakage gate.

## Targets

Two target builders:

- `build_target_rv_total(bars, target_session, *, symbol)` — realized
  variance target with row-level provenance, CA-adjacency reason coding
  (`CA_ADJACENT`), and the `is_path_dependent=False` invariant.
- `build_intraday_reversal_target(*, signal_id, symbol, signal_ts,
  fill_ts, horizon, next_close_ts)` — FIXED-HORIZON intraday reversal
  target with `target_end_ts = min(fill_ts + horizon, next_close_ts)`
  and `is_path_dependent=False` invariantly. Reasons emit on
  fill-before-signal or non-positive horizon.

Both target rows expose `availability_ts` + `valid_from` so a downstream
leakage assertion can read them by attribute without importing the
feature module.

## Multiscale features

`build_multiscale_features` returns the daily / weekly / monthly
realized-variance roll-ups + the linear slope across windows + their
log-space equivalents. Every row carries coverage metadata
(`has_full_window`, `observed_count`) so the harness can drop
short-history rows without re-computing.

## Regime features

Three families:

- `compute_semivariance(returns, window)` — downside vs upside semivariance
  decomposition with the additivity property
  `downside_rv + upside_rv == bar_rv` (modulo a zero floor) and explicit
  coverage metadata.
- `compute_asymmetry_regression(returns)` — rolling asymmetry regression
  (coefficients `b0`, `b1`, `b2`, `b3`) on lagged returns; on a
  symmetric series `b3 ≈ 0`.
- `derive_gap_jump_labels(estimate, *, threshold_gap, threshold_range,
  threshold_jump)` — canonical labels:
  `GAP_DOMINATED_VOL`,
  `INTRADAY_RANGE_DOMINATED_VOL`,
  `JUMP_DAY`.
  Thresholds are required keyword args (no hardcoded defaults).
  Frozen registry values: `threshold_gap=0.60`, `threshold_range=0.40`,
  `threshold_jump=0.10` (set via grid search with a held-out OOS verdict
  on the gap_jump augmentation).

Multi-label resolution uses `resolve_multi_label(labels, scores)` which
keeps every label tied for the maximum multiplier. Conservatism labels
combine by **MAX multiplier**, not product (products compound leverage
cuts unpredictably).

## Serving clocks

Three invariants are asserted at every layer:

1. Feature → forecast: every consumed feature has
   `availability_ts <= forecast_origin_ts`.
2. Forecast → sizing: every emitted forecast has
   `forecast_available_ts <= sizing_decision_ts`.
3. Sizing → order: every emitted size has
   `size_vol.valid_from <= order_decision_ts < size_vol.valid_until`.

The pure assertion helpers shipped here are
`assert_feature_forecast_clock`, `assert_forecast_target_clock`,
`assert_forecast_size_clock`, `assert_sizing_order_clock`, and
`assert_no_straddle`. The leakage gate ships in the testing surface;
these helpers are the local versions used inside the feature library.

## Universes

`compute_universe_membership(symbol, asof, eligibility)` returns one of
three buckets: `common-eligible`, `production-coverage`,
`limited-history`. Membership is recorded under a versioned
`UNIVERSE_DEFINITION_ID` so downstream reporting never silently mixes
universes.

The three universes:

- `common-eligible` — baseline-safe intersection used for production
  comparability metrics; no fallbacks.
- `production-coverage` — at least one approved forecaster eligible;
  fallbacks reason-coded.
- `limited-history` — deterministic carve-out for symbols below the
  GARCH minimum window; blocks promotion-eligible scoring.

## Example

```python
from datetime import UTC, datetime, timedelta

from liq.features.vol_forecast import (
    GAP_DOMINATED_VOL,
    build_intraday_reversal_target,
    derive_gap_jump_labels,
)

target = build_intraday_reversal_target(
    signal_id="aapl_2024_03_15_a",
    symbol="AAPL",
    signal_ts=datetime(2024, 3, 15, 18, 30, tzinfo=UTC),
    fill_ts=datetime(2024, 3, 15, 18, 31, tzinfo=UTC),
    horizon=timedelta(minutes=60),
    next_close_ts=datetime(2024, 3, 15, 20, 0, tzinfo=UTC),
)
assert target.is_path_dependent is False

labels = derive_gap_jump_labels(
    {
        "overnight_gap_var": 0.006,
        "intraday_range_var": 0.003,
        "jump_var": 0.001,
        "total_var": 0.01,
    },
    threshold_gap=0.60,
    threshold_range=0.40,
    threshold_jump=0.10,
)
assert GAP_DOMINATED_VOL in labels
```
