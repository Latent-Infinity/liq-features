# RV-noise gate — liq-features decisions

## Decisions

1. **RV-noise gate uses `price_movement² < unexplained_threshold · RV_5m`,
   not an absolute price-movement floor.** The early draft used
   `abs(price_movement) < 0.005` as the "no concurrent shock" criterion,
   but realistic intraday sessions at sigma ≈ 0.013 (the daily-volatility
   default) routinely produce |return| of 1%+. An absolute floor would
   trip the gate on every quiet session whose price merely drifted.
   The ratio-based formulation makes the gate scale-invariant: it asks
   "does the directional move explain the inflated RV?" by comparing the
   move² to a low-noise reference (`RV_5m`).
2. **`compute_jv = max(RV - BPV, 0)` clipped at zero, not raw.** In
   finite samples (especially low-noise GBM with n = 390) BPV can
   marginally exceed RV due to sampling jitter. Allowing negative
   `jv` would propagate that jitter as "negative variance" to
   downstream consumers; clipping at zero matches the canonical
   Barndorff-Nielsen / Shephard convention and keeps the JV component
   non-negative for sizing / regime logic.
3. **§5.4 patterns use multiplicative ratios, not absolute deltas.**
   GAP_DOMINATED_VOL fires when `ctc ≥ 4× parkinson`, not
   `ctc - parkinson > some_constant`. The multiplicative form is
   scale-invariant (works at any annualized vol level) and matches
   the §5.4 text's "Parkinson low + CtC high" intuition. Ratio = 4
   is conservative — most gap-dominated days hit a much wider gap.
4. **`derive_quality_flags` tolerates missing estimator keys.** A
   degraded-fallback caller emitting only CtC + Parkinson can still
   surface `GAP_DOMINATED_VOL` without crashing the derivation.
   Each pattern checks its required inputs individually, then skips if
   any are absent. The function never raises on partial input.
5. **`estimator_dispersion` is population std (`pstdev`), not sample
   std.** The dict carries *all* the estimators we ran; we're not
   estimating a population mean from a sample of estimators. Using
   `pstdev` keeps the dispersion well-defined at `n = 2` (sample std
   would divide by zero); at `n ≥ 3` the difference is small.
6. **RV-noise gate thresholds are CLI-callable, not hard-coded.**
   `rv_noise_gate(rv_by_interval, *, price_movement, ratio_threshold=2.0,
   unexplained_threshold=0.5)` — every threshold has a default and a
   keyword override. The defaults match research plan §5.3's "materially
   exceeds" intuition; the keyword arguments let the §10.2 gate-parameter
   ablation sweep these without code edits.

## Deviations

- **No `volatility_signature` VolComponent emitted yet.** The impl plan
  asks for "volatility-signature plot data emitted as a `VolComponent`
  named `volatility_signature` when minute data are available". The
  primitives (compute_rv at 1m / 5m / 15m) are in `rv.py`; wiring them
  into `VolEstimate.components` waits for the minute-data caller path
  in `estimate.py`, which lands alongside the §3.6 mode-availability
  table (minute-enabled production vs degraded fallback). The RV-noise gate
  acceptance does not require this VolComponent surface, and the
  decision keeps the change tightly scoped.
- **`derive_quality_flags` is not yet called from `estimate.py`.** The
  pattern detectors are implemented and unit-tested but
  `estimate_variance` does not invoke `derive_quality_flags` against a
  multi-estimator dict because the windowed code path runs ONE
  resolved estimator per call (see quality/fallback work decisions). Multi-estimator
  emission lands when the minute-data path comes online.
