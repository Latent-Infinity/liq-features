# Phase 1 — liq-features decisions

## Decisions

1. **`VolEstimate` / `VolComponent` fields typed as `pl.Series`, not
   `pd.Series`.** Research plan §3.6 wrote the contract with
   `pd.Series` annotations and the impl plan §1.5 listed pandas as a
   "bridge for `VolEstimate.var_per_bar: pd.Series`" in `liq-validation`
   only — never in `liq-features` runtime deps. With Phase 1 actually
   constructing `VolEstimate` instances at runtime, that gap forces a
   choice. Aligned to `polars.Series` because every other LIQ surface
   uses polars; pandas would have been a single-purpose dep just to
   satisfy a docstring shape. Consumers needing pandas use
   `series.to_pandas()` at the boundary.
2. **`liq-validation` added to `liq-features` dev deps + uv.sources.**
   The impl plan asks "the pytest plugin loads into liq-features's
   test session". `pytest11` entry-point auto-discovery only works
   when the package is installed in the test env; declaring
   `liq-validation` as a dev dep + the local editable path in
   `[tool.uv.sources]` is the workspace-clean way to wire it. The
   transitive dep weight (`liq-sim`, `liq-risk`, `liq-signals`,
   `liq-metrics`) lives in dev only — never in the runtime install.
3. **Formula registry takes log prices, not raw OHLC.** Research plan
   `[APPENDIX_FORMULAS]` writes every formula in
   `o = log(open), h = log(high), ...` notation; passing log prices
   keeps the closed-form expressions exactly as written and makes
   scale invariance free (every formula is a polynomial in
   *differences* of logs; multiplying OHLC by `k > 0` adds `log(k)` to
   every term, which cancels). The `_log_columns` helper in
   `estimate.py` does the conversion once per call so the registry
   functions stay arithmetic-pure.
4. **YZ composition decomposes into `cont` and `overnight_gap`
   components for the c2c-risk-variance target.** Research plan §3.1
   asks for `risk_var_t = cont_var_t + overnight_gap_var_t`. With YZ
   as the canonical c2c estimator, the implementation:
   - computes `cont = mean(0) + k*mean(open_close) + (1-k)*mean(rs)` —
     YZ with the overnight term zeroed
   - computes `overnight_gap = mean(overnight_terms)` — trailing mean
     of `(o - c_prev)^2`
   - composes `risk_var_t = cont + overnight_gap` per the canonical
     recipe
   This makes the component decomposition exactly what
   `[DESIGN_DECOMPOSITION]` requires and surfaces `cont` and
   `overnight_gap` on `VolEstimate.components` so downstream
   regime/sizing code can read them directly.
5. **`NaN` (float "not a number") is the sentinel for "not yet emittable",
   not `None`.** Polars Series carry float NaN cleanly; downstream
   `vol_per_bar = sqrt(var_per_bar)` propagates NaN naturally. Using
   `None` would have forced `pl.List[Optional[Float]]` or constant
   `is_null()` checks in consumers. The `windowed.NAN` export is the
   only NaN factory — every call site uses it for consistency.
6. **Phase 0 strict-xfail tests flipped green, the file kept.** Three
   pinned future-contract tests in
   `tests/test_volatility_contract_xfail.py` were the
   `xfail(strict=True)` anchors from Phase 0; satisfying the contract
   converted them to plain passing tests, which `strict=True` correctly
   surfaced as `XPASS(strict)`. Removed the xfail markers and the
   obsolete `test_scaffold_raises_not_implemented_today` test. Left
   the filename `_xfail` for the operator to rename at their
   discretion — touching the filename mid-phase would mix scope.

## Deviations

- **`VolUnavailableError` raise-sites NOT landed in Phase 1.** That
  exception exists in the hierarchy but no code path raises it yet —
  the mode-availability table (research plan §3.6) gets wired up
  alongside the data-quality / fallback chain in the next hardening
  iteration. Phase 1 ships the canonical-mode (minute-enabled-when-
  available, daily-OHLC always) path only.
- **Pre-existing phase-language in
  `liq-features/src/liq/features/regime/bootstrap.py` was removed.**
  The comment now refers to the online-fit-per-fold contract without
  embedding a phase number in shipped source.
