# Exception hierarchy (`liq.features.volatility`)

Every error raised by the canonical risk-variance estimator derives
from `VolFeatureError`. Catch the family with a single `except` when
the response is "fall back to incumbent vol"; catch a leaf when the
response depends on the cause.

```text
VolFeatureError
├── VolSpecError            # malformed VolEstimatorSpec
├── VolDataQualityError     # input bars failed quality with no fallback eligible
├── VolPITViolationError    # max(input.valid_from) > t
└── VolUnavailableError     # component cannot be produced in the active mode
```

| Leaf class | When raised |
| --- | --- |
| `VolSpecError` | Spec validation fails — e.g. `estimator='rv'` without `rv_spec`, or `target='quadratic_variation'` against a non-minute estimator. Detected before any bar is consumed. |
| `VolDataQualityError` | A bar fails a data-quality rule and the active `VolQualityPolicy` has no fallback eligible (e.g. missing close on a non-skippable bar, or the window's data-quality failure rate exceeds `max_data_quality_failure_rate`). |
| `VolPITViolationError` | The set of input rows feeding the estimate at time `t` includes any row with `valid_from > t`. Hard PIT gate from research plan §3.5. |
| `VolUnavailableError` | The caller requested a component the active mode cannot produce — typically intraday RV / BPV / jump in the degraded daily-OHLC-only fallback per research plan §3.6 mode-availability table. |

All four leaf classes carry the standard `Exception` interface; runtime
context is emitted through structured log records immediately before
the corresponding raise-site when a rejection is observable.

## Raise-sites + retry eligibility

| Class | Raise-site | Caller response | Retry-eligible? |
| --- | --- | --- | --- |
| `VolSpecError` | `estimate.py::_validate_spec` (spec self-inconsistency) and `_validate_bars` (missing columns / empty frame). | Fix the spec / frame; the call cannot succeed as-is. | No — deterministic; same input always raises. |
| `VolPITViolationError` | `estimate.py::_check_pit` — raised when `max(input.valid_from) > asof`. | Audit the frame's `valid_from` column or move `asof` forward; never silently drop the offending row. | No — by design; the violation indicates a data-pipeline bug upstream. |
| `VolDataQualityError` | `quality.py::enforce_failure_rate` — raised when the window's `hard error` rate exceeds `quality_policy.max_data_quality_failure_rate`. The structured-log event `data_quality_rejected` fires immediately before the raise. | Investigate the offending bars (the structured log carries the failing rule); ingest cleaner data or accept the gap as a NaN-emit elsewhere. | Conditional — re-run after data fix; the underlying inputs must change. |
| `VolUnavailableError` | Reserved; raised when a `VolComponent` is requested in the active mode that cannot produce it (e.g. intraday RV / BPV / jump in the daily-OHLC-only degraded fallback). | Either switch to the minute-enabled production mode or drop the component from the consumer's contract. | No — mode-determined. |

Every `VolDataQualityError` is paired with at least one
`data_quality_rejected` structured-log event so audit tooling can
correlate the raise with its underlying rule and bar index.
