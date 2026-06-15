# Logging contract (`liq.features.volatility`)

The estimator emits structured log events on the `liq.features.volatility`
logger. Every decision branch point (estimator selection, fallback,
quality-flag emission, gate firing, refinement application) emits an
INFO event with the required fields below. Errors include the exception
type and context per the autonomous-plan-guide's error-handling standard.

## Required fields (every event)

| Field | Source | Purpose |
| --- | --- | --- |
| `timestamp` | std-logging | event time |
| `level` | std-logging | INFO / WARNING / ERROR |
| `event` | structured | event name (catalog below) |
| `correlation_id` | bound by caller or generated | trace ID across one estimate run |
| `estimator_used` | bound per estimate | which formula produced this bar |
| `spec_hash` | bound per estimate | hash of the `VolEstimatorSpec` |

## Per-estimate bound fields

Set once per `estimate_variance(...)` call:

`symbol`, `asof`, `window`, `quality_flags`.

## Event catalog

The events below are emitted on every `estimate_variance(...)` call by
the data-quality / fallback / logging hookups. Future iterations add
the minute-mode events (RV-noise gate, BPV/JV decomposition).

| Event | Trigger / context |
| --- | --- |
| `estimator_selected` | Per research plan §4.2 fallback chain — which estimator the spec resolved to. Includes `resolved` field carrying the post-fallback name. |
| `estimator_fallback_applied` | A fallback was used instead of the requested estimator. Fields: `from`, `to`, `reason`. |
| `quality_flag_set` | A `quality_flag` was set on a bar. Debounced to one event per bar even when multiple flags fire (avoids duplication with the windowed log path). |
| `data_quality_rejected` (ERROR) | A bar failed quality with no fallback eligible. Includes the rule that fired. |
| `rv_noise_gate_fired` | The RV-noise gate (research plan §5.3) rejected `RV_1m` and fell back. |
| `mcs_eliminated` | The Hansen-Lunde-Nason elimination step removed a candidate. |
| `tie_break_applied` | The pre-registered `tie_break_order` decided between MCS members. |
| `ablation_recalibration_proposed` | An ablation produced a `RecalibrationProposal` (research plan §10.2). |
| `refinement_applied` | 6A arbitration or 6B calibration was applied (when adopted per `[PHASE25_REFINEMENT]`). |

## Example records

Captured via `caplog` in `tests/test_volatility_logging.py`. Field
names match the structured-payload accessor (`record.structured`):

```python
# estimator_selected — every emit
{
    "timestamp": "2026-06-14T18:42:11.137081+00:00",
    "level": "INFO",
    "event": "estimator_selected",
    "correlation_id": "e08fa040d56d4da48fa6d0d4b80f6561",
    "estimator_used": "yang_zhang",
    "spec_hash": "deadbeef",
    "symbol": "AAPL",
    "asof": "2024-06-03T20:00:00Z",
    "window": 21,
    "resolved": "yang_zhang",
}

# estimator_fallback_applied — when the chain demoted
{
    "event": "estimator_fallback_applied",
    "from": "yang_zhang",
    "to": "rogers_satchell",
    "reason": "missing_prev_close",
    # ... plus the required + bound fields
}

# quality_flag_set — debounced per bar
{
    "event": "quality_flag_set",
    "bar_index": 2,
    "quality_flags": ["GAP_DOMINATED_VOL", "HIGH_LOW_OUTLIER"],
}

# data_quality_rejected (ERROR) — paired with VolDataQualityError
{
    "level": "ERROR",
    "event": "data_quality_rejected",
    "rule": "failure_rate_exceeded",
    "failure_rate": "data-quality failure rate 0.0476 exceeds threshold 0.0100",
}
```

## Debouncing

`quality_flag_set` is debounced per `(estimator_call, bar_index)` — a
second call for the same bar updates the existing record's
`quality_flags` field (UNION semantics) rather than emitting a second
event. This guarantees the "one event per bar" contract even when both
the formula path and the windowed path mark the same bar.

## No secrets

No estimator input or output is sensitive. The logger never serializes
API keys or other secrets — the package does not hold any (the only
network-dependent calibration consults `liq-data` which carries its own
secrets in `liq-data/.env`). Any field name in
`{secret, password, token, api_key}` (case-insensitive) is redacted to
`"***"` before the structured payload is committed to the record.
