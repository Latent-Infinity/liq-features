# Phase 1H — liq-features decisions

## Decisions

1. **Window-level fallback scans the whole window, not only the last
   bar.** The chain still resolves one estimator per
   `estimate_variance` call, because mixing per-bar estimators inside a
   single trailing mean would aggregate apples and oranges (a YZ
   contribution at bar t-3 mixed with a Parkinson contribution at bar
   t-2). The decision now inspects every verdict in the window so an
   earlier missing open or suspect H/L row cannot be missed by a clean
   final bar. Per-bar `estimator_used` equals the resolved estimator
   for every bar in this call. The richer "per-bar swap" model lands
   when the minute-RV path comes online and the windowed composition
   genuinely supports mixed inputs.
2. **`spec_hash` uses BLAKE2-8 over a flattened-asdict canonical
   encoding.** Two runs with the same `VolEstimatorSpec` produce the
   same hash regardless of dataclass field-order changes. Short hash
   (8 bytes / 16 hex chars) is enough to disambiguate spec versions in
   logs without bloat. BLAKE2 over `hashlib.sha256` for speed; cryptographic
   strength is not the goal — collision-resistance for human-readable
   audit trails is.
3. **Secret-key redaction is a fixed deny-list, case-insensitive.**
   `{secret, password, token, api_key}` covers the operational surface;
   the package doesn't handle PII so this list is intentionally narrow.
   A more permissive policy would force every consumer to opt out of
   redaction for non-sensitive fields like `from`/`to` that happen to
   contain "key"-shaped substrings.
4. **`quality_flag_set` debouncing uses record mutation, not event
   suppression.** When a second call adds a flag to the same
   `(bar_index)`, the emitter walks back to the prior record in
   handler buffers and merges flag sets directly. This guarantees the
   "one event per bar" contract without losing information — the
   alternative (suppress the second call) would silently drop flags
   added later in the call.
5. **PIT-safe outlier detection consumes ONLY `past_bars`.** The
   detector accepts a sequence of bars strictly before index `i`;
   there is no API path that lets a caller pass future bars. The PIT
   contract is enforced by *construction*, not by runtime check. The
   downside: it forces the caller (`estimate.py::_per_bar_quality`) to
   slice `rows[:i]` explicitly.
6. **Fallback `select_estimator` is window-once but based on a
   window-priority verdict.** Suspect H/L wins over missing open, and
   missing open wins over an otherwise clean final-bar verdict. Even
   when the fallback chain is a no-op (clean inputs → return
   requested), `estimator_selected` still fires with the resolved
   value. This guarantees consumers always see the "this is what we
   used" event, even when no demotion occurred.
7. **Failure-rate gate counts only `is_hard_error == True`.** Soft
   flags (LOW_CONFIDENCE, HIGH_LOW_OUTLIER, ZERO_VOLUME_DAY) do not
   count toward the Gate-3 threshold. The threshold is meant to catch
   "this window's data is unusable" scenarios, not "some bars have
   flags".

## Deviations

- **`VolUnavailableError` still has no raise-site.** That lands when
  the minute-mode component table is wired up; the daily-OHLC path
  this iteration ships does not need it.
- **`quality.py` accepts `dict[str, object]` bar rows from polars'
  `to_dicts()`.** The columns are `open`, `high`, `low`, `close`,
  optionally `volume`. Strict typing (`Bar` dataclass) is deferred —
  rolling our own boundary type would couple `liq-features` to a new
  cross-cutting protocol; threading polars-row dicts keeps the
  surface flat and matches how the windowed code already reads bars.
- **The structured `timestamp` field in log records is computed via
  `datetime.now(UTC)` at emit time.** Two events from the same emitter
  CAN carry different `timestamp` values when the emit fires
  microseconds apart. For the audit-trail contract this is correct
  (each event's `timestamp` reflects WHEN it fired); for
  reproducibility tooling the deterministic anchor is `spec_hash` +
  `correlation_id`, not `timestamp`.
