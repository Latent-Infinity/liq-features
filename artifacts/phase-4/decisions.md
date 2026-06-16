# Phase 4 — liq-features negative-result skip

| Field | Value |
| --- | --- |
| Plan | `liq-features-canonical-risk-variance-impl-plan.md` v1.2 |
| Work package | 4 — Refinement (CONDITIONAL — 6A arbitration + 6B calibration) |
| Outcome | **SKIPPED (negative-result anchor recorded)** |
| Recorded on | 2026-06-15 |
| Repo | `liq-features` |

## Why this phase did not run

Per impl plan §Phase 4 — "Conditional phase. Runs ONLY if Phase 3's
decisions registry says `trigger_6a == true` OR `trigger_6b == true`.
If both triggers say `false` (the expected outcome per research plan
§6.2 "skip both"), the agent writes a `decisions.md` note recording
the negative result, skips this phase, and goes directly to Phase 5."

Phase 3 sealed both refinement triggers as
[**`trigger-not-fired`**](../../../liq-docs/plans/liq-features-canonical-risk-variance-ablation-ledger.md)
and the decisions registry §6 documents the rationale:

| menu_id | verdict | decision | frozen_on |
| --- | --- | --- | --- |
| `6a_arbitration` | `trigger-not-fired` | `simpler-kept` | 2026-06-15T18:44:34Z |
| `6b_calibration` | `trigger-not-fired` | `simpler-kept` | 2026-06-15T18:44:34Z |

**§6a (regime arbitration) — trigger condition NOT met:**
MCS_A returned a single survivor (`garman_klass`) and MCS_B returned a
single survivor (`gk_yang_zhang`). The §6a entry gate (research plan
§6.3) requires *multiple* MCS survivors so a regime-conditional
combiner has something to arbitrate between; a single-winner MCS
leaves no arbitration to perform.

**§6b (RV-teacher calibration) — trigger condition NOT met:**
Calibration re-computes against a downstream pipeline change. No
downstream pipeline change has been requested in scope; the §6b entry
gate (research plan §6.4) is therefore not engaged.

## What this means for liq-features

- **No code change in this phase.** Neither
  `src/liq/features/volatility/refinement/arbitration.py` nor
  `src/liq/features/volatility/refinement/calibration.py` is created —
  per the impl plan, those files only land when the conditional
  triggers fire.
- **No new dependencies, no new exception classes, no new structured-
  log events.** The §6.3 production-observability hook and the §6.4
  PIT-multiplier persistence layer remain unimplemented because no
  consumer requires them yet.
- **One-shot rule unaffected.** The §1.7 one-shot rule for
  `6a_arbitration` and `6b_calibration` is preserved as "trigger-not-
  fired". If later evidence invalidates either trigger outcome, work
  must start from an explicit plan/ledger amendment; this skip does not
  silently reopen a sealed row.

## Next phase

Phase 5 (downstream sizing validation against `risk_var_t` + ATR-
compatibility bridge) is the next workstream. The canonical estimator
it consumes is **`gk_yang_zhang`** per decisions registry §5 — the
MCS_B winner under Option B.

## Acceptance

- Ablation ledger rows for `6a_arbitration` and `6b_calibration`
  remain frozen at `trigger-not-fired` / `simpler-kept`.
- Coordinator state file marks all Phase 4 cells as `skipped` with
  this file as the negative-result anchor.
