# AGENT_STATE.md — liq-features

State tracked per implementation plan that touches this repo.

## Plan: `liq-features-canonical-risk-variance-impl-plan` (v1.2)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Volatility package scaffold: contracts, exceptions, scaffolded `estimate_variance`, xfail-strict contract tests, doc skeletons | done | recorded |
| 1 | Formula registry (CtC, Parkinson, GK, RS, GK-YZ, YZ) + windowed aggregation + `estimate_variance` orchestration (spec validation, PIT enforcement, dispatch, annualization, c2c risk composition) + smoke script + docs/volatility.md formula section | done | recorded |
| 1H | Data-quality rules (§9) + PIT-safe outlier detection + fallback chain (§4.2) + structured logging (§1.9) wired into `estimate_variance`; perf smoke (80 calls/s at 252×500); docs/exceptions.md + docs/logging.md updated | done | recorded |
| 2H | rv.py: compute_rv / compute_bpv / compute_jv / rv_noise_gate (§5.3); decomposition.py: estimator_dispersion + §5.4 quality-flag derivation; docs/volatility.md minute-mode + dispersion sections; docs/logging.md events | ready-for-review | recorded |
| 4 | Refinement (conditional): SKIPPED — both `6a_arbitration` and `6b_calibration` triggers landed `trigger-not-fired` in Phase 3 (single-survivor MCS_A/B + no downstream pipeline change). No `refinement/arbitration.py` or `refinement/calibration.py` shipped; negative-result anchor recorded | skipped | `artifacts/phase-4/decisions.md` |
| F | Documentation consolidation: docs/volatility.md status banner refreshed, production/degraded mode availability table added, full quality-flag vocabulary enumerated, ATR migration pointer added, calendar worked example, refinement-skipped note; docs/logging.md split out MCS events to `liq.experiments.vol` logger with cross-doc pointer; README.md Volatility section added pointing at `estimate_variance` | ready-for-review | `artifacts/phase-F/` |


## Plan: `mean-reversion-swing-scan-impl-plan` (v0.3)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| F0 | Mean-reversion feature namespace under construction; vol/base helper import contract pinned by strict xfail test | ready-for-review | `src/liq/features/mean_reversion/__init__.py`; `tests/test_mean_reversion_contract_imports.py` |
