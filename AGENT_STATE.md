# AGENT_STATE.md — liq-features

State tracked per implementation plan that touches this repo.

## Plan: `liq-features-canonical-risk-variance-impl-plan` (v1.2)

| Phase | Capability | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Volatility package scaffold: contracts, exceptions, scaffolded `estimate_variance`, xfail-strict contract tests, doc skeletons | done | `artifacts/phase-0/` |
| 1 | Formula registry (CtC, Parkinson, GK, RS, GK-YZ, YZ) + windowed aggregation + `estimate_variance` orchestration (spec validation, PIT enforcement, dispatch, annualization, c2c risk composition) + smoke script + docs/volatility.md formula section | done | `artifacts/phase-1/` |
| 1H | Data-quality rules (§9) + PIT-safe outlier detection + fallback chain (§4.2) + structured logging (§1.9) wired into `estimate_variance`; perf smoke (80 calls/s at 252×500); docs/exceptions.md + docs/logging.md updated | ready-for-review | `artifacts/phase-1H/` |
