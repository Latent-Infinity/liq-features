# Phase F — liq-features documentation consolidation

| Field | Value |
| --- | --- |
| Plan | `liq-features-canonical-risk-variance-impl-plan.md` v1.2 |
| Work package | F — Documentation Consolidation & Polish |
| Recorded on | 2026-06-15 |
| Repo | `liq-features` |

## Capability delivered

- `docs/volatility.md` consolidation:
  - Status banner refreshed to reflect production-ready state
    (formula registry + RV/BPV/JV + RV-noise gate + decomposition +
    `volatility_signature` VolComponent all landed).
  - **Production / degraded-fallback mode availability table** added
    (per research plan §3.6). Lists per-component availability under
    each mode and the `VolComponent.source` value.
  - **Full quality-flag vocabulary** enumerated — 10 flags from
    `[DESIGN_DATA_QUALITY]` + §5.4 + §5.3 RV-noise gate; describes
    the source module that sets each flag.
  - **Calendar policy worked example** added: weeknight gap vs
    weekend gap closed-hours computation + `gap_var_per_closed_hour_t`
    diagnostic interpretation.
  - **ATR migration pointer** added linking to
    `liq-risk/docs/atr-retirement.md` for the head-to-head non-
    inferiority evidence.
  - Refinement-skipped note added (matches decisions registry §6).

- `docs/logging.md` cleanup:
  - MCS / ablation / tie-break events split into a "Related" section
    that points to the `liq.experiments.vol` logger. Prior version
    documented these on the features logger by mistake; the actual
    emitters live in `liq-experiments` per Phase 3H.

- `README.md`:
  - "Volatility (canonical risk-variance estimator)" section added
    pointing at `estimate_variance`, the docs surface, and the ATR
    migration evidence.

## What did not change

- `docs/exceptions.md` is already complete and consistent with the
  current exception hierarchy (4 leaf classes + raise-sites + retry-
  eligibility table). No edits required.
- No code change in this phase.

## Acceptance

- Phase-language sweep: only the standalone "refinement (if adopted)"
  → "Refinement (SKIPPED)" status update reference; no phase numbering
  in shipped docs.
- Documentation consolidation list from impl plan §Phase F satisfied
  for liq-features.
