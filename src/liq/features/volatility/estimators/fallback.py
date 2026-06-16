"""Per-bar fallback chain — research plan §4.2.

Given a requested estimator and the bar's :class:`QualityVerdict`, this
module decides which estimator actually emits. The chain is ORDERED:

1. Hard data-quality error → no emit.
2. Suspect H/L (the bar's high or low contradicts O/C) → CtC if
   prev close is available, otherwise no emit.
3. Missing open → Parkinson (uses only H/L).
4. Requested estimator needs prev close but it's missing →
   Rogers-Satchell (for YZ) or Garman-Klass (for GK-YZ); CtC cannot
   degrade further, so it goes to no-emit.
5. Otherwise → emit the requested estimator.

The decision carries `requested`, `selected`, and `reason`, which the
caller threads into the structured-log event
``estimator_fallback_applied``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from liq.features.volatility.quality import (
    FLAG_DATA_QUALITY_FAILURE,
    FLAG_MISSING_OPEN,
)

if TYPE_CHECKING:
    from liq.features.volatility.quality import QualityVerdict


# Estimators that need the previous close to emit.
_NEEDS_PREV_CLOSE = frozenset({"ctc", "yang_zhang", "gk_yang_zhang"})


@dataclass(frozen=True)
class FallbackDecision:
    """Per-bar fallback outcome.

    ``selected is None`` means the bar cannot emit at all under the
    active policy; the caller emits NaN and a quality flag.
    """

    requested: str
    selected: str | None
    reason: str | None


def select_estimator(
    *,
    requested: str,
    verdict: QualityVerdict,
    has_prev_close: bool,
) -> FallbackDecision:
    """Apply research plan §4.2 in order; return the active selection.

    The fallback chain favors the cheapest estimator that can still
    emit *something* — Parkinson when the open is gone, CtC when the
    range is suspect — so the caller never has to silently emit NaN
    when a downgrade can save the bar.
    """
    if verdict.is_hard_error:
        return FallbackDecision(
            requested=requested, selected=None, reason="hard_data_quality_error"
        )

    # H/L suspect rule (rows 4-5 of §9): always demotes to CtC.
    if FLAG_DATA_QUALITY_FAILURE in verdict.flags and not verdict.is_hard_error:
        if has_prev_close:
            return FallbackDecision(requested=requested, selected="ctc", reason="suspect_high_low")
        return FallbackDecision(
            requested=requested,
            selected=None,
            reason="suspect_high_low_no_prev_close",
        )

    # Missing open (row 2 of §9): YZ/RS/GK-YZ/GK can't run; Parkinson works.
    if FLAG_MISSING_OPEN in verdict.flags:
        return FallbackDecision(requested=requested, selected="parkinson", reason="missing_open")

    # Missing prev close demotion for estimators that need it.
    if requested in _NEEDS_PREV_CLOSE and not has_prev_close:
        if requested == "yang_zhang":
            return FallbackDecision(
                requested=requested, selected="rogers_satchell", reason="missing_prev_close"
            )
        if requested == "gk_yang_zhang":
            return FallbackDecision(
                requested=requested, selected="garman_klass", reason="missing_prev_close"
            )
        # CtC cannot demote further — no emit.
        return FallbackDecision(requested=requested, selected=None, reason="missing_prev_close")

    # No demotion needed.
    return FallbackDecision(requested=requested, selected=requested, reason=None)


__all__ = ["FallbackDecision", "select_estimator"]
