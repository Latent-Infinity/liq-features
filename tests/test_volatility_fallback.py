"""Contract tests for the volatility fallback chain.

Mirrors research plan §4.2 exactly:

    O/H/L/C and previous close reliable        -> Yang-Zhang
    prev close / overnight gap unavailable,
      but O/H/L/C reliable                       -> Rogers-Satchell
    O missing, but H/L/C reliable                -> Parkinson (if H/L pass), else CtC
    H/L suspect                                  -> CtC
    minute data + microstructure filters pass    -> RV mode (out of scope here)

Each chain entry pins one row of that table.
"""

from __future__ import annotations

import pytest

from liq.features.volatility.estimators.fallback import (
    FallbackDecision,
    select_estimator,
)
from liq.features.volatility.quality import (
    FLAG_DATA_QUALITY_FAILURE,
    FLAG_MISSING_OPEN,
    QualityVerdict,
)


def _verdict(
    bar_index: int = 0,
    *,
    flags: tuple[str, ...] = (),
    is_hard: bool = False,
    fallback_eligible: bool = True,
    rule: str | None = None,
) -> QualityVerdict:
    return QualityVerdict(
        bar_index=bar_index,
        is_hard_error=is_hard,
        flags=flags,
        fallback_eligible=fallback_eligible,
        rule=rule,
    )


class TestPrimaryEstimatorWhenInputsClean:
    @pytest.mark.parametrize(
        "requested,expected",
        [
            ("yang_zhang", "yang_zhang"),
            ("rogers_satchell", "rogers_satchell"),
            ("parkinson", "parkinson"),
            ("garman_klass", "garman_klass"),
            ("gk_yang_zhang", "gk_yang_zhang"),
            ("ctc", "ctc"),
        ],
    )
    def test_no_fallback_when_inputs_clean(self, requested: str, expected: str) -> None:
        decision = select_estimator(
            requested=requested,
            verdict=_verdict(),
            has_prev_close=True,
        )
        assert decision.selected == expected
        assert decision.reason is None


class TestFallbackOnMissingPrevClose:
    def test_yang_zhang_falls_back_to_rogers_satchell(self) -> None:
        decision = select_estimator(
            requested="yang_zhang", verdict=_verdict(), has_prev_close=False
        )
        assert decision.selected == "rogers_satchell"
        assert decision.reason == "missing_prev_close"

    def test_gk_yang_zhang_falls_back_to_garman_klass(self) -> None:
        decision = select_estimator(
            requested="gk_yang_zhang", verdict=_verdict(), has_prev_close=False
        )
        assert decision.selected == "garman_klass"
        assert decision.reason == "missing_prev_close"

    def test_ctc_falls_back_to_close_to_close_within_session(self) -> None:
        # CtC without prev close cannot emit; chain falls back to no-emit
        # (the bar is unfit) which is encoded as fallback_eligible=False.
        decision = select_estimator(requested="ctc", verdict=_verdict(), has_prev_close=False)
        assert decision.selected is None
        assert decision.reason == "missing_prev_close"


class TestFallbackOnMissingOpen:
    def test_yang_zhang_with_missing_open_falls_back_to_parkinson(self) -> None:
        decision = select_estimator(
            requested="yang_zhang",
            verdict=_verdict(flags=(FLAG_MISSING_OPEN,), rule="missing_open"),
            has_prev_close=True,
        )
        assert decision.selected == "parkinson"
        assert decision.reason == "missing_open"

    def test_rogers_satchell_with_missing_open_falls_back_to_parkinson(self) -> None:
        decision = select_estimator(
            requested="rogers_satchell",
            verdict=_verdict(flags=(FLAG_MISSING_OPEN,), rule="missing_open"),
            has_prev_close=True,
        )
        assert decision.selected == "parkinson"
        assert decision.reason == "missing_open"


class TestFallbackOnSuspectHighLow:
    def test_yang_zhang_with_suspect_hl_falls_back_to_ctc(self) -> None:
        decision = select_estimator(
            requested="yang_zhang",
            verdict=_verdict(flags=(FLAG_DATA_QUALITY_FAILURE,)),
            has_prev_close=True,
        )
        assert decision.selected == "ctc"
        assert decision.reason == "suspect_high_low"

    def test_parkinson_with_suspect_hl_falls_back_to_ctc(self) -> None:
        decision = select_estimator(
            requested="parkinson",
            verdict=_verdict(flags=(FLAG_DATA_QUALITY_FAILURE,)),
            has_prev_close=True,
        )
        assert decision.selected == "ctc"
        assert decision.reason == "suspect_high_low"

    def test_suspect_hl_without_prev_close_yields_no_emit(self) -> None:
        decision = select_estimator(
            requested="yang_zhang",
            verdict=_verdict(flags=(FLAG_DATA_QUALITY_FAILURE,)),
            has_prev_close=False,
        )
        # H/L suspect → CtC needed; CtC needs prev close → no fit.
        assert decision.selected is None
        assert decision.reason == "suspect_high_low_no_prev_close"


class TestHardErrorYieldsNoEmit:
    def test_hard_error_returns_none(self) -> None:
        decision = select_estimator(
            requested="yang_zhang",
            verdict=_verdict(is_hard=True, fallback_eligible=False, rule="high_lt_low"),
            has_prev_close=True,
        )
        assert decision.selected is None
        assert decision.reason == "hard_data_quality_error"


class TestFallbackDecisionShape:
    def test_decision_carries_from_to_reason(self) -> None:
        decision = select_estimator(
            requested="yang_zhang", verdict=_verdict(), has_prev_close=False
        )
        assert isinstance(decision, FallbackDecision)
        assert decision.requested == "yang_zhang"
        assert decision.selected == "rogers_satchell"
        assert decision.reason == "missing_prev_close"
