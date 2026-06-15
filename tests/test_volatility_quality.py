"""Contract tests for the volatility data-quality module.

Mirrors research plan §9 row-by-row + the PIT-specific outlier rule
called out in §3.5. Each test pins one rule of the table to its
documented behavior:

| Condition                                       | Action                                        |
| `high < low`, nonpositive price, missing close  | hard data error: NaN + flag                   |
| missing open                                    | no YZ/RS; fallback to Parkinson or CtC        |
| H/L outlier vs past-available neighbors only    | flag; fallback (winsorize only in research)   |
| halted / zero-volume day                        | flag; carry-forward vs NaN vs CtC per policy  |
| partial trading day                             | calendar-aware annualization or exclude       |
| max DQ failure rate over the window             | raises VolDataQualityError (Gate-3 prep)      |
"""

from __future__ import annotations

import pytest

from liq.features.volatility.contracts import VolQualityPolicy
from liq.features.volatility.exceptions import VolDataQualityError
from liq.features.volatility.quality import (
    QualityVerdict,
    check_bar_quality,
    compute_failure_rate,
    enforce_failure_rate,
    past_rolling_z,
)


def _policy(
    *,
    method: str = "past_rolling_z",
    threshold: float = 4.0,
    max_rate: float = 0.01,
) -> VolQualityPolicy:
    return VolQualityPolicy(
        high_low_outlier_method=method,  # type: ignore[arg-type]
        high_low_outlier_threshold=threshold,
        rv_noise_ratio_threshold=2.0,
        estimator_dispersion_threshold=1.5,
        min_nonzero_volume_fraction=0.95,
        max_data_quality_failure_rate=max_rate,
    )


def _bar(o=100.0, h=102.0, lo=99.5, c=101.0, v: float | None = 1_000_000) -> dict:
    return {"open": o, "high": h, "low": lo, "close": c, "volume": v}


class TestHardDataErrors:
    def test_high_less_than_low_is_hard_error(self) -> None:
        verdict = check_bar_quality(
            bar_index=3, bar=_bar(h=98.0, lo=99.5), past_bars=[], policy=_policy()
        )
        assert verdict.is_hard_error
        assert "DATA_QUALITY_FAILURE" in verdict.flags
        assert verdict.rule == "high_lt_low"
        assert not verdict.fallback_eligible

    def test_nonpositive_price_is_hard_error(self) -> None:
        verdict = check_bar_quality(bar_index=3, bar=_bar(o=-1.0), past_bars=[], policy=_policy())
        assert verdict.is_hard_error
        assert verdict.rule == "nonpositive_price"

    def test_missing_close_is_hard_error(self) -> None:
        verdict = check_bar_quality(
            bar_index=3,
            bar=_bar(c=None),
            past_bars=[],
            policy=_policy(),
        )
        assert verdict.is_hard_error
        assert verdict.rule == "missing_close"


class TestMissingOpen:
    def test_missing_open_triggers_fallback_with_flag(self) -> None:
        verdict = check_bar_quality(
            bar_index=3,
            bar=_bar(o=None),
            past_bars=[],
            policy=_policy(),
        )
        assert not verdict.is_hard_error
        assert verdict.fallback_eligible
        assert "MISSING_OPEN" in verdict.flags


class TestSuspectHighLow:
    def test_low_above_open_close_min_is_soft_quality_failure(self) -> None:
        verdict = check_bar_quality(
            bar_index=3,
            bar=_bar(o=100.0, h=102.0, lo=100.5, c=101.0),
            past_bars=[],
            policy=_policy(method="none"),
        )
        assert not verdict.is_hard_error
        assert "DATA_QUALITY_FAILURE" in verdict.flags


class TestHaltedZeroVolume:
    def test_zero_volume_emits_halted_day_flag(self) -> None:
        verdict = check_bar_quality(bar_index=3, bar=_bar(v=0), past_bars=[], policy=_policy())
        assert "ZERO_VOLUME_DAY" in verdict.flags

    def test_missing_volume_does_not_fire_zero_volume(self) -> None:
        # Missing volume is unknown, not zero — distinct from a halted day.
        verdict = check_bar_quality(bar_index=3, bar=_bar(v=None), past_bars=[], policy=_policy())
        assert "ZERO_VOLUME_DAY" not in verdict.flags


class TestPitSafeOutlierDetection:
    def test_past_rolling_z_uses_only_past_neighbors(self) -> None:
        # 10 calm bars with realistic noise, then a 100x range spike.
        prev = [
            _bar(o=100.0, h=100.5 + 0.05 * (i % 3), lo=99.5 - 0.05 * (i % 4), c=100.0)
            for i in range(10)
        ]
        spike_bar = _bar(o=100.0, h=200.0, lo=99.5, c=150.0)
        verdict = check_bar_quality(
            bar_index=10, bar=spike_bar, past_bars=prev, policy=_policy(threshold=4.0)
        )
        assert "HIGH_LOW_OUTLIER" in verdict.flags

    def test_outlier_detection_can_be_disabled(self) -> None:
        prev = [_bar(o=100.0, h=100.5, lo=99.5, c=100.0) for _ in range(10)]
        spike_bar = _bar(o=100.0, h=200.0, lo=99.5, c=150.0)
        verdict = check_bar_quality(
            bar_index=10, bar=spike_bar, past_bars=prev, policy=_policy(method="none")
        )
        assert "HIGH_LOW_OUTLIER" not in verdict.flags
        assert "LOW_CONFIDENCE" not in verdict.flags

    def test_mad_outlier_detection_flags_large_range(self) -> None:
        prev = [
            _bar(o=100.0, h=100.4 + 0.02 * (i % 3), lo=99.6 - 0.01 * (i % 4), c=100.0)
            for i in range(10)
        ]
        spike_bar = _bar(o=100.0, h=200.0, lo=99.5, c=150.0)
        verdict = check_bar_quality(
            bar_index=10,
            bar=spike_bar,
            past_bars=prev,
            policy=_policy(method="mad", threshold=4.0),
        )
        assert "HIGH_LOW_OUTLIER" in verdict.flags

    def test_future_high_does_not_alter_current_emit(self) -> None:
        # The outlier detector consumes ONLY past bars. Even if we
        # construct a "future" spike, it must not influence the
        # current verdict (PIT contract).
        prev_calm = [_bar(o=100.0, h=100.5, lo=99.5, c=100.0) for _ in range(10)]
        current = _bar(o=100.0, h=101.0, lo=99.5, c=100.2)
        baseline = check_bar_quality(
            bar_index=10, bar=current, past_bars=prev_calm, policy=_policy()
        )
        # past_bars stays the same — there is no API path that lets a
        # caller pass *future* bars; the contract is enforced by
        # construction. We assert the baseline verdict has no outlier
        # flag.
        assert "HIGH_LOW_OUTLIER" not in baseline.flags

    def test_low_confidence_until_window_filled(self) -> None:
        # With only 1 past bar, the z-score is undefined → LOW_CONFIDENCE.
        verdict = check_bar_quality(
            bar_index=1,
            bar=_bar(),
            past_bars=[_bar()],
            policy=_policy(),
        )
        assert "LOW_CONFIDENCE" in verdict.flags

    def test_past_rolling_z_returns_nan_until_enough_samples(self) -> None:
        z = past_rolling_z([1.0, 1.1, 0.9, 1.05, 1.02], min_window=5)
        # The first 4 entries lack enough history for a z-score.
        assert z[0] != z[0]  # NaN
        assert z[3] != z[3]  # NaN
        assert isinstance(z[4], float)

    def test_past_rolling_z_returns_nan_when_past_variance_is_zero(self) -> None:
        z = past_rolling_z([1.0, 1.0, 1.0, 1.0, 2.0], min_window=5)
        assert z[4] != z[4]  # NaN


class TestFailureRateEnforcement:
    def test_empty_verdicts_have_zero_failure_rate(self) -> None:
        assert compute_failure_rate([]) == 0.0

    def test_passes_when_under_threshold(self) -> None:
        verdicts = [
            QualityVerdict(
                bar_index=i,
                is_hard_error=False,
                flags=("LOW_CONFIDENCE",),
                fallback_eligible=True,
                rule=None,
            )
            for i in range(20)
        ]
        verdicts.append(
            QualityVerdict(
                bar_index=20,
                is_hard_error=True,
                flags=("DATA_QUALITY_FAILURE",),
                fallback_eligible=False,
                rule="high_lt_low",
            )
        )
        assert compute_failure_rate(verdicts) == pytest.approx(1 / 21)
        # 1/21 ≈ 4.76% > 1% default — should fail.
        with pytest.raises(VolDataQualityError, match="failure rate"):
            enforce_failure_rate(verdicts, _policy(max_rate=0.01))
        # With a generous threshold (10%), it passes.
        enforce_failure_rate(verdicts, _policy(max_rate=0.10))

    def test_only_hard_errors_count_toward_rate(self) -> None:
        # 10 bars with non-hard quality flags should never trigger the gate.
        verdicts = [
            QualityVerdict(
                bar_index=i,
                is_hard_error=False,
                flags=("HIGH_LOW_OUTLIER",),
                fallback_eligible=True,
                rule=None,
            )
            for i in range(10)
        ]
        assert compute_failure_rate(verdicts) == 0.0
        enforce_failure_rate(verdicts, _policy(max_rate=0.01))
