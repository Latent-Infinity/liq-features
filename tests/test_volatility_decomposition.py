"""Contract tests for the decomposition layer.

``estimator_dispersion(estimates)`` returns the sample standard
deviation across estimator variance outputs — the §5.4 "disagreement is
information" signal.

``derive_quality_flags(estimates, *, dispersion_threshold)`` derives
the §5.4 quality-flag set per the table:

- Parkinson low + CtC high → ``GAP_DOMINATED_VOL``
- RS high + CtC low → ``INTRADAY_RANGE_DOMINATED_VOL``
- wide spread overall → ``HIGH_ESTIMATOR_DISAGREEMENT``
- range estimators ≠ CtC → ``CTC_DISAGREES_WITH_RANGE``
"""

from __future__ import annotations

import math

import pytest

from liq.features.volatility.decomposition import (
    derive_quality_flags,
    estimator_dispersion,
)
from liq.features.volatility.quality import (
    FLAG_CTC_DISAGREES_WITH_RANGE,
    FLAG_GAP_DOMINATED_VOL,
    FLAG_HIGH_ESTIMATOR_DISAGREEMENT,
    FLAG_INTRADAY_RANGE_DOMINATED_VOL,
)


class TestEstimatorDispersion:
    def test_zero_dispersion_when_all_estimators_agree(self) -> None:
        estimates = {"ctc": 0.04, "parkinson": 0.04, "garman_klass": 0.04}
        assert estimator_dispersion(estimates) == pytest.approx(0.0, abs=1e-12)

    def test_dispersion_is_population_std(self) -> None:
        estimates = {"a": 0.01, "b": 0.02, "c": 0.03}
        # Mean = 0.02; variance = ((0.01-0.02)² + 0 + (0.03-0.02)²)/3 = 2e-4/3
        expected = math.sqrt((0.0001 + 0.0 + 0.0001) / 3.0)
        assert estimator_dispersion(estimates) == pytest.approx(expected)

    def test_empty_estimates_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            estimator_dispersion({})


class TestDeriveQualityFlags:
    def test_gap_dominated_when_parkinson_low_ctc_high(self) -> None:
        """A gap-day: CtC sees the gap squared into its return, Parkinson
        only sees the within-bar range so it stays small."""
        estimates = {
            "ctc": 0.04,
            "parkinson": 0.005,
            "garman_klass": 0.006,
            "rogers_satchell": 0.006,
        }
        flags = derive_quality_flags(estimates, dispersion_threshold=0.005)
        assert FLAG_GAP_DOMINATED_VOL in flags

    def test_intraday_range_dominated_when_rs_high_ctc_low(self) -> None:
        """A whipsaw day: huge intraday range, but the close returns to
        near the open — Rogers-Satchell registers it, CtC does not."""
        estimates = {
            "ctc": 0.001,
            "parkinson": 0.02,
            "garman_klass": 0.02,
            "rogers_satchell": 0.03,
        }
        flags = derive_quality_flags(estimates, dispersion_threshold=0.005)
        assert FLAG_INTRADAY_RANGE_DOMINATED_VOL in flags

    def test_high_disagreement_when_spread_exceeds_threshold(self) -> None:
        estimates = {
            "ctc": 0.001,
            "parkinson": 0.04,
            "garman_klass": 0.06,
            "rogers_satchell": 0.001,
        }
        flags = derive_quality_flags(estimates, dispersion_threshold=0.005)
        assert FLAG_HIGH_ESTIMATOR_DISAGREEMENT in flags

    def test_ctc_disagrees_with_range_when_ctc_far_from_range_mean(self) -> None:
        """CtC reports a different magnitude than the range estimators
        do — typically a gap-only or intraday-only day."""
        estimates = {
            "ctc": 0.001,
            "parkinson": 0.020,
            "garman_klass": 0.022,
            "rogers_satchell": 0.021,
        }
        flags = derive_quality_flags(estimates, dispersion_threshold=0.005)
        assert FLAG_CTC_DISAGREES_WITH_RANGE in flags

    def test_no_flags_when_estimators_agree(self) -> None:
        estimates = {
            "ctc": 0.04,
            "parkinson": 0.04,
            "garman_klass": 0.04,
            "rogers_satchell": 0.04,
        }
        flags = derive_quality_flags(estimates, dispersion_threshold=0.005)
        # No flag is positive; the empty tuple is the no-flag signal.
        assert flags == ()

    def test_missing_estimator_keys_does_not_crash(self) -> None:
        """If a degraded run only has CtC + Parkinson, derive_quality_flags
        should still return what it can without raising."""
        estimates = {"ctc": 0.04, "parkinson": 0.01}
        flags = derive_quality_flags(estimates, dispersion_threshold=0.005)
        assert isinstance(flags, tuple)
