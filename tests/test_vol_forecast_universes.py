from __future__ import annotations

from datetime import date

from liq.features.vol_forecast import BaselineEligibility, compute_universe_membership


def test_common_eligible_requires_all_baselines() -> None:
    membership = compute_universe_membership(
        "NVDA",
        date(2024, 1, 3),
        {
            "ewma": BaselineEligibility(True, 252, 22),
            "har": BaselineEligibility(True, 252, 22),
            "garch": BaselineEligibility(True, 900, 750),
        },
    )

    assert membership == "common-eligible"


def test_missing_long_history_is_limited_history() -> None:
    membership = compute_universe_membership(
        "NVDA",
        date(2024, 1, 3),
        {
            "ewma": BaselineEligibility(True, 252, 22),
            "har": BaselineEligibility(True, 252, 22),
            "garch": BaselineEligibility(False, 252, 750),
        },
    )

    assert membership == "limited-history"


def test_production_coverage_when_non_long_history_baseline_is_missing() -> None:
    membership = compute_universe_membership(
        "NVDA",
        date(2024, 1, 3),
        {
            "ewma": BaselineEligibility(True, 252, 22),
            "har": BaselineEligibility(False, 10, 22),
        },
    )

    assert membership == "production-coverage"
