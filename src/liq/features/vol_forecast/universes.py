"""Deterministic forecast-universe assignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

UniverseMembership = Literal["common-eligible", "production-coverage", "limited-history"]
UNIVERSE_DEFINITION_ID = "vol_forecast_universe_v1"


@dataclass(frozen=True)
class BaselineEligibility:
    """Eligibility status for one baseline at an as-of date."""

    eligible: bool
    sessions_available: int
    required_sessions: int


def _coerce(value: BaselineEligibility | bool) -> BaselineEligibility:
    if isinstance(value, BaselineEligibility):
        return value
    return BaselineEligibility(
        eligible=bool(value),
        sessions_available=1 if value else 0,
        required_sessions=1,
    )


def compute_universe_membership(
    symbol: str,
    asof: date | datetime,
    baseline_eligibility: Mapping[str, BaselineEligibility | bool],
) -> UniverseMembership:
    """Return the versioned universe bucket for a symbol at an as-of date."""

    if not symbol:
        raise ValueError("symbol must not be empty")
    if asof is None:
        raise ValueError("asof must not be None")
    if not baseline_eligibility:
        return "limited-history"

    coerced = {name.lower(): _coerce(value) for name, value in baseline_eligibility.items()}
    if all(value.eligible for value in coerced.values()):
        return "common-eligible"

    has_production_history = any(value.sessions_available >= 22 for value in coerced.values())
    garch = coerced.get("garch") or coerced.get("garch_rv")
    if garch is not None and not garch.eligible and has_production_history:
        return "limited-history"
    if has_production_history:
        return "production-coverage"
    return "limited-history"


__all__ = [
    "BaselineEligibility",
    "UNIVERSE_DEFINITION_ID",
    "UniverseMembership",
    "compute_universe_membership",
]
