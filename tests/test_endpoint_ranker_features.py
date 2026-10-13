"""Source-free window controls; no fabricated financial observations."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from liq.features.ranker_features import (
    CloseObservation,
    FeatureUnavailable,
    PriceHistory,
    SourceIdentity,
    reversal_21_sessions,
    skip_month_momentum,
)


def monthly_endpoints() -> tuple[datetime, ...]:
    return tuple(datetime(2023, month, 28, tzinfo=UTC) for month in range(1, 13))


def session_endpoints() -> tuple[datetime, ...]:
    return tuple(datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(22))


def missing_history(endpoints: tuple[datetime, ...]) -> PriceHistory:
    return PriceHistory(
        "security",
        SourceIdentity("contract-only", "v1", "0" * 64),
        "unadmitted",
        tuple(CloseObservation(ts, None, ts) for ts in endpoints),
    )


def evaluate(kind: str, history: PriceHistory, endpoints: tuple[datetime, ...], cutoff: datetime):
    if kind == "momentum":
        return skip_month_momentum(history, endpoints, date(2024, 1, 1), cutoff)
    return reversal_21_sessions(history, endpoints, cutoff)


@pytest.mark.parametrize("kind", ["momentum", "reversal"])
@pytest.mark.parametrize("mutation", ["short", "long", "reverse", "duplicate", "future", "naive"])
def test_inexact_endpoint_metadata_is_refused(kind: str, mutation: str) -> None:
    endpoints = monthly_endpoints() if kind == "momentum" else session_endpoints()
    history = missing_history(endpoints)
    cutoff = endpoints[-1]
    if mutation == "short":
        endpoints = endpoints[:-1]
    elif mutation == "long":
        endpoints = (*endpoints, cutoff + timedelta(days=1))
    elif mutation == "reverse":
        endpoints = tuple(reversed(endpoints))
    elif mutation == "duplicate":
        endpoints = (endpoints[1], *endpoints[1:])
    elif mutation == "future":
        cutoff -= timedelta(seconds=1)
    else:
        endpoints = (endpoints[0].replace(tzinfo=None), *endpoints[1:])
    with pytest.raises(FeatureUnavailable, match="sessions|cutoff|UTC"):
        evaluate(kind, history, endpoints, cutoff)


@pytest.mark.parametrize("anchor", [date(2024, 1, 2), date(2024, 2, 1), date(2023, 12, 1)])
def test_anchor_requires_exact_preceding_twelve_months(anchor: date) -> None:
    endpoints = monthly_endpoints()
    with pytest.raises(FeatureUnavailable, match="month"):
        skip_month_momentum(missing_history(endpoints), endpoints, anchor, endpoints[-1])


def test_missing_month_cannot_be_compressed_into_available_rows() -> None:
    endpoints = monthly_endpoints()
    endpoints = (datetime(2022, 12, 28, tzinfo=UTC), *endpoints[1:])
    with pytest.raises(FeatureUnavailable, match="consecutive.*month"):
        skip_month_momentum(missing_history(endpoints), endpoints, date(2024, 1, 1), endpoints[-1])


def test_year_boundary_and_no_anchor_price_reach_required_window_check() -> None:
    endpoints = monthly_endpoints()
    # The skipped anchor month has no close or availability dependency.
    with pytest.raises(FeatureUnavailable, match="finite positive price"):
        skip_month_momentum(missing_history(endpoints), endpoints, date(2024, 1, 1), endpoints[-1])


@pytest.mark.parametrize("kind", ["momentum", "reversal"])
@pytest.mark.parametrize(
    "mutation", ["missing_close", "unknown_clock", "future_clock", "identity", "basis"]
)
def test_incomplete_source_window_is_refused(kind: str, mutation: str) -> None:
    endpoints = monthly_endpoints() if kind == "momentum" else session_endpoints()
    history = missing_history(endpoints)
    if mutation == "missing_close":
        history = replace(history, closes=history.closes[:-1])
        reason = "every required session"
    elif mutation in {"unknown_clock", "future_clock"}:
        available = None if mutation == "unknown_clock" else endpoints[-1] + timedelta(days=1)
        history = replace(
            history,
            closes=(replace(history.closes[0], available_at=available), *history.closes[1:]),
        )
        reason = "availability"
    else:
        history = replace(history, **{"security_id" if mutation == "identity" else "basis": ""})
        reason = "identity.*basis"
    with pytest.raises(FeatureUnavailable, match=reason):
        evaluate(kind, history, endpoints, endpoints[-1])
