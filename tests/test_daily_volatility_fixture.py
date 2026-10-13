"""The admitted current-vintage case; no historical-session or trading claim."""

import hashlib
import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from liq.features.ranker_features import (
    CloseObservation,
    FeatureUnavailable,
    PriceHistory,
    SourceIdentity,
    residual_low_volatility,
)


@pytest.fixture(scope="module")
def daily_case() -> tuple[PriceHistory, PriceHistory, tuple[datetime, ...], datetime]:
    location = os.environ.get("RANKER_DAILY_FIXTURE")
    if location is None:
        pytest.skip("exact admitted private daily fixture not supplied")
    from liq.data.alpaca_daily_regression import read_daily_pair

    graph = (
        Path(__file__).resolve().parents[2]
        / "liq-experiments/artifacts/intraday_slate/classical_cross_sectional_ranker/finite-daily-fixture-proposal-revised-2026-10-03.json"
    )
    raw = graph.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "6fd8067f323a962782d8c699122847e1f771f1448f04b93c0d532caf68029cd9"
    )
    labels = tuple(
        datetime.fromisoformat(value)
        for value in json.loads(raw)["branches"]["D_daily"]["expected_utc_bar_start_labels"]
    )
    histories = []
    for fixture in read_daily_pair(Path(location), labels):
        histories.append(
            PriceHistory(
                fixture.request.symbol,
                SourceIdentity(
                    "Alpaca SIP " + fixture.request.symbol,
                    "current-vintage asof2023-04-24",
                    fixture.source_sha256,
                ),
                "all",
                tuple(
                    CloseObservation(label, float(price), fixture.available_at)
                    for label, price in zip(labels, fixture.closes, strict=True)
                ),
            )
        )
    cutoff = datetime.fromisoformat("2026-10-03T06:41:16.536665+00:00")
    return histories[0], histories[1], labels, cutoff


def test_frozen_daily_volatility_matches_original_decimal_reference(daily_case) -> None:
    security, benchmark, labels, cutoff = daily_case
    actual = residual_low_volatility(security, benchmark, labels, cutoff)
    reference = Decimal("-0.010740405417536178266824487943737912992523954466655")
    assert abs(Decimal(str(actual.value)) - reference) <= Decimal("1e-12")
    assert actual == residual_low_volatility(security, benchmark, labels, cutoff)
    assert actual.available_at == cutoff


def test_current_capture_cannot_be_used_at_historical_cutoff(daily_case) -> None:
    security, benchmark, labels, _ = daily_case
    with pytest.raises(FeatureUnavailable, match="cutoff"):
        residual_low_volatility(
            security, benchmark, labels, datetime.fromisoformat("2023-04-24T05:00:00+00:00")
        )


def test_missing_real_terminal_observation_is_unavailable(daily_case) -> None:
    security, benchmark, labels, cutoff = daily_case
    incomplete = PriceHistory(
        security.security_id, security.source, security.basis, security.closes[:-1]
    )
    with pytest.raises(FeatureUnavailable, match="every required session"):
        residual_low_volatility(incomplete, benchmark, labels, cutoff)
