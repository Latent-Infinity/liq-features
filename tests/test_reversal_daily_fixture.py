"""One bounded assertion on the independently admitted current-vintage page."""

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from liq.features.ranker_features import (
    CloseObservation,
    FeatureUnavailable,
    PriceHistory,
    SourceIdentity,
    reversal_21_sessions,
)


@pytest.fixture(scope="module")
def reversal_case():
    location = os.environ.get("RANKER_DAILY_FIXTURE")
    if location is None:
        pytest.skip("exact private daily case not supplied")
    from liq.data.alpaca_daily_regression import read_daily_pair

    artifacts = (
        Path(__file__).resolve().parents[2]
        / "liq-experiments/artifacts/intraday_slate/classical_cross_sectional_ranker"
    )
    graph = (artifacts / "finite-daily-fixture-proposal-revised-2026-10-03.json").read_bytes()
    assert (
        hashlib.sha256(graph).hexdigest()
        == "6fd8067f323a962782d8c699122847e1f771f1448f04b93c0d532caf68029cd9"
    )
    labels = tuple(
        datetime.fromisoformat(ts)
        for ts in json.loads(graph)["branches"]["D_daily"]["expected_utc_bar_start_labels"]
    )
    raw_reference = (artifacts / "reversal-reference-2026-10-03.json").read_bytes()
    assert (
        hashlib.sha256(raw_reference).hexdigest()
        == "56e4e5d9b44f5a3753f7c444e93f9b3fb3920e46f4d09d474b5754e78913126f"
    )
    reference = json.loads(raw_reference)
    khc, _ = read_daily_pair(Path(location), labels)
    endpoints = labels[-22:]
    assert len(endpoints) == 22
    assert endpoints[0].isoformat() == "2023-03-23T04:00:00+00:00"
    assert endpoints[-1].isoformat() == "2023-04-24T04:00:00+00:00"
    assert reference["source_sha256"] == khc.source_sha256
    assert reference["labels"] == [ts.isoformat() for ts in endpoints]
    history = PriceHistory(
        "KHC",
        SourceIdentity("Alpaca SIP KHC", "current-vintage asof2023-04-24", khc.source_sha256),
        "all",
        tuple(
            CloseObservation(ts, float(price), khc.available_at)
            for ts, price in zip(endpoints, khc.closes[-22:], strict=True)
        ),
    )
    return history, endpoints, khc.available_at, Decimal(reference["value"])


def test_reversal_matches_independent_frozen_decimal_reference(reversal_case) -> None:
    history, endpoints, cutoff, reference = reversal_case
    actual = reversal_21_sessions(history, endpoints, cutoff)
    assert abs(Decimal(str(actual.value)) - reference) <= Decimal("1e-12")
    assert actual == reversal_21_sessions(history, endpoints, cutoff)
    assert actual.available_at == cutoff
    assert actual.source_hashes == (history.source.sha256,)


def test_reversal_current_capture_refuses_historical_cutoff(reversal_case) -> None:
    history, endpoints, _, _ = reversal_case
    with pytest.raises(FeatureUnavailable, match="availability"):
        reversal_21_sessions(
            history, endpoints, datetime.fromisoformat("2023-04-24T05:00:00+00:00")
        )


@pytest.mark.parametrize("index", [0, 10, 21])
def test_reversal_requires_every_real_window_close(reversal_case, index: int) -> None:
    history, endpoints, cutoff, _ = reversal_case
    missing = replace(history, closes=history.closes[:index] + history.closes[index + 1 :])
    with pytest.raises(FeatureUnavailable, match="every required session"):
        reversal_21_sessions(missing, endpoints, cutoff)
