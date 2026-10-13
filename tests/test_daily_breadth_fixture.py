"""Exact two-series arithmetic; no historical S&P membership assertion."""

import fcntl
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from liq.data.alpaca_daily_fixture import DailyFixture
from liq.data.alpaca_daily_regression import read_daily_pair
from liq.data.lockbox import CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1, LockboxGuard
from liq.features.ranker_features import (
    CloseObservation,
    CohortSnapshot,
    FeatureUnavailable,
    PriceHistory,
    SourceIdentity,
    complete_cohort_breadth,
)

WORKSPACE = Path(__file__).resolve().parents[2]
EVIDENCE = (
    WORKSPACE
    / "liq-experiments/artifacts/intraday_slate/classical_cross_sectional_ranker"
    / "fixed-pair-breadth-engineering-2026-10-03"
)
REFERENCE_SHA256 = "1d8462f40b1d742ce5a9e0540f53829180555140e95532a0766c5d4ed3bea014"


def read_case(
    contract_raw: bytes,
) -> tuple[tuple[DailyFixture, DailyFixture], tuple[datetime, ...]]:
    """Charge one of three exact same-object reads before opening price pages."""
    contract = json.loads(contract_raw)
    contract_hash = hashlib.sha256(contract_raw).hexdigest()
    for name, digest in contract["files"].items():
        assert hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() == digest
    graph = WORKSPACE / contract["graph_path"]
    raw = graph.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == contract["graph_sha256"]
    labels = tuple(datetime.fromisoformat(value) for value in contract["labels"])
    assert tuple(json.loads(raw)["branches"]["D_daily"]["expected_utc_bar_start_labels"]) == tuple(
        contract["labels"]
    )
    assert len(labels) == 61
    location = os.environ["RANKER_BREADTH_FIXTURE"]
    assert location == contract["fixture_root"]
    ledger = Path(contract["usage_log"])
    assert not ledger.is_symlink()
    descriptor = os.open(ledger, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as usage:
        metadata = os.fstat(usage.fileno())
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_nlink == 1
        fcntl.flock(usage, fcntl.LOCK_EX)
        raw_usage = usage.read(32_768)
        assert len(raw_usage) < 32_768
        rows = [json.loads(line) for line in raw_usage.splitlines()]
        assert all(row["contract_sha256"] == contract_hash for row in rows)
        assert len(rows) < len(contract["invocations"]) == 3
        invocation = contract["invocations"][len(rows)]
        assert os.environ["RANKER_BREADTH_INVOCATION"] == invocation
        LockboxGuard(
            Path(contract["lockbox_usage_log"]),
            CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1,
        ).assert_period_allowed(
            "alpaca_ranker_daily_fixture",
            labels[0].date(),
            labels[-1].date(),
            purpose="dev_smoke",
            arm_id="classical_cross_sectional_ranker_fixed_pair_breadth",
        )
        usage.write(
            json.dumps(
                {
                    "contract_sha256": contract_hash,
                    "invocation": invocation,
                    "purpose": "DEV_SMOKE_ONLY",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "planned_decoded_observations": 122,
                    "planned_arithmetic_observations": 120,
                    "start": labels[0].isoformat(),
                    "end": labels[-1].isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        usage.flush()
        os.fsync(usage.fileno())
        pair = read_daily_pair(Path(location), labels)
    return pair, labels


@pytest.fixture(scope="module")
def breadth_case():
    if "RANKER_BREADTH_FIXTURE" not in os.environ:
        pytest.skip("exact admitted private two-series breadth fixture not supplied")
    raw = (EVIDENCE / "reference.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256
    reference = json.loads(raw)
    contract_raw = (EVIDENCE / "contract.json").read_bytes()
    assert hashlib.sha256(contract_raw).hexdigest() == reference["contract_sha256"]
    contract = json.loads(contract_raw)
    pair, labels = read_case(contract_raw)
    available = datetime.fromisoformat(contract["population_available_at"])
    cohort = CohortSnapshot(
        ("KHC", "SPY"),
        SourceIdentity(
            "exact two captured series; engineering only", "frozen", reference["contract_sha256"]
        ),
        labels[-1],
        available,
    )
    histories = tuple(
        PriceHistory(
            fixture.request.symbol,
            SourceIdentity(
                "Alpaca SIP " + fixture.request.symbol, "current-vintage", fixture.source_sha256
            ),
            "all",
            tuple(
                CloseObservation(label, float(price), fixture.available_at)
                for label, price in zip(labels[-60:], fixture.closes[-60:], strict=True)
            ),
        )
        for fixture in pair
    )
    return cohort, histories, labels[-60:], available, reference


def test_exact_real_pair_matches_frozen_decimal_reference(breadth_case) -> None:
    cohort, histories, labels, cutoff, reference = breadth_case
    actual = complete_cohort_breadth(cohort, histories, labels, cutoff)
    assert actual.breadth.value == float(reference["percentage"])
    assert actual.enabled is reference["enabled"]
    assert actual.breadth.available_at == cutoff
    assert actual.breadth.source_hashes == tuple(
        sorted({cohort.source.sha256, *(history.source.sha256 for history in histories)})
    )


def test_engineering_population_is_unavailable_at_historical_cutoff(breadth_case) -> None:
    cohort, histories, labels, _, _ = breadth_case
    with pytest.raises(FeatureUnavailable, match="cohort availability"):
        complete_cohort_breadth(cohort, histories, labels, labels[-1])


def test_missing_real_member_does_not_shrink_denominator(breadth_case) -> None:
    cohort, histories, labels, cutoff, _ = breadth_case
    with pytest.raises(FeatureUnavailable, match="exact unique cohort"):
        complete_cohort_breadth(cohort, histories[:1], labels, cutoff)
