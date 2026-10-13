"""Contract tests; numerical regression requires the admitted private fixture."""

import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from liq.features.matched_returns import Endpoint, matched_endpoint_return

ADMITTED_FIXTURE_SHA256 = "47c9ae95699968fc76c3a1e4435338dfedd1a52d3dcdab507f2739199689abaa"
# Public receipt 32215567a91d3ea59e5e45082e9ec6210930b6ba5ab7c5532310b98d598db0ce:
# liq-experiments/artifacts/intraday_slate/classical_cross_sectional_ranker/
# matched-return-receipt-2026-10-03.json


def test_missing_observations_are_unavailable() -> None:
    start = Endpoint(datetime(2023, 4, 21, 13, 31, tzinfo=UTC), None, "raw")
    end = Endpoint(datetime(2023, 4, 21, 13, 32, tzinfo=UTC), None, "raw")
    with pytest.raises(ValueError, match="finite positive"):
        matched_endpoint_return(start, end, start, end)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_observations_are_refused(price: float) -> None:
    start = Endpoint(datetime(2023, 4, 21, 13, 31, tzinfo=UTC), price, "raw")
    end = Endpoint(datetime(2023, 4, 21, 13, 32, tzinfo=UTC), None, "raw")
    with pytest.raises(ValueError, match="finite positive"):
        matched_endpoint_return(start, end, start, end)


def test_unmatched_endpoints_are_refused_before_arithmetic() -> None:
    start = Endpoint(datetime(2023, 4, 21, 13, 31, tzinfo=UTC), None, "raw")
    end = Endpoint(datetime(2023, 4, 21, 13, 32, tzinfo=UTC), None, "raw")
    with pytest.raises(ValueError, match="matched"):
        matched_endpoint_return(start, end, end, end)


def test_nonincreasing_endpoints_are_refused() -> None:
    point = Endpoint(datetime(2023, 4, 21, 13, 31, tzinfo=UTC), None, "raw")
    with pytest.raises(ValueError, match="precede"):
        matched_endpoint_return(point, point, point, point)


def test_naive_timestamps_are_refused() -> None:
    point = Endpoint(datetime(2023, 4, 21, 13, 31), None, "raw")
    with pytest.raises(ValueError, match="timezone"):
        matched_endpoint_return(point, point, point, point)


@pytest.mark.parametrize("basis", ["", "all"])
def test_incoherent_price_basis_is_refused(basis: str) -> None:
    start = Endpoint(datetime(2023, 4, 21, 13, 31, tzinfo=UTC), None, "raw")
    end = Endpoint(datetime(2023, 4, 21, 13, 32, tzinfo=UTC), None, basis)
    with pytest.raises(ValueError, match="basis"):
        matched_endpoint_return(start, end, start, end)


def test_admitted_private_return_fixture() -> None:
    """The same frozen four observations; no replacement data when unavailable."""
    location = os.environ.get("RANKER_RETURN_FIXTURE")
    if location is None:
        pytest.skip("admitted private Alpaca return fixture is unavailable")
    path = Path(location)
    assert not path.is_symlink()
    assert path.stat().st_mode & 0o777 == 0o600
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ADMITTED_FIXTURE_SHA256
    fixture = json.loads(raw)
    prices = [Decimal(value) for symbol in ("KHC", "SPY") for value in fixture[symbol]]
    with localcontext() as context:
        context.prec = 50
        expected_security = prices[1] / prices[0] - 1
        expected_benchmark = prices[3] / prices[2] - 1
        expected = 100 * (expected_security - expected_benchmark)
    starts = datetime(2023, 4, 21, 13, 31, tzinfo=UTC)
    ends = datetime(2023, 4, 21, 13, 32, tzinfo=UTC)
    points = [
        Endpoint(ts, float(price), "raw")
        for ts, price in zip((starts, ends, starts, ends), prices, strict=True)
    ]
    actual = matched_endpoint_return(*points)
    assert actual.security_return == pytest.approx(float(expected_security), abs=1e-12, rel=0)
    assert actual.benchmark_return == pytest.approx(float(expected_benchmark), abs=1e-12, rel=0)
    assert actual.residual_pp == pytest.approx(float(expected), abs=1e-12, rel=0)
