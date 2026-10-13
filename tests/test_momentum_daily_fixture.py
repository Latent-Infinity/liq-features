"""One real monthly momentum case; current-vintage engineering only."""

import hashlib
import importlib.util
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import polars as pl
import pytest
from pydantic import TypeAdapter

from liq.data.alpaca_momentum_fixture import MomentumDailyFixture
from liq.data.alpaca_momentum_numerical_models import Invocation, MomentumNumericalRefused
from liq.data.alpaca_momentum_numerical_runtime import ReferenceBinding
from liq.data.alpaca_momentum_numerical_scope import NumericalBinding, adopt_numerical_scope
from liq.data.alpaca_momentum_regression import read_momentum_case
from liq.data.lockbox import CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1, LockboxGuard
from liq.features.ranker_features import (
    CloseObservation,
    FeatureUnavailable,
    PriceHistory,
    SourceIdentity,
    skip_month_momentum,
)

WORKSPACE: Final = Path(__file__).resolve().parents[2]
EVIDENCE: Final = (
    WORKSPACE
    / "liq-experiments/artifacts/intraday_slate/classical_cross_sectional_ranker"
    / "momentum-numerical-engineering-2026-10-03"
)
SCOPE_ROOT: Final = WORKSPACE / ".omo/ranker-momentum-numerical-fixture-2026-10-03"
REFERENCE_SHA256 = "4190c53437641d2fdcfc7b168990622a9ca1f8ebd5cce173c759f01abc620f6d"
CANONICAL_PATH: Final = (
    WORKSPACE / "liq-experiments/src/liq/experiments/ranker_champions/champions.py"
)
CANONICAL_SHA256: Final = "ebc42b85faa615c47027be36c85108a4341ab9e4ca37bb0fd90dccd672ad8166"


def load_canonical() -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Import code only; invoke solely the fixed pure feature callable later."""
    if hashlib.sha256(CANONICAL_PATH.read_bytes()).hexdigest() != CANONICAL_SHA256:
        raise MomentumNumericalRefused("canonical_code")
    spec = importlib.util.spec_from_file_location("frozen_momentum_canonical", CANONICAL_PATH)
    if spec is None or spec.loader is None:
        raise MomentumNumericalRefused("canonical_loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.momentum_12_1


CANONICAL_MOMENTUM: Final = load_canonical()


@contextmanager
def scope_case(invocation: Invocation) -> Iterator[tuple[NumericalBinding, MomentumDailyFixture]]:
    expected = (
        os.environ["RANKER_MOMENTUM_CONTRACT_SHA256"],
        os.environ["RANKER_MOMENTUM_AUTHORITY_SHA256"],
    )
    with adopt_numerical_scope(
        SCOPE_ROOT / "contract.json", SCOPE_ROOT / "authority.json", expected, invocation
    ) as binding:
        if os.environ["RANKER_MOMENTUM_FIXTURE"] != binding.contract.fixture_root:
            raise MomentumNumericalRefused("fixture_root_environment")
        guard = LockboxGuard(
            Path(binding.contract.usage_root) / "numerical-lockbox-usage.jsonl",
            CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1,
        )
        fixture = read_momentum_case(binding, guard)
        yield binding, fixture


@dataclass(frozen=True, slots=True)
class MomentumCase:
    history: PriceHistory
    endpoints: tuple[datetime, ...]
    anchor: date
    cutoff: datetime
    expected: Decimal


@pytest.fixture(scope="module")
def momentum_case() -> Iterator[MomentumCase]:
    if "RANKER_MOMENTUM_FIXTURE" not in os.environ:
        pytest.skip("exact admitted private momentum numerical fixture not supplied")
    raw = (EVIDENCE / "reference.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256
    reference = ReferenceBinding.model_validate_json(raw)
    expected = Decimal(reference.value)
    assert expected.is_finite()
    invocation = TypeAdapter(Invocation).validate_python(os.environ["RANKER_MOMENTUM_INVOCATION"])
    assert invocation in {"focused", "full-feature"}
    with scope_case(invocation) as (binding, fixture):
        assert reference.contract_sha256 == binding.contract_sha256
        indexed = dict(zip(fixture.calendar.expected_labels, fixture.closes, strict=True))
        endpoints = fixture.calendar.monthly_endpoints
        history = PriceHistory(
            "KHC",
            SourceIdentity("Alpaca SIP KHC", "current-vintage", fixture.source_sha256),
            "all",
            tuple(
                CloseObservation(timestamp, float(indexed[timestamp]), fixture.available_at)
                for timestamp in endpoints
            ),
        )
        yield MomentumCase(
            history, endpoints, fixture.calendar.anchor_month, fixture.available_at, expected
        )


def test_real_case_owning_and_pure_canonical_match_independent_reference(momentum_case) -> None:
    case = momentum_case
    actual = skip_month_momentum(case.history, case.endpoints, case.anchor, case.cutoff)
    assert actual.value == pytest.approx(float(case.expected), rel=0, abs=1e-12)
    assert actual.available_at == case.cutoff
    assert actual.source_hashes == (case.history.source.sha256,)
    # The last row is an evaluation slot, not an observation or invented price.
    frame = pl.DataFrame(
        {
            "date": [*(timestamp.date() for timestamp in case.endpoints), case.anchor],
            "KHC": pl.Series(
                [*(close.price for close in case.history.closes), None], dtype=pl.Float64
            ),
        }
    )
    assert frame.height == 13 and frame["KHC"][-1] is None
    canonical = CANONICAL_MOMENTUM(frame)
    assert canonical["date"][-1] == case.anchor
    assert canonical["KHC"][-1] == pytest.approx(float(case.expected), rel=0, abs=1e-12)
    assert actual == skip_month_momentum(case.history, case.endpoints, case.anchor, case.cutoff)


def test_current_vintage_cannot_supply_historical_knowledge_cutoff(momentum_case) -> None:
    case = momentum_case
    with pytest.raises(FeatureUnavailable):
        skip_month_momentum(
            case.history, case.endpoints, case.anchor, datetime(2023, 4, 1, tzinfo=UTC)
        )


def test_removal_of_required_real_endpoint_is_unavailable(momentum_case) -> None:
    case = momentum_case
    incomplete = PriceHistory(
        case.history.security_id,
        case.history.source,
        case.history.basis,
        case.history.closes[:-1],
    )
    with pytest.raises(FeatureUnavailable):
        skip_month_momentum(incomplete, case.endpoints, case.anchor, case.cutoff)
