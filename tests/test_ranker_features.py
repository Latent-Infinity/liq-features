"""Source-free refusal tests, not numerical or historical admission evidence."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from liq.features.ranker_features import (
    CloseObservation,
    CohortSnapshot,
    FeatureUnavailable,
    PriceHistory,
    SourceIdentity,
    complete_cohort_breadth,
    residual_low_volatility,
)


def instants(count: int) -> tuple[datetime, ...]:
    """Ordered contract timestamps only; no asserted exchange calendar."""
    return tuple(datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(count))


def missing_history(security_id: str, count: int) -> PriceHistory:
    return PriceHistory(
        security_id,
        SourceIdentity("contract-test", "v1", "0" * 64),
        "unadmitted-test-basis",
        tuple(CloseObservation(ts, None, ts) for ts in instants(count)),
    )


def cohort(members: tuple[str, ...]) -> CohortSnapshot:
    return CohortSnapshot(
        members, SourceIdentity("contract-test", "v1", "0" * 64), instants(60)[-1], instants(60)[-1]
    )


def test_low_volatility_requires_sixty_returns() -> None:
    # Given an empty source-free request.
    history = missing_history("security", 0)
    # When / Then: a shorter window cannot emit a score.
    with pytest.raises(FeatureUnavailable, match="61"):
        residual_low_volatility(history, history, (), instants(1)[0])


def test_low_volatility_refuses_missing_prices() -> None:
    # Given all required timestamps, but no financial observations.
    history = missing_history("security", 61)
    # When / Then: completeness is checked before arithmetic.
    with pytest.raises(FeatureUnavailable, match="price"):
        residual_low_volatility(history, history, instants(61), instants(61)[-1])


def test_breadth_refuses_empty_cohort() -> None:
    # Given no independently supplied cohort.
    # When / Then: missing breadth cannot become an observed off regime.
    with pytest.raises(FeatureUnavailable, match="nonempty"):
        complete_cohort_breadth(cohort(()), (), instants(60), instants(60)[-1])


def test_breadth_refuses_reduced_denominator() -> None:
    # Given one unresolved member of a two-member contract, without prices.
    histories = (missing_history("first", 60),)
    # When / Then: available names cannot replace the registered cohort.
    with pytest.raises(FeatureUnavailable, match="cohort"):
        complete_cohort_breadth(
            cohort(("first", "missing")), histories, instants(60), instants(60)[-1]
        )


def test_breadth_refuses_missing_member_prices() -> None:
    # Given an exact identity set but no price observations.
    histories = (missing_history("first", 60),)
    # When / Then: a missing predicate cannot become false.
    with pytest.raises(FeatureUnavailable, match="price"):
        complete_cohort_breadth(cohort(("first",)), histories, instants(60), instants(60)[-1])


@pytest.mark.parametrize("sessions", [instants(60), tuple(reversed(instants(61)))])
def test_low_volatility_refuses_inexact_session_contract(sessions: tuple[datetime, ...]) -> None:
    # Given a missing or reversed timestamp contract.
    history = missing_history("security", 61)
    # When / Then: the kernel never selects a convenient replacement window.
    with pytest.raises(FeatureUnavailable, match="sessions"):
        residual_low_volatility(history, history, sessions, instants(61)[-1])


def test_future_session_is_refused_before_prices() -> None:
    # Given a requested endpoint after the decision cutoff.
    history = missing_history("security", 61)
    # When / Then: even absent values do not bypass clock validation.
    with pytest.raises(FeatureUnavailable, match="cutoff"):
        residual_low_volatility(history, history, instants(61), instants(61)[-2])


@pytest.mark.parametrize(
    "source,version,digest", [("", "v1", "0" * 64), ("s", "", "0" * 64), ("s", "v1", "bad")]
)
def test_source_identity_requires_version_and_hash(source: str, version: str, digest: str) -> None:
    # Given incomplete nonfinancial provenance.
    # When / Then: no validated source identity can be constructed.
    with pytest.raises(FeatureUnavailable, match="source"):
        SourceIdentity(source, version, digest)


def test_close_refuses_backdated_availability() -> None:
    # Given timestamp metadata only, with availability before observation.
    earlier, later = instants(2)
    # When / Then: no observed close can claim that clock.
    with pytest.raises(FeatureUnavailable, match="availability"):
        CloseObservation(later, None, earlier)


def test_close_requires_utc() -> None:
    # Given a naive timestamp, without a price.
    # When / Then: implicit timezone assumptions are refused.
    with pytest.raises(FeatureUnavailable, match="UTC"):
        CloseObservation(datetime(2023, 1, 1), None, None)


def test_cohort_availability_cannot_be_backdated() -> None:
    # Given identity metadata unavailable at the requested observation cutoff.
    snapshot = CohortSnapshot(
        ("first",),
        SourceIdentity("contract-test", "v1", "0" * 64),
        instants(60)[-1],
        instants(61)[-1],
    )
    # When / Then: complete price identities cannot override cohort knowledge.
    with pytest.raises(FeatureUnavailable, match="cohort availability"):
        complete_cohort_breadth(
            snapshot, (missing_history("first", 60),), instants(60), instants(60)[-1]
        )


@pytest.mark.parametrize("available", [None, instants(62)[-1]])
def test_unknown_or_future_close_availability_is_refused(available: datetime | None) -> None:
    # Given clock metadata only; values are deliberately absent.
    history = missing_history("security", 61)
    first = replace(history.closes[0], available_at=available)
    history = replace(history, closes=(first, *history.closes[1:]))
    # When / Then: current retrieval cannot masquerade as historical knowledge.
    with pytest.raises(FeatureUnavailable, match="availability"):
        residual_low_volatility(history, history, instants(61), instants(61)[-1])


@pytest.mark.parametrize("security_id,basis", [("", "basis"), ("security", "")])
def test_missing_history_identity_or_basis_is_refused(security_id: str, basis: str) -> None:
    # Given an incomplete nonfinancial source contract.
    history = replace(missing_history(security_id, 61), basis=basis)
    # When / Then: metadata is required before arithmetic.
    with pytest.raises(FeatureUnavailable, match="identity.*basis"):
        residual_low_volatility(history, history, instants(61), instants(61)[-1])


def test_incompatible_security_benchmark_basis_is_refused() -> None:
    # Given differing basis labels, without financial values.
    history = missing_history("security", 61)
    benchmark = replace(history, basis="different")
    # When / Then: a residual cannot mix bases.
    with pytest.raises(FeatureUnavailable, match="common price basis"):
        residual_low_volatility(history, benchmark, instants(61), instants(61)[-1])


def test_missing_session_cannot_be_skipped() -> None:
    # Given 60 timestamp placeholders for a required 61-endpoint window.
    history = missing_history("security", 60)
    # When / Then: available rows cannot choose the window.
    with pytest.raises(FeatureUnavailable, match="every required session"):
        residual_low_volatility(history, history, instants(61), instants(61)[-1])


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_price_tokens_are_refused(price: float) -> None:
    # Given a malformed token, not a fabricated market observation.
    history = missing_history("security", 61)
    history = replace(
        history, closes=(replace(history.closes[0], price=price), *history.closes[1:])
    )
    # When / Then: malformed tokens never enter return arithmetic.
    with pytest.raises(FeatureUnavailable, match="finite positive price"):
        residual_low_volatility(history, history, instants(61), instants(61)[-1])


def test_breadth_cannot_carry_an_older_cohort_snapshot() -> None:
    # Given the correct names attached to an earlier observation session.
    snapshot = replace(cohort(("first",)), observation_time=instants(60)[-2])
    # When / Then: the missing current snapshot is not carried forward.
    with pytest.raises(FeatureUnavailable, match="exact observation session"):
        complete_cohort_breadth(
            snapshot, (missing_history("first", 60),), instants(60), instants(60)[-1]
        )


def test_unknown_cohort_availability_is_unavailable() -> None:
    # Given unresolved knowledge time, without prices.
    snapshot = replace(cohort(("first",)), available_at=None)
    # When / Then: missing provenance cannot produce observed breadth.
    with pytest.raises(FeatureUnavailable, match="cohort availability"):
        complete_cohort_breadth(snapshot, (), instants(60), instants(60)[-1])


@pytest.mark.parametrize("members", [("first", "first"), ("",)])
def test_duplicate_or_unidentified_cohort_members_are_refused(members: tuple[str, ...]) -> None:
    # Given a malformed identity contract, with no prices.
    # When / Then: no denominator is inferred or deduplicated.
    with pytest.raises(FeatureUnavailable, match="cohort"):
        complete_cohort_breadth(cohort(members), (), instants(60), instants(60)[-1])


def test_duplicate_histories_are_refused() -> None:
    # Given a duplicated member history, with no prices.
    history = missing_history("first", 60)
    # When / Then: duplicate rows cannot inflate the denominator.
    with pytest.raises(FeatureUnavailable, match="unique cohort"):
        complete_cohort_breadth(
            cohort(("first",)), (history, history), instants(60), instants(60)[-1]
        )
