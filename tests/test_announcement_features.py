"""Contract/refusal tests with absent residuals; no financial sample is fabricated."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from liq.features.announcement_features import (
    AnnouncementEvent,
    AnnouncementHistory,
    ResidualVersion,
    trailing_four_announcement_mean,
)
from liq.features.ranker_features import FeatureUnavailable, SourceIdentity

CUTOFF = datetime(2023, 1, 10, tzinfo=UTC)
SOURCE = SourceIdentity("contract-test", "v1", "0" * 64)


def event(index: int) -> AnnouncementEvent:
    timestamp = CUTOFF - timedelta(days=index)
    return AnnouncementEvent(f"contract-{index}", "security", timestamp, timestamp, SOURCE, ())


def history(events: tuple[AnnouncementEvent, ...]) -> AnnouncementHistory:
    return AnnouncementHistory("security", SOURCE, CUTOFF, CUTOFF, events)


def absent_version(item: AnnouncementEvent) -> ResidualVersion:
    return ResidualVersion(
        SOURCE,
        None,
        item.event_time - timedelta(minutes=1),
        item.event_time + timedelta(minutes=1),
        "unadmitted-test-basis",
        CUTOFF,
        (item.event_time,) * 5,
        (SOURCE.sha256,),
        "1" * 64,
        "2" * 64,
    )


def test_four_events_are_required() -> None:
    # Given three metadata records; when evaluated, a shorter mean is unavailable.
    with pytest.raises(FeatureUnavailable, match="four distinct"):
        trailing_four_announcement_mean(history(tuple(event(i) for i in range(1, 4))), CUTOFF)


def test_latest_missing_event_cannot_be_replaced_by_older_event() -> None:
    # Given five empty records; refusal must name the latest, before filtering values.
    with pytest.raises(FeatureUnavailable, match="contract-1.*version"):
        trailing_four_announcement_mean(history(tuple(event(i) for i in range(1, 6))), CUTOFF)


def test_future_event_does_not_supply_the_fourth_record() -> None:
    # Given three preceding records and one future record; cutoff determines population.
    with pytest.raises(FeatureUnavailable, match="four distinct"):
        trailing_four_announcement_mean(history(tuple(event(i) for i in (-1, 1, 2, 3))), CUTOFF)


def test_duplicate_event_identity_is_unresolved() -> None:
    records = (event(1), event(1), event(2), event(3), event(4))
    with pytest.raises(FeatureUnavailable, match="duplicate event"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_tied_latest_event_order_is_unresolved() -> None:
    records = (event(1), replace(event(2), event_time=event(1).event_time), event(3), event(4))
    with pytest.raises(FeatureUnavailable, match="tied event"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_tie_at_selection_boundary_is_unresolved() -> None:
    records = (
        *tuple(event(i) for i in range(1, 5)),
        replace(event(5), event_time=event(4).event_time),
    )
    with pytest.raises(FeatureUnavailable, match="tied event"):
        trailing_four_announcement_mean(history(records), CUTOFF)


@pytest.mark.parametrize("known", [None, CUTOFF + timedelta(seconds=1)])
def test_population_must_be_known_by_cutoff(known: datetime | None) -> None:
    request = replace(history(()), available_at=known)
    with pytest.raises(FeatureUnavailable, match="population availability"):
        trailing_four_announcement_mean(request, CUTOFF)


def test_population_must_cover_cutoff() -> None:
    request = replace(history(()), complete_through=CUTOFF - timedelta(seconds=1))
    with pytest.raises(FeatureUnavailable, match="population coverage"):
        trailing_four_announcement_mean(request, CUTOFF)


def test_events_must_belong_to_requested_security() -> None:
    request = history((replace(event(1), security_id="other"),))
    with pytest.raises(FeatureUnavailable, match="security identity"):
        trailing_four_announcement_mean(request, CUTOFF)


@pytest.mark.parametrize("known", [None, CUTOFF + timedelta(seconds=1)])
def test_selected_event_must_be_known_by_cutoff(known: datetime | None) -> None:
    records = (replace(event(1), available_at=known), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="contract-1.*availability"):
        trailing_four_announcement_mean(history(records), CUTOFF)


@pytest.mark.parametrize("known", [None, CUTOFF + timedelta(seconds=1)])
def test_unavailable_revision_cannot_be_used(known: datetime | None) -> None:
    latest = event(1)
    version = replace(absent_version(latest), available_at=known)
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="contract-1.*version"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_duplicate_revision_identity_is_unresolved() -> None:
    latest = event(1)
    version = absent_version(latest)
    records = (replace(latest, versions=(version, version)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="duplicate revision"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_equal_revision_availability_is_unresolved() -> None:
    latest = event(1)
    version = absent_version(latest)
    other = replace(version, source=SourceIdentity("contract-test", "v2", "3" * 64))
    records = (replace(latest, versions=(version, other)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="tied revision"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_missing_selected_residual_is_unavailable() -> None:
    latest = event(1)
    records = (
        replace(latest, versions=(absent_version(latest),)),
        *tuple(event(i) for i in range(2, 5)),
    )
    with pytest.raises(FeatureUnavailable, match="contract-1.*residual"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_revision_cannot_backdate_dependency_availability() -> None:
    latest = event(1)
    version = replace(
        absent_version(latest), contributing_available_at=(CUTOFF + timedelta(seconds=1),) * 5
    )
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="dependency availability"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_unresolved_dependency_availability_is_unavailable() -> None:
    latest = event(1)
    version = replace(absent_version(latest), contributing_available_at=(None,) * 5)
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="dependency availability"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_revision_window_must_surround_event_strictly() -> None:
    latest = event(1)
    version = replace(absent_version(latest), window_start=latest.event_time)
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="announcement window"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_event_requires_utc_clock() -> None:
    with pytest.raises(FeatureUnavailable, match="UTC"):
        replace(event(1), event_time=datetime(2023, 1, 1))


def test_revision_requires_producing_code_and_config_identities() -> None:
    with pytest.raises(FeatureUnavailable, match="provenance"):
        replace(absent_version(event(1)), code_sha256="")


@pytest.mark.parametrize("token", [float("nan"), float("inf"), float("-inf"), True])
def test_malformed_residual_tokens_are_refused(token: float) -> None:
    latest = event(1)
    version = replace(absent_version(latest), residual_pp=token)
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="residual.*invalid"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_feature_cannot_be_available_at_its_pre_event_endpoint() -> None:
    latest = event(1)
    version = replace(absent_version(latest), available_at=latest.event_time)
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="dependency availability.*backdated"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_price_and_event_dependency_clocks_are_required() -> None:
    latest = event(1)
    version = replace(absent_version(latest), contributing_available_at=(latest.event_time,) * 4)
    records = (replace(latest, versions=(version,)), *tuple(event(i) for i in range(2, 5)))
    with pytest.raises(FeatureUnavailable, match="dependency availability"):
        trailing_four_announcement_mean(history(records), CUTOFF)


@pytest.mark.parametrize("basis,hashes", [("", (SOURCE.sha256,)), ("basis", ())])
def test_residual_basis_and_input_hashes_are_required(basis: str, hashes: tuple[str, ...]) -> None:
    with pytest.raises(FeatureUnavailable, match="provenance"):
        replace(absent_version(event(1)), price_basis=basis, contributing_sha256=hashes)


def test_event_cannot_claim_knowledge_before_announcement() -> None:
    latest = event(1)
    records = (
        replace(latest, available_at=latest.event_time - timedelta(seconds=1)),
        *tuple(event(i) for i in range(2, 5)),
    )
    with pytest.raises(FeatureUnavailable, match="event availability"):
        trailing_four_announcement_mean(history(records), CUTOFF)


def test_unidentified_event_is_refused() -> None:
    with pytest.raises(FeatureUnavailable, match="identities"):
        replace(event(1), event_id="")
