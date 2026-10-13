"""Trailing announcement residuals with explicit event and revision knowledge.

The source wrapper admits population completeness, event identity, calendar
endpoints and all contributing observations. These contracts preserve those
claims and refuse unresolved inputs; they do not supply historical admission.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import fmean

from liq.features.ranker_features import (
    FeatureUnavailable,
    FeatureValue,
    SourceIdentity,
    require_utc,
)


@dataclass(frozen=True, slots=True)
class ResidualVersion:
    """Persisted unit-beta residual in pp, with immutable dependency provenance."""

    source: SourceIdentity
    residual_pp: float | None
    window_start: datetime
    window_end: datetime
    price_basis: str
    available_at: datetime | None
    contributing_available_at: tuple[datetime | None, ...]
    contributing_sha256: tuple[str, ...]
    code_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        require_utc(self.window_start)
        require_utc(self.window_end)
        if self.available_at is not None:
            require_utc(self.available_at)
        for timestamp in self.contributing_available_at:
            if timestamp is not None:
                require_utc(timestamp)
        hashes = (*self.contributing_sha256, self.code_sha256, self.config_sha256)
        if (
            not self.price_basis.strip()
            or not self.contributing_sha256
            or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in hashes)
        ):
            raise FeatureUnavailable("residual basis and complete provenance are required")


@dataclass(frozen=True, slots=True)
class AnnouncementEvent:
    """One distinct source-admitted event; revisions do not create new events."""

    event_id: str
    security_id: str
    event_time: datetime
    available_at: datetime | None
    source: SourceIdentity
    versions: tuple[ResidualVersion, ...]

    def __post_init__(self) -> None:
        require_utc(self.event_time)
        if self.available_at is not None:
            require_utc(self.available_at)
        if not self.event_id.strip() or not self.security_id.strip():
            raise FeatureUnavailable("event and security identities are required")


@dataclass(frozen=True, slots=True)
class AnnouncementHistory:
    """Explicit population completeness and knowledge, independently of residuals."""

    security_id: str
    source: SourceIdentity
    complete_through: datetime
    available_at: datetime | None
    events: tuple[AnnouncementEvent, ...]

    def __post_init__(self) -> None:
        require_utc(self.complete_through)
        if self.available_at is not None:
            require_utc(self.available_at)


def _available_version(event: AnnouncementEvent, cutoff: datetime) -> ResidualVersion:
    """Choose the latest demonstrably known version, before inspecting its value."""
    identities = tuple(version.source.version for version in event.versions)
    if len(set(identities)) != len(identities):
        raise FeatureUnavailable(f"{event.event_id}: duplicate revision identity")
    if any(version.available_at is None for version in event.versions):
        raise FeatureUnavailable(f"{event.event_id}: unresolved version availability")
    known = tuple(
        (version.available_at, version)
        for version in event.versions
        if version.available_at is not None and version.available_at <= cutoff
    )
    if not known:
        raise FeatureUnavailable(f"{event.event_id}: no available version")
    clocks = tuple(clock for clock, _version in known)
    if len(set(clocks)) != len(clocks):
        raise FeatureUnavailable(f"{event.event_id}: tied revision availability")
    # Every known version has an evidenced clock; selection is independent of value.
    return max(known, key=lambda pair: pair[0])[1]


def trailing_four_announcement_mean(history: AnnouncementHistory, cutoff: datetime) -> FeatureValue:
    """Unweighted mean of the latest four distinct events' known residuals, in pp.

    Missing residuals remain in the event ordering. Ties, missing versions or
    unknown clocks make the feature unavailable; no older convenient substitute
    is selected. Execution at the measured opening price is a separate forbidden
    portfolio action, not granted by this scalar calculation.
    """
    require_utc(cutoff)
    if history.available_at is None or history.available_at > cutoff:
        raise FeatureUnavailable("population availability is unresolved or after cutoff")
    if history.complete_through < cutoff:
        raise FeatureUnavailable("population coverage does not reach cutoff")
    if not history.security_id.strip() or any(
        item.security_id != history.security_id for item in history.events
    ):
        raise FeatureUnavailable("population security identity is inconsistent")
    eligible = tuple(item for item in history.events if item.event_time <= cutoff)
    if len({item.event_id for item in eligible}) != len(eligible):
        raise FeatureUnavailable("duplicate event identity in population")
    if len(eligible) < 4:
        raise FeatureUnavailable("four distinct preceding events are required")
    ordered = sorted(eligible, key=lambda item: item.event_time, reverse=True)
    selected = ordered[:4]
    if len({item.event_time for item in selected}) != 4 or (
        len(ordered) > 4 and ordered[3].event_time == ordered[4].event_time
    ):
        raise FeatureUnavailable("tied event ordering is unresolved")
    values: list[float] = []
    hashes = {history.source.sha256}
    available = history.available_at
    for item in selected:
        if item.available_at is None or not item.event_time <= item.available_at <= cutoff:
            raise FeatureUnavailable(f"{item.event_id}: event availability is unresolved")
        version = _available_version(item, cutoff)
        if not version.window_start < item.event_time < version.window_end:
            raise FeatureUnavailable(f"{item.event_id}: announcement window is incoherent")
        if version.available_at is None or version.available_at < max(
            version.window_end, item.available_at
        ):
            raise FeatureUnavailable(f"{item.event_id}: dependency availability is backdated")
        if len(version.contributing_available_at) < 5 or any(
            clock is None or clock > version.available_at
            for clock in version.contributing_available_at
        ):
            raise FeatureUnavailable(f"{item.event_id}: dependency availability is unresolved")
        value = version.residual_pp
        if value is None or isinstance(value, bool) or not isfinite(value):
            raise FeatureUnavailable(f"{item.event_id}: residual is unavailable or invalid")
        values.append(value)
        available = max(available, item.available_at, version.available_at)
        hashes.update((item.source.sha256, version.source.sha256, *version.contributing_sha256))
    return FeatureValue(fmean(values), available, tuple(sorted(hashes)))
