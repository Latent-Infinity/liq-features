"""Fixed ranker arithmetic on complete, explicitly supplied source windows.

These kernels do not admit a calendar, historical cohort or price basis. Callers
must supply those contracts and honest observation availability. Retrieved-vintage
inputs cannot be used at a historical cutoff preceding their availability.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from math import isfinite
from statistics import stdev

from liq.features.matched_returns import Endpoint, matched_endpoint_return


class FeatureUnavailable(ValueError):
    """A required input is absent, ambiguous or unavailable at the cutoff."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def require_utc(timestamp: datetime) -> None:
    """Reject implicit timezone conversion at this boundary."""
    if timestamp.utcoffset() != timedelta(0):
        raise FeatureUnavailable("timestamps require explicit UTC")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Immutable source version, not an assertion of historical admission."""

    source_id: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.source_id.strip()
            or not self.version.strip()
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            raise FeatureUnavailable("source identifier, version and SHA256 are required")


@dataclass(frozen=True, slots=True)
class CloseObservation:
    """A close at a source-admitted endpoint with its evidenced availability."""

    timestamp: datetime
    price: float | None
    available_at: datetime | None

    def __post_init__(self) -> None:
        require_utc(self.timestamp)
        if self.available_at is not None:
            require_utc(self.available_at)
            if self.available_at < self.timestamp:
                raise FeatureUnavailable("close availability precedes observation")


@dataclass(frozen=True, slots=True)
class PriceHistory:
    """One identified security's exact window in a declared common basis."""

    security_id: str
    source: SourceIdentity
    basis: str
    closes: tuple[CloseObservation, ...]


@dataclass(frozen=True, slots=True)
class CohortSnapshot:
    """Source-admitted membership at the observation, with a separate knowledge clock."""

    members: tuple[str, ...]
    source: SourceIdentity
    observation_time: datetime
    available_at: datetime | None

    def __post_init__(self) -> None:
        require_utc(self.observation_time)
        if self.available_at is not None:
            require_utc(self.available_at)


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """An arithmetic result and dependency clock; no research verdict."""

    value: float
    available_at: datetime
    source_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BreadthValue:
    """Observed breadth and its fixed gate; unavailable is an exception."""

    breadth: FeatureValue
    enabled: bool


def _sessions(sessions: tuple[datetime, ...], count: int, cutoff: datetime) -> None:
    require_utc(cutoff)
    if len(sessions) != count:
        raise FeatureUnavailable(f"exactly {count} sessions are required")
    for timestamp in sessions:
        require_utc(timestamp)
    if any(left >= right for left, right in pairwise(sessions)):
        raise FeatureUnavailable("sessions must be unique and increasing")
    if sessions[-1] > cutoff:
        raise FeatureUnavailable("required session is after cutoff")


def _window(
    history: PriceHistory, sessions: tuple[datetime, ...], cutoff: datetime
) -> tuple[tuple[float, ...], datetime]:
    if not history.security_id.strip() or not history.basis.strip():
        raise FeatureUnavailable("security identity and price basis are required")
    if tuple(close.timestamp for close in history.closes) != sessions:
        raise FeatureUnavailable("history must match every required session exactly")
    prices: list[float] = []
    available = sessions[-1]
    for close in history.closes:
        if close.available_at is None or close.available_at > cutoff:
            raise FeatureUnavailable("close availability is unresolved or after cutoff")
        price = close.price
        if price is None or isinstance(price, bool) or not isfinite(price) or price <= 0:
            raise FeatureUnavailable("every required close needs a finite positive price")
        available = max(available, close.available_at)
        prices.append(price)
    return tuple(prices), available


def _endpoint_feature(
    history: PriceHistory, sessions: tuple[datetime, ...], cutoff: datetime, *, contrarian: bool
) -> FeatureValue:
    prices, available = _window(history, sessions, cutoff)
    value = prices[-1] / prices[0] - 1
    if not isfinite(value):
        raise FeatureUnavailable("endpoint return must be finite")
    return FeatureValue(-value if contrarian else value, available, (history.source.sha256,))


def skip_month_momentum(
    history: PriceHistory,
    monthly_endpoints: tuple[datetime, ...],
    anchor_month: date,
    cutoff: datetime,
) -> FeatureValue:
    """Return price[t-1]/price[t-12]-1, with no price dependency on month t.

    Anchor metadata is a first-day date. All twelve required UTC months must be
    complete and available; the caller proves actual calendar month-end identity.
    This conservative window convention differs from legacy endpoint null checks.
    """
    _sessions(monthly_endpoints, 12, cutoff)
    if type(anchor_month) is not date or anchor_month.day != 1:
        raise FeatureUnavailable("anchor month requires first-day date metadata")
    anchor = anchor_month.year * 12 + anchor_month.month - 1
    months = tuple(ts.year * 12 + ts.month - 1 for ts in monthly_endpoints)
    if months != tuple(range(anchor - 12, anchor)):
        raise FeatureUnavailable(
            "exactly twelve consecutive months preceding anchor month required"
        )
    return _endpoint_feature(history, monthly_endpoints, cutoff, contrarian=False)


def reversal_21_sessions(
    history: PriceHistory, sessions: tuple[datetime, ...], cutoff: datetime
) -> FeatureValue:
    """Negative simple return across exactly twenty-one adjacent session intervals.

    All twenty-two closes must be available. Ordered timestamps alone do not prove
    session adjacency; the owning loader supplies the admitted calendar contract.
    """
    _sessions(sessions, 22, cutoff)
    return _endpoint_feature(history, sessions, cutoff, contrarian=True)


def residual_low_volatility(
    security: PriceHistory,
    benchmark: PriceHistory,
    sessions: tuple[datetime, ...],
    cutoff: datetime,
) -> FeatureValue:
    """Negative ddof=1 standard deviation of exactly 60 unit-beta returns.

    Units are simple-return fractions, without annualization. The caller supplies
    all 61 consecutive session endpoints; no window is selected from price data.
    """
    _sessions(sessions, 61, cutoff)
    if security.basis != benchmark.basis:
        raise FeatureUnavailable("security and benchmark require a common price basis")
    security_prices, security_available = _window(security, sessions, cutoff)
    benchmark_prices, benchmark_available = _window(benchmark, sessions, cutoff)
    residuals = [
        matched_endpoint_return(
            Endpoint(sessions[i], security_prices[i], security.basis),
            Endpoint(sessions[i + 1], security_prices[i + 1], security.basis),
            Endpoint(sessions[i], benchmark_prices[i], benchmark.basis),
            Endpoint(sessions[i + 1], benchmark_prices[i + 1], benchmark.basis),
        ).residual_pp
        / 100
        for i in range(60)
    ]
    return FeatureValue(
        -stdev(residuals),
        max(security_available, benchmark_available),
        tuple(sorted({security.source.sha256, benchmark.source.sha256})),
    )


def complete_cohort_breadth(
    cohort: CohortSnapshot,
    histories: tuple[PriceHistory, ...],
    sessions: tuple[datetime, ...],
    cutoff: datetime,
) -> BreadthValue:
    """Require every member's 60 closes, then gate percent below maximum at 90.

    Cohort identity/availability and session continuity are source-admission
    obligations. Neither current survivors nor available-price names may define
    the denominator here. This function never carries a prior observation.
    """
    _sessions(sessions, 60, cutoff)
    if cohort.observation_time != sessions[-1]:
        raise FeatureUnavailable("cohort must identify the exact observation session")
    if cohort.available_at is None or cohort.available_at > cutoff:
        raise FeatureUnavailable("cohort availability is unresolved or after cutoff")
    if not cohort.members or any(not member.strip() for member in cohort.members):
        raise FeatureUnavailable("a nonempty identified cohort is required")
    identifiers = tuple(history.security_id for history in histories)
    if (
        len(set(cohort.members)) != len(cohort.members)
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(cohort.members)
    ):
        raise FeatureUnavailable("histories must cover the exact unique cohort")
    below = 0
    available = max(sessions[-1], cohort.available_at)
    for history in histories:
        prices, history_available = _window(history, sessions, cutoff)
        below += prices[-1] < max(prices)
        available = max(available, history_available)
    value = 100 * below / len(cohort.members)
    return BreadthValue(
        FeatureValue(
            value,
            available,
            tuple(sorted({cohort.source.sha256, *(item.source.sha256 for item in histories)})),
        ),
        value >= 90,
    )
