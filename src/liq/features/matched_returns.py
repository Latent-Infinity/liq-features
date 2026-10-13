"""Unit-beta endpoint arithmetic, without event or historical-basis admission."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite


@dataclass(frozen=True)
class Endpoint:
    """Observed price at an aware comparison endpoint, in a labelled price basis."""

    timestamp: datetime
    price: float | None
    basis: str


@dataclass(frozen=True)
class MatchedReturn:
    """Raw simple component returns and their unit-beta difference in pp."""

    security_return: float
    benchmark_return: float
    residual_pp: float


def matched_endpoint_return(
    security_start: Endpoint,
    security_end: Endpoint,
    benchmark_start: Endpoint,
    benchmark_end: Endpoint,
) -> MatchedReturn:
    """Refuse missing, nonpositive, unmatched or incoherent observations.

    A shared basis label is necessary, but is not evidence of causal corporate
    action treatment. Callers own source, calendar and availability admission.
    This scalar operation does not identify or admit announcement events.
    """
    points = (security_start, security_end, benchmark_start, benchmark_end)
    if any(point.timestamp.utcoffset() is None for point in points):
        raise ValueError("endpoints require timezone-aware timestamps")
    if (
        security_start.timestamp != benchmark_start.timestamp
        or security_end.timestamp != benchmark_end.timestamp
    ):
        raise ValueError("security and benchmark endpoints must be matched")
    if security_start.timestamp >= security_end.timestamp:
        raise ValueError("start endpoint must precede end endpoint")
    if any(not point.basis.strip() or point.basis != security_start.basis for point in points):
        raise ValueError("endpoints require a common nonempty price basis")
    prices: list[float] = []
    for point in points:
        price = point.price
        if price is None or isinstance(price, bool) or not isfinite(price) or price <= 0:
            raise ValueError("observations must be finite positive prices")
        prices.append(price)
    security_return = prices[1] / prices[0] - 1
    benchmark_return = prices[3] / prices[2] - 1
    residual_pp = 100 * (security_return - benchmark_return)
    if not all(isfinite(value) for value in (security_return, benchmark_return, residual_pp)):
        raise ValueError("returns must be finite")
    return MatchedReturn(security_return, benchmark_return, residual_pp)
