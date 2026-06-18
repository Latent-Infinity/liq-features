"""Target construction for volatility forecast training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from .contracts import ReasonCode

TARGET_DEFINITION = "target_rv_total"
TARGET_CONSTRUCTION_VERSION = "target_rv_total_v1"
INTRADAY_REVERSAL_TARGET_DEFINITION = "intraday_reversal_fixed_horizon_v1"


@dataclass(frozen=True)
class TargetRvTotal:
    """Realized variance target with row-level source provenance."""

    symbol: str
    target_session: date
    target_rv_total: float
    overnight_gap_var: float
    intraday_bar_rv: float
    target_start_ts: datetime
    target_end_ts: datetime
    raw_price_start: float
    raw_price_end: float
    adjustment_factor_start: float
    adjustment_factor_end: float
    corporate_action_event_id: str | None
    adjustment_as_of_ts: datetime | None
    promotion_eligible: bool
    reason_codes: tuple[ReasonCode, ...]
    construction_version: str = TARGET_CONSTRUCTION_VERSION


def _ts_expr() -> pl.Expr:
    if pl.__version__.startswith("0."):
        return pl.col("timestamp")
    return pl.col("timestamp")


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return None


def _session_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"target_session must be a date or datetime, got {type(value)!r}")


def _session_frame(bars: pl.DataFrame, target_session: date) -> pl.DataFrame:
    if "timestamp" not in bars.columns:
        raise ValueError("daily_bars_1m must contain a timestamp column")
    return (
        bars.with_columns(_ts_expr().dt.date().alias("_session_date"))
        .filter(pl.col("_session_date") == target_session)
        .sort("timestamp")
    )


def _previous_close(bars: pl.DataFrame, target_start_ts: datetime) -> float:
    previous = bars.filter(pl.col("timestamp") < target_start_ts).sort("timestamp")
    if previous.is_empty():
        raise ValueError("target construction requires at least one prior close")
    return float(previous.get_column("close")[-1])


def _intraday_rv(session: pl.DataFrame) -> float:
    opens = session.get_column("open").to_numpy().astype(float)
    closes = session.get_column("close").to_numpy().astype(float)
    if len(closes) == 0:
        raise ValueError("target session contains no bars")
    log_closes = [math.log(float(value)) for value in closes]
    returns = [log_closes[0] - math.log(float(opens[0]))]
    returns.extend(log_closes[idx] - log_closes[idx - 1] for idx in range(1, len(log_closes)))
    return float(sum(value * value for value in returns))


def _is_ca_adjacent(
    *, session: pl.DataFrame, target_session: date, ca_window_sessions: int
) -> tuple[bool, str | None, datetime | None]:
    if "corporate_action_effective_ts" not in session.columns:
        return False, None, None
    values = [
        value for value in session.get_column("corporate_action_effective_ts").to_list() if value
    ]
    if not values:
        return False, None, None
    event_id = (
        str(session.get_column("corporate_action_event_id")[0])
        if "corporate_action_event_id" in session.columns
        and session.get_column("corporate_action_event_id")[0] is not None
        else None
    )
    for value in values:
        effective_ts = _as_utc(value)
        if effective_ts is None:
            continue
        if abs((target_session - effective_ts.date()).days) <= ca_window_sessions:
            return True, event_id, effective_ts
    return False, event_id, _as_utc(values[0])


def build_target_rv_total(
    daily_bars_1m: pl.DataFrame,
    target_session: date | datetime,
    *,
    symbol: str,
    ca_window_sessions: int = 2,
) -> TargetRvTotal:
    """Build ``target_rv_total`` from real 1-minute bars.

    The total is the strict sum of overnight gap variance and intraday
    realized variance. Corporate-action-adjacent rows remain available
    for audit but are marked ineligible for promotion aggregates.
    """

    session_date = _session_date(target_session)
    required = {"timestamp", "open", "close"}
    missing = required.difference(daily_bars_1m.columns)
    if missing:
        raise ValueError(f"daily_bars_1m missing required columns: {sorted(missing)}")

    session = _session_frame(daily_bars_1m, session_date)
    if session.is_empty():
        raise ValueError(f"no bars for target_session={session_date.isoformat()}")

    target_start_ts = _as_utc(session.get_column("timestamp")[0])
    target_end_ts = _as_utc(session.get_column("timestamp")[-1])
    if target_start_ts is None or target_end_ts is None:
        raise ValueError("timestamp values must be datetimes")

    raw_price_start = float(session.get_column("open")[0])
    raw_price_end = float(session.get_column("close")[-1])
    previous_close = _previous_close(daily_bars_1m, target_start_ts)
    overnight_gap = math.log(raw_price_start) - math.log(previous_close)
    overnight_gap_var = float(overnight_gap * overnight_gap)
    intraday_bar_rv = _intraday_rv(session)
    target_rv_total = overnight_gap_var + intraday_bar_rv

    if not math.isclose(
        target_rv_total, overnight_gap_var + intraday_bar_rv, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError("target_rv_total additivity failed")

    is_ca_adjacent, ca_event_id, ca_effective_ts = _is_ca_adjacent(
        session=session,
        target_session=session_date,
        ca_window_sessions=ca_window_sessions,
    )
    reason_codes: tuple[ReasonCode, ...] = ()
    if is_ca_adjacent:
        reason_codes = (
            ReasonCode(
                code="CA_ADJACENT",
                stage="feature",
                severity="warning",
                source_object_id=ca_event_id,
                details={
                    "target_session": session_date.isoformat(),
                    "ca_window_sessions": ca_window_sessions,
                },
            ),
        )

    adjustment_factor_start = (
        float(session.get_column("adjustment_factor_start")[0])
        if "adjustment_factor_start" in session.columns
        else 1.0
    )
    adjustment_factor_end = (
        float(session.get_column("adjustment_factor_end")[-1])
        if "adjustment_factor_end" in session.columns
        else 1.0
    )
    adjustment_as_of_ts = (
        _as_utc(session.get_column("adjustment_as_of_ts")[-1])
        if "adjustment_as_of_ts" in session.columns
        else ca_effective_ts
    )

    return TargetRvTotal(
        symbol=symbol,
        target_session=session_date,
        target_rv_total=float(target_rv_total),
        overnight_gap_var=overnight_gap_var,
        intraday_bar_rv=intraday_bar_rv,
        target_start_ts=target_start_ts,
        target_end_ts=target_end_ts,
        raw_price_start=raw_price_start,
        raw_price_end=raw_price_end,
        adjustment_factor_start=adjustment_factor_start,
        adjustment_factor_end=adjustment_factor_end,
        corporate_action_event_id=ca_event_id,
        adjustment_as_of_ts=adjustment_as_of_ts,
        promotion_eligible=not is_ca_adjacent,
        reason_codes=reason_codes,
    )


@dataclass(frozen=True)
class IntradayReversalTarget:
    """FIXED-HORIZON intraday reversal target row.

    The target is a leakage-safe time-horizon window: the model is trained to
    forecast realized variance over ``[fill_ts, target_end_ts]`` where
    ``target_end_ts = min(fill_ts + horizon, next_close)``. ``is_path_dependent``
    is invariantly ``False``. REALIZED-EXIT labels live elsewhere and are
    never folded into pure forecast-model comparisons.
    """

    signal_id: str
    symbol: str
    signal_ts: datetime
    fill_ts: datetime
    target_start_ts: datetime
    target_end_ts: datetime
    horizon: timedelta
    next_close_ts: datetime
    is_path_dependent: bool
    target_definition: str
    target_construction_version: str
    reason_codes: tuple[ReasonCode, ...]


def build_intraday_reversal_target(
    *,
    signal_id: str,
    symbol: str,
    signal_ts: datetime,
    fill_ts: datetime,
    horizon: timedelta,
    next_close_ts: datetime,
    reason_codes: tuple[ReasonCode, ...] = (),
) -> IntradayReversalTarget:
    """Construct a FIXED-HORIZON intraday reversal target row.

    Enforces the intraday target invariants:

    - ``fill_ts >= signal_ts`` (target starts at fill, sized post-execution).
    - ``horizon`` strictly positive.
    - ``target_end_ts = min(fill_ts + horizon, next_close_ts)``: never past
      next close, so the row is leakage-safe.
    - ``is_path_dependent`` is invariantly ``False``.
    """

    if fill_ts < signal_ts:
        raise ValueError("fill_ts must not precede signal_ts")
    if horizon.total_seconds() <= 0:
        raise ValueError("horizon must be positive")
    if next_close_ts <= fill_ts:
        raise ValueError("next_close_ts must follow fill_ts")
    target_end_ts = min(fill_ts + horizon, next_close_ts)
    return IntradayReversalTarget(
        signal_id=signal_id,
        symbol=symbol,
        signal_ts=signal_ts,
        fill_ts=fill_ts,
        target_start_ts=fill_ts,
        target_end_ts=target_end_ts,
        horizon=horizon,
        next_close_ts=next_close_ts,
        is_path_dependent=False,
        target_definition=INTRADAY_REVERSAL_TARGET_DEFINITION,
        target_construction_version=INTRADAY_REVERSAL_TARGET_DEFINITION,
        reason_codes=reason_codes,
    )


__all__ = [
    "INTRADAY_REVERSAL_TARGET_DEFINITION",
    "IntradayReversalTarget",
    "TARGET_CONSTRUCTION_VERSION",
    "TARGET_DEFINITION",
    "TargetRvTotal",
    "build_intraday_reversal_target",
    "build_target_rv_total",
]
