"""Frozen policy + output dataclasses for the canonical risk-variance
estimator.

The shapes here mirror research plan §3.4 (the spec) and §3.6 (the
output object) verbatim. Every field that affects the emitted number
lives on ``VolEstimatorSpec``; nothing that changes the result is set
in prose. See ``[PHASE0_CONTRACT]`` for the canonical-target decision
(Option B — close-to-close risk variance) and ``[DECISIONS]`` for the
locked production defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

CalendarBasis = Literal["trading_time", "calendar_time", "session_time"]
OvernightHandling = Literal["fold_into_session", "separate_component", "exclude"]
WeekendHolidayPolicy = Literal["single_gap", "calendar_scaled", "exclude"]
PartialDayPolicy = Literal["scale_by_session_fraction", "exclude"]
HaltPolicy = Literal["carry_forward", "nan", "ctc_over_halt"]

SamplingInterval = Literal["1m", "5m", "15m"]
SessionScope = Literal["regular", "extended", "full"]
RVProxy = Literal["rv", "bpv", "realized_kernel"]

OutlierMethod = Literal["none", "past_rolling_z", "mad"]

BarTimestampSemantics = Literal["start_time", "end_time"]
DataLatencyPolicy = Literal["close_plus_0", "close_plus_delay", "next_open"]
CorporateActionPolicy = Literal[
    "raw_for_live",
    "adjusted_for_research",
    "point_in_time_adjusted",
]

EstimatorName = Literal[
    "ctc",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "gk_yang_zhang",
    "yang_zhang",
    "rv",
]
EstimatorTarget = Literal[
    "close_to_close_risk_variance",
    "continuous_intraday_variance",
    "quadratic_variation",
]
OutputUnit = Literal[
    "per_bar_variance",
    "annualized_variance",
    "per_bar_vol",
    "annualized_vol",
]
PriceBasis = Literal["raw", "split_adjusted", "point_in_time_split_adjusted"]

ComponentSource = Literal["daily_ohlc", "minute_rv", "derived", "unavailable"]
ComponentUnit = Literal["per_bar_variance", "annualized_variance"]


@dataclass(frozen=True)
class VolCalendarPolicy:
    """Calendar / annualization policy. Production defaults
    (`[DECISIONS]`): trading-time basis, separate-component overnight,
    single-gap weekend/holiday, session-fraction partial days,
    carry-forward halts, gap classification on.
    """

    annualization_basis: CalendarBasis = "trading_time"
    overnight_handling: OvernightHandling = "separate_component"
    weekend_holiday: WeekendHolidayPolicy = "single_gap"
    partial_day: PartialDayPolicy = "scale_by_session_fraction"
    halt_policy: HaltPolicy = "carry_forward"
    classify_gaps: bool = True
    periods_per_year: int | None = None


@dataclass(frozen=True)
class RVSpec:
    """Realized-variance sub-spec. Required iff the estimator is
    ``"rv"`` or the target is ``"quadratic_variation"``.
    """

    sampling_interval: SamplingInterval
    session: SessionScope
    include_overnight_gap: bool
    proxy: RVProxy


@dataclass(frozen=True)
class VolQualityPolicy:
    """Data-quality enforcement thresholds. ``max_data_quality_failure_rate``
    is the Gate-3 bar (default 1% after fallback, per research plan §10.1).
    """

    high_low_outlier_method: OutlierMethod
    high_low_outlier_threshold: float
    rv_noise_ratio_threshold: float
    estimator_dispersion_threshold: float
    min_nonzero_volume_fraction: float
    max_data_quality_failure_rate: float = 0.01


@dataclass(frozen=True)
class TimingPolicy:
    """Timestamp / latency / corporate-action policy. Promoted from
    research plan prose into the spec in v0.7 — every field affecting
    the emitted number lives here, not in a separate text section.
    """

    bar_timestamp_semantics: BarTimestampSemantics
    data_latency_policy: DataLatencyPolicy
    corporate_action_policy: CorporateActionPolicy


@dataclass(frozen=True)
class VolEstimatorSpec:
    """Single source of truth for an estimator run. Two runs with the
    same spec on the same bars must produce identical output; the
    Gate-1 golden tests enforce this contract.
    """

    estimator: EstimatorName
    target: EstimatorTarget
    window: int
    min_periods: int
    output_unit: OutputUnit
    price_basis: PriceBasis
    calendar_policy: VolCalendarPolicy
    timing_policy: TimingPolicy
    quality_policy: VolQualityPolicy
    rv_spec: RVSpec | None = None


@dataclass(frozen=True)
class VolComponent:
    """Persisted variance component with explicit availability metadata.

    A component whose ``source`` is ``"unavailable"`` carries NaN values
    by convention; the caller must not interpret an unavailable
    component as zero. See research plan §3.6 mode-availability table.
    """

    value: pl.Series
    unit: ComponentUnit
    source: ComponentSource
    valid_from: pl.Series
    quality_flags: pl.Series


@dataclass(frozen=True)
class VolEstimate:
    """Output of ``estimate_variance``. ``var_per_bar`` is the canonical
    ``risk_var_t``; ``vol_per_bar = sqrt(var_per_bar)`` is the derived
    convenience. Annualized fields are ``None`` when the spec's
    ``output_unit`` does not request them.
    """

    var_per_bar: pl.Series
    vol_per_bar: pl.Series
    var_annualized: pl.Series | None
    vol_annualized: pl.Series | None
    components: dict[str, VolComponent]
    quality_flags: pl.Series
    estimator_used: pl.Series
    valid_from: pl.Series


__all__ = [
    "BarTimestampSemantics",
    "CalendarBasis",
    "ComponentSource",
    "ComponentUnit",
    "CorporateActionPolicy",
    "DataLatencyPolicy",
    "EstimatorName",
    "EstimatorTarget",
    "HaltPolicy",
    "OutlierMethod",
    "OutputUnit",
    "OvernightHandling",
    "PartialDayPolicy",
    "PriceBasis",
    "RVProxy",
    "RVSpec",
    "SamplingInterval",
    "SessionScope",
    "TimingPolicy",
    "VolCalendarPolicy",
    "VolComponent",
    "VolEstimate",
    "VolEstimatorSpec",
    "VolQualityPolicy",
    "WeekendHolidayPolicy",
]
