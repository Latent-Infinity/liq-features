"""Data-quality enforcement for the canonical risk-variance estimator.

Implements research plan §9 row-by-row plus the PIT-specific outlier
rule from §3.5. Every emitted bar passes through
:func:`check_bar_quality`, which returns a :class:`QualityVerdict`
carrying:

- ``is_hard_error``: ``True`` when the bar violates a contract-level
  rule (e.g. ``high < low``); call sites raise
  :class:`~liq.features.volatility.exceptions.VolDataQualityError` only
  if no fallback rescues the bar.
- ``flags``: the quality-flag vocabulary terms from §7.4 that apply.
- ``fallback_eligible``: whether a fallback estimator can produce a
  number for this bar.
- ``rule``: the rule name that fired (for the structured-log payload).

Window-level enforcement (the Gate-3 ``max_data_quality_failure_rate``)
lives in :func:`enforce_failure_rate`.

The PIT-safe outlier detector :func:`past_rolling_z` consumes only
*past* values; future neighbors never alter the verdict at time ``t``,
per the v0.7 PIT rule.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from liq.features.volatility.exceptions import VolDataQualityError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from liq.features.volatility.contracts import VolQualityPolicy


# Quality flag vocabulary — research plan §7.4 + data-quality additions.
FLAG_GAP_DOMINATED_VOL = "GAP_DOMINATED_VOL"
FLAG_INTRADAY_RANGE_DOMINATED_VOL = "INTRADAY_RANGE_DOMINATED_VOL"
FLAG_HIGH_ESTIMATOR_DISAGREEMENT = "HIGH_ESTIMATOR_DISAGREEMENT"
FLAG_CTC_DISAGREES_WITH_RANGE = "CTC_DISAGREES_WITH_RANGE"
FLAG_JUMP_DAY = "JUMP_DAY"
FLAG_NOISY_RV_TARGET = "NOISY_RV_TARGET"
FLAG_LOW_CONFIDENCE = "LOW_CONFIDENCE"
# Data-quality flags added here:
FLAG_MISSING_OPEN = "MISSING_OPEN"
FLAG_HIGH_LOW_OUTLIER = "HIGH_LOW_OUTLIER"
FLAG_HALTED_DAY = "HALTED_DAY"
FLAG_ZERO_VOLUME_DAY = "ZERO_VOLUME_DAY"
FLAG_PARTIAL_TRADING_DAY = "PARTIAL_TRADING_DAY"
FLAG_DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"


@dataclass(frozen=True)
class QualityVerdict:
    """Per-bar data-quality verdict.

    ``is_hard_error`` and ``fallback_eligible`` are independent: a bar
    with ``high < low`` is a hard error AND ineligible for fallback (no
    estimator can rescue impossible OHLC); a bar with missing ``open``
    is NOT a hard error but IS fallback-eligible (Parkinson and CtC
    work without it).
    """

    bar_index: int
    is_hard_error: bool
    flags: tuple[str, ...]
    fallback_eligible: bool
    rule: str | None


def _is_present(value: object) -> bool:
    """A value is "present" iff it's not ``None`` and not NaN."""
    if value is None:
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _hard_error(bar_index: int, rule: str, flag: str = FLAG_DATA_QUALITY_FAILURE) -> QualityVerdict:
    return QualityVerdict(
        bar_index=bar_index,
        is_hard_error=True,
        flags=(flag,),
        fallback_eligible=False,
        rule=rule,
    )


def check_bar_quality(
    *,
    bar_index: int,
    bar: dict[str, object],
    past_bars: Sequence[dict[str, object]],
    policy: VolQualityPolicy,
) -> QualityVerdict:
    """Apply the per-bar quality rules from research plan §9.

    Order matters: hard errors (``high < low``, nonpositive,
    ``missing_close``) short-circuit; only after those pass do we
    consider missing-open, outlier detection, and halt/volume flags.
    """
    high = bar.get("high")
    low = bar.get("low")
    close = bar.get("close")
    open_ = bar.get("open")
    volume = bar.get("volume")

    # 1. Missing close — hard error.
    if not _is_present(close):
        return _hard_error(bar_index, "missing_close")

    # 2. Nonpositive prices — hard error.
    for value in (open_, high, low, close):
        if value is not None and isinstance(value, int | float) and value <= 0:
            return _hard_error(bar_index, "nonpositive_price")

    # 3. high < low — hard error.
    if (
        _is_present(high)
        and _is_present(low)
        and isinstance(high, int | float)
        and isinstance(low, int | float)
        and high < low
    ):
        return _hard_error(bar_index, "high_lt_low")

    # 4. high < max(open, close) or low > min(open, close) — soft error;
    # mark suspect-H/L → falls back to CtC at the chain level.
    flags: list[str] = []
    suspect_hl = False
    if _is_present(high) and _is_present(low) and _is_present(close):
        max_oc = max((v for v in (open_, close) if _is_present(v)), default=None)
        min_oc = min((v for v in (open_, close) if _is_present(v)), default=None)
        if (
            max_oc is not None
            and isinstance(high, int | float)
            and isinstance(max_oc, int | float)
            and high < max_oc
        ):
            suspect_hl = True
        if (
            min_oc is not None
            and isinstance(low, int | float)
            and isinstance(min_oc, int | float)
            and low > min_oc
        ):
            suspect_hl = True

    if suspect_hl:
        flags.append(FLAG_DATA_QUALITY_FAILURE)

    # 5. Missing open — fallback-eligible (Parkinson / CtC).
    if not _is_present(open_):
        flags.append(FLAG_MISSING_OPEN)
        return QualityVerdict(
            bar_index=bar_index,
            is_hard_error=False,
            flags=tuple(flags),
            fallback_eligible=True,
            rule="missing_open",
        )

    # 6. PIT-safe outlier detection on the bar's H/L log-spread.
    if policy.high_low_outlier_method != "none" and high is not None and low is not None:
        spreads = []
        for prev in past_bars:
            ph = prev.get("high")
            pl = prev.get("low")
            if (
                _is_present(ph)
                and _is_present(pl)
                and isinstance(ph, int | float)
                and isinstance(pl, int | float)
                and ph > 0
                and pl > 0
            ):
                spreads.append(math.log(float(ph) / float(pl)))
        current_spread = (
            math.log(float(high) / float(low))
            if isinstance(high, int | float) and isinstance(low, int | float)
            else 0.0
        )
        if len(spreads) < 5:
            flags.append(FLAG_LOW_CONFIDENCE)
        else:
            z = _z_score(current_spread, spreads, method=policy.high_low_outlier_method)
            if z is not None and abs(z) > policy.high_low_outlier_threshold:
                flags.append(FLAG_HIGH_LOW_OUTLIER)

    # 7. Halt / zero-volume detection (when volume is present).
    if _is_present(volume) and isinstance(volume, int | float) and volume == 0:
        flags.append(FLAG_ZERO_VOLUME_DAY)

    return QualityVerdict(
        bar_index=bar_index,
        is_hard_error=False,
        flags=tuple(flags),
        fallback_eligible=True,
        rule=None,
    )


def _z_score(current: float, past: Sequence[float], *, method: str) -> float | None:
    """One-sided z-score against ``past`` only — PIT-safe by
    construction. Supports ``past_rolling_z`` (mean / stdev) and
    ``mad`` (median / median-absolute-deviation)."""
    if not past:
        return None
    if method == "past_rolling_z":
        mu = statistics.fmean(past)
        sigma = statistics.pstdev(past)
        if sigma == 0:
            return None
        return (current - mu) / sigma
    if method == "mad":
        med = statistics.median(past)
        mad_value = statistics.median(abs(x - med) for x in past)
        if mad_value == 0:
            return None
        # Scale MAD by 1.4826 to approximate the std under normality.
        return (current - med) / (1.4826 * mad_value)
    return None


def past_rolling_z(values: Sequence[float], *, min_window: int) -> list[float]:
    """One-sided z-score series over ``values`` using only past
    neighbors. The first ``min_window - 1`` entries are NaN because
    there isn't enough history to compute a z-score yet.

    PIT-safe by construction: the z at index ``i`` uses only
    ``values[:i]`` — never any ``values[i:]``.
    """
    out: list[float] = []
    for i in range(len(values)):
        past = values[max(0, i - min_window + 1) : i]
        if len(past) < min_window - 1:
            out.append(float("nan"))
            continue
        mu = statistics.fmean(past)
        sigma = statistics.pstdev(past)
        if sigma == 0:
            out.append(float("nan"))
            continue
        out.append((values[i] - mu) / sigma)
    return out


def compute_failure_rate(verdicts: Sequence[QualityVerdict]) -> float:
    """Fraction of bars with ``is_hard_error == True``. Soft flags do
    not count toward the Gate-3 threshold — only true rejections do."""
    if not verdicts:
        return 0.0
    n_hard = sum(1 for v in verdicts if v.is_hard_error)
    return n_hard / len(verdicts)


def enforce_failure_rate(verdicts: Sequence[QualityVerdict], policy: VolQualityPolicy) -> None:
    """Raise :class:`VolDataQualityError` when the rate of hard errors
    in ``verdicts`` exceeds ``policy.max_data_quality_failure_rate``.

    This is the Gate-3 prep gate from research plan §10.1.
    """
    rate = compute_failure_rate(verdicts)
    if rate > policy.max_data_quality_failure_rate:
        raise VolDataQualityError(
            f"data-quality failure rate {rate:.4f} exceeds threshold "
            f"{policy.max_data_quality_failure_rate:.4f}"
        )


__all__ = [
    "FLAG_CTC_DISAGREES_WITH_RANGE",
    "FLAG_DATA_QUALITY_FAILURE",
    "FLAG_GAP_DOMINATED_VOL",
    "FLAG_HALTED_DAY",
    "FLAG_HIGH_ESTIMATOR_DISAGREEMENT",
    "FLAG_HIGH_LOW_OUTLIER",
    "FLAG_INTRADAY_RANGE_DOMINATED_VOL",
    "FLAG_JUMP_DAY",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_MISSING_OPEN",
    "FLAG_NOISY_RV_TARGET",
    "FLAG_PARTIAL_TRADING_DAY",
    "FLAG_ZERO_VOLUME_DAY",
    "QualityVerdict",
    "check_bar_quality",
    "compute_failure_rate",
    "enforce_failure_rate",
    "past_rolling_z",
]
