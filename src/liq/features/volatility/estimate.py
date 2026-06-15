"""Public entry-point for the canonical risk-variance estimator.

The signature + docstring are stable; the body dispatches to the formula
registry, applies the windowed aggregation, composes `risk_var_t = cont
+ overnight_gap` when the spec targets close-to-close risk variance, and
returns a ``VolEstimate``.

Anchors: ``[PHASE0_CONTRACT]`` (canonical scalar = close-to-close risk
variance), ``[CANONICALIZATION]`` (this is the one canonical path),
``[APPENDIX_FORMULAS]`` (the registry the body dispatches to).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from .contracts import (
    ComponentSource,
    VolComponent,
    VolEstimate,
    VolEstimatorSpec,
)
from .estimators.fallback import select_estimator
from .estimators.formulas import (
    ctc_var_contribution,
    garman_klass_var_contribution,
    gk_yang_zhang_var_contribution,
    parkinson_var_contribution,
    rogers_satchell_var_contribution,
    yz_open_close_term,
    yz_overnight_term,
    yz_rs_term,
)
from .estimators.windowed import NAN, trailing_mean, yang_zhang_var
from .exceptions import VolDataQualityError, VolPITViolationError, VolSpecError
from .logging import build_emitter
from .quality import (
    FLAG_DATA_QUALITY_FAILURE,
    FLAG_MISSING_OPEN,
    FLAG_NOISY_RV_TARGET,
    QualityVerdict,
    check_bar_quality,
    enforce_failure_rate,
)
from .rv import compute_rv, rv_noise_gate

if TYPE_CHECKING:
    pass


def _spec_hash(spec: VolEstimatorSpec) -> str:
    """Stable hex digest of the spec — used as ``spec_hash`` in the
    structured-log payload. Two runs with the same ``VolEstimatorSpec``
    produce identical ``spec_hash`` so log consumers can correlate
    estimates across calls."""
    import hashlib
    from dataclasses import asdict

    canonical = repr(sorted(_flatten(asdict(spec)).items()))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()


def _flatten(payload: dict[str, object], prefix: str = "") -> dict[str, object]:
    """Flatten a nested dataclass dict for deterministic hashing.

    Sorted-by-key encoding ensures `asdict` produces a canonical byte
    sequence regardless of dataclass field-order changes.
    """
    out: dict[str, object] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            nested: dict[str, object] = {str(k): v for k, v in value.items()}
            out.update(_flatten(nested, path))
        else:
            out[path] = value
    return out


def _per_bar_quality(bars: pl.DataFrame, spec: VolEstimatorSpec) -> list[QualityVerdict]:
    """Run the per-bar quality check across the whole frame.

    Each bar's verdict consumes ONLY past bars (PIT-safe by
    construction); the outlier detector accepts up to
    ``spec.window`` prior bars as the trailing context.
    """
    rows = bars.to_dicts()
    verdicts: list[QualityVerdict] = []
    for i, row in enumerate(rows):
        past = rows[max(0, i - spec.window) : i]
        verdicts.append(
            check_bar_quality(
                bar_index=i,
                bar=row,
                past_bars=past,
                policy=spec.quality_policy,
            )
        )
    return verdicts


def _extract_symbol(bars: pl.DataFrame) -> str:
    """Pull a symbol identifier from the frame; default to ``"unknown"``
    when not present so the structured-log payload always has a
    ``symbol`` key."""
    if "symbol" in bars.columns:
        col = bars.get_column("symbol")
        unique = col.unique().to_list()
        if len(unique) == 1:
            return str(unique[0])
        return "MULTI"
    return "unknown"


def _validate_spec(spec: VolEstimatorSpec) -> None:
    """Raise ``VolSpecError`` when the spec is internally inconsistent.

    The two checks landed here are the ones flagged in the v0.7
    iteration hook: ``rv_spec`` must be set when the estimator is
    ``"rv"`` or the target is ``"quadratic_variation"``; ``window`` and
    ``min_periods`` must be positive with ``min_periods <= window``.
    """
    if spec.window <= 0:
        raise VolSpecError(f"window must be positive (got {spec.window})")
    if spec.min_periods <= 0:
        raise VolSpecError(f"min_periods must be positive (got {spec.min_periods})")
    if spec.min_periods > spec.window:
        raise VolSpecError(f"min_periods ({spec.min_periods}) cannot exceed window ({spec.window})")
    needs_rv_spec = spec.estimator == "rv" or spec.target == "quadratic_variation"
    if needs_rv_spec and spec.rv_spec is None:
        raise VolSpecError(
            "rv_spec is required when estimator='rv' or target='quadratic_variation'"
        )


def _validate_bars(bars: pl.DataFrame) -> None:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise VolSpecError(f"bars is missing required columns: {sorted(missing)}")
    if bars.height < 1:
        raise VolSpecError("bars must have at least one row")


def _resolve_asof(bars: pl.DataFrame, asof: datetime | None) -> datetime:
    if asof is not None:
        return asof
    last_ts = bars.get_column("timestamp")[-1]
    if isinstance(last_ts, datetime):
        return last_ts
    return datetime.fromisoformat(str(last_ts))


def _valid_from_series(bars: pl.DataFrame) -> pl.Series:
    """Return the ``valid_from`` series, defaulting to ``timestamp`` when
    not provided. Used by both the PIT check and the output payload."""
    if "valid_from" in bars.columns:
        return bars.get_column("valid_from")
    return bars.get_column("timestamp").alias("valid_from")


def _check_pit(valid_from: pl.Series, asof: datetime) -> None:
    """Hard PIT gate per research plan §3.5: assert
    ``max(input.valid_from) <= t``. Raises ``VolPITViolationError`` on
    any violation."""
    series_max = valid_from.max()
    if series_max is None:
        return
    if isinstance(series_max, datetime) and series_max > asof:
        raise VolPITViolationError(
            f"PIT violation: max(valid_from) = {series_max!r} > asof = {asof!r}"
        )


def _log_columns(
    bars: pl.DataFrame,
) -> tuple[list[float], list[float], list[float], list[float], list[float | None]]:
    """Convert OHLC + prior close into per-bar log-price tuples.

    Returns ``(o, h, lo, c, c_prev)`` parallel lists. ``c_prev[i]`` is
    ``log(close[i-1])``; ``c_prev[0]`` is ``None`` (caller emits NaN for
    estimators that require it).
    """

    def safe_log(value: object) -> float:
        if value is None:
            return NAN
        if not isinstance(value, str | int | float):
            return NAN
        try:
            x = float(value)
        except (TypeError, ValueError):
            return NAN
        if math.isnan(x) or x <= 0:
            return NAN
        return math.log(x)

    opens = [safe_log(x) for x in bars.get_column("open").to_list()]
    highs = [safe_log(x) for x in bars.get_column("high").to_list()]
    lows = [safe_log(x) for x in bars.get_column("low").to_list()]
    closes = [safe_log(x) for x in bars.get_column("close").to_list()]
    c_prevs: list[float | None] = [None] + closes[:-1]
    return opens, highs, lows, closes, c_prevs


def _window_fallback_verdict(verdicts: list[QualityVerdict]) -> QualityVerdict:
    """Choose the verdict that drives the documented window-level fallback.

    Fallback remains window-level to avoid mixing estimator families
    inside one trailing mean, but the decision must inspect the whole
    window. Otherwise a missing open at t-3 would be invisible if t is
    clean and formula dispatch could still try to log the missing open.
    """
    if not verdicts:
        return QualityVerdict(
            bar_index=0,
            is_hard_error=False,
            flags=(),
            fallback_eligible=True,
            rule=None,
        )
    for verdict in verdicts:
        if FLAG_DATA_QUALITY_FAILURE in verdict.flags and not verdict.is_hard_error:
            return verdict
    for verdict in verdicts:
        if FLAG_MISSING_OPEN in verdict.flags:
            return verdict
    return verdicts[-1]


def _per_bar_for(
    name: str,
    o: list[float],
    h: list[float],
    lo: list[float],
    c: list[float],
    c_prev: list[float | None],
) -> list[float]:
    """Compute the per-bar series for the named estimator over the
    pre-converted log-price columns. Bars whose required ``c_prev`` is
    ``None`` (i.e. bar 0 when the formula needs prev close) emit NaN."""
    n = len(o)
    out: list[float] = [NAN] * n
    for i in range(n):
        cp = c_prev[i]
        if name == "ctc":
            if cp is None:
                continue
            out[i] = ctc_var_contribution(c[i], cp)
        elif name == "parkinson":
            out[i] = parkinson_var_contribution(h[i], lo[i])
        elif name == "garman_klass":
            out[i] = garman_klass_var_contribution(o[i], h[i], lo[i], c[i])
        elif name == "rogers_satchell":
            out[i] = rogers_satchell_var_contribution(o[i], h[i], lo[i], c[i])
        elif name == "gk_yang_zhang":
            if cp is None:
                continue
            out[i] = gk_yang_zhang_var_contribution(o[i], h[i], lo[i], c[i], cp)
        elif name == "yz_overnight_term":
            if cp is None:
                continue
            out[i] = yz_overnight_term(o[i], cp)
        elif name == "yz_open_close_term":
            out[i] = yz_open_close_term(o[i], c[i])
        elif name == "yz_rs_term":
            out[i] = yz_rs_term(o[i], h[i], lo[i], c[i])
        else:
            raise VolSpecError(f"unsupported estimator: {name!r}")
    return out


def _windowed_for_estimator(
    spec: VolEstimatorSpec,
    o: list[float],
    h: list[float],
    lo: list[float],
    c: list[float],
    c_prev: list[float | None],
) -> tuple[list[float], dict[str, list[float]]]:
    """Return the windowed per-bar variance series for the spec's
    primary estimator plus a dict of supporting component series
    (``cont`` and ``overnight_gap`` when the spec composes
    close-to-close risk variance).
    """
    window = spec.window
    min_periods = spec.min_periods
    components: dict[str, list[float]] = {}

    if spec.estimator == "yang_zhang":
        overnight = _per_bar_for("yz_overnight_term", o, h, lo, c, c_prev)
        open_close = _per_bar_for("yz_open_close_term", o, h, lo, c, c_prev)
        rs = _per_bar_for("yz_rs_term", o, h, lo, c, c_prev)
        primary = yang_zhang_var(
            overnight_terms=overnight,
            open_close_terms=open_close,
            rs_terms=rs,
            window=window,
            min_periods=min_periods,
        )
        # Decompose components for the close-to-close risk-variance path.
        components["cont"] = yang_zhang_var(
            overnight_terms=[0.0] * len(overnight),
            open_close_terms=open_close,
            rs_terms=rs,
            window=window,
            min_periods=min_periods,
        )
        components["overnight_gap"] = trailing_mean(
            overnight, window=window, min_periods=min_periods
        )
        return primary, components

    per_bar = _per_bar_for(spec.estimator, o, h, lo, c, c_prev)
    primary = trailing_mean(per_bar, window=window, min_periods=min_periods)
    return primary, components


def _signature_for_bar(
    bar_index: int,
    intra_returns: list[float],
    emitter,
    *,
    ratio_threshold: float,
) -> tuple[float, bool]:
    """Compute the §5.3-gated realized variance for ``bar_index``.

    Samples RV at 1m, 5m, and 15m strides over ``intra_returns``,
    applies the noise gate, emits ``rv_noise_gate_fired`` when it trips,
    and returns the 5m RV in that case (the canonical fallback). When
    the gate stays down, returns the 1m RV.
    """
    rv_1m = compute_rv(intra_returns)
    rv_5m = compute_rv(_coarsen_returns(intra_returns, stride=5))
    rv_15m = compute_rv(_coarsen_returns(intra_returns, stride=15))
    price_movement = sum(intra_returns)
    rv_by_interval = {"1m": rv_1m, "5m": rv_5m, "15m": rv_15m}
    if rv_noise_gate(
        rv_by_interval,
        price_movement=price_movement,
        ratio_threshold=ratio_threshold,
    ):
        emitter.rv_noise_gate_fired(
            bar_index=bar_index,
            rv_1m=rv_1m,
            rv_5m=rv_5m,
            rv_15m=rv_15m,
            price_movement=price_movement,
            target="rv_5m",
        )
        return rv_5m, True
    return rv_1m, False


def _coarsen_returns(intra_returns: list[float], *, stride: int) -> list[float]:
    """Aggregate consecutive one-minute returns into coarser returns."""
    if stride <= 1:
        return list(intra_returns)
    return [
        sum(intra_returns[i : i + stride])
        for i in range(0, len(intra_returns) - stride + 1, stride)
    ]


def _build_volatility_signature(
    *,
    n_bars: int,
    intra_bar_returns: dict[int, list[float]],
    emitter,
    ratio_threshold: float,
) -> tuple[list[float], set[int]]:
    """Build the per-bar volatility-signature series.

    Bars present in ``intra_bar_returns`` get the §5.3-gated RV; bars
    without minute data carry NaN so consumers can distinguish
    "unobserved" from "zero variance".
    """
    series: list[float] = []
    noisy_indices: set[int] = set()
    for i in range(n_bars):
        returns = intra_bar_returns.get(i)
        if not returns:
            series.append(NAN)
            continue
        value, noisy = _signature_for_bar(
            i,
            returns,
            emitter,
            ratio_threshold=ratio_threshold,
        )
        series.append(value)
        if noisy:
            noisy_indices.add(i)
    return series, noisy_indices


def _maybe_annualize(
    var_per_bar: list[float], spec: VolEstimatorSpec
) -> tuple[list[float] | None, list[float] | None]:
    """Build annualized variance + vol series when the output_unit
    requests them; otherwise return ``(None, None)`` per the v0.7 spec
    ambiguity fix recorded in the impl plan."""
    unit = spec.output_unit
    if unit in {"per_bar_variance", "per_bar_vol"}:
        return None, None
    ppy = spec.calendar_policy.periods_per_year
    if ppy is None:
        raise VolSpecError(
            "calendar_policy.periods_per_year must be set when output_unit "
            "is 'annualized_variance' or 'annualized_vol'"
        )
    var_ann = [v * ppy if not math.isnan(v) else NAN for v in var_per_bar]
    vol_ann = [math.sqrt(v) if not math.isnan(v) and v >= 0 else NAN for v in var_ann]
    return var_ann, vol_ann


def estimate_variance(
    bars: pl.DataFrame,
    spec: VolEstimatorSpec,
    *,
    asof: datetime | None = None,
    intra_bar_returns: dict[int, list[float]] | None = None,
) -> VolEstimate:
    """Compute the canonical point-in-time trailing risk variance.

    The output is variance-first per research plan §3.1: ``var_per_bar``
    is the canonical ``risk_var_t`` and ``vol_per_bar = sqrt(var_per_bar)``
    is the derived convenience. Annualized fields are populated only
    when ``spec.output_unit`` requests them.

    Hard contract (research plan §3.5):

    - **Point-in-time.** For every emitted timestamp ``t``,
      ``max(input.valid_from) <= t``. Violations raise
      ``VolPITViolationError``.
    - **Scale invariance.** Multiplying all OHLC inputs by ``k > 0``
      leaves the per-bar variance unchanged within tolerance
      ``1e-10`` (windowed) / ``1e-12`` (deterministic).
    - **Spec authority.** Two runs with identical ``spec`` on identical
      ``bars`` produce bit-identical output.

    Args:
        bars: A ``polars.DataFrame`` carrying at least
            ``timestamp``, ``open``, ``high``, ``low``, ``close``.
            An optional ``valid_from`` column overrides the default
            (``timestamp``) for PIT checking.
        spec: The frozen ``VolEstimatorSpec`` that governs every
            policy affecting the emitted number.
        asof: Optional explicit timestamp anchoring the PIT check;
            when ``None``, the latest bar's timestamp is used.
        intra_bar_returns: Optional ``{bar_index: list[float]}`` mapping
            of intra-bar minute log-returns. When provided, the
            ``volatility_signature`` ``VolComponent`` is populated with
            the resolved per-bar realized variance (5m fallback when
            the §5.3 RV-noise gate fires, 1m otherwise). When omitted,
            no ``volatility_signature`` component is emitted.

    Returns:
        A ``VolEstimate`` per research plan §3.6.
    """
    _validate_spec(spec)
    _validate_bars(bars)
    asof_resolved = _resolve_asof(bars, asof)
    valid_from = _valid_from_series(bars)
    _check_pit(valid_from, asof_resolved)

    emitter = build_emitter(
        symbol=_extract_symbol(bars),
        asof=asof_resolved.isoformat(),
        window=spec.window,
        spec_hash=_spec_hash(spec),
        estimator=spec.estimator,
    )

    # Pre-flight data-quality scan (research plan §9 + §3.5 PIT-safe
    # outlier rule).
    verdicts = _per_bar_quality(bars, spec)
    try:
        enforce_failure_rate(verdicts, spec.quality_policy)
    except VolDataQualityError as exc:
        emitter.data_quality_rejected(
            rule="failure_rate_exceeded",
            details={"failure_rate": str(exc)},
        )
        raise

    # Window-level fallback decision: probe with the worst verdict that
    # would shape the chain. ``has_prev_close=True`` here means "the
    # input frame carries a previous-close column or the first bar's
    # neighbor exists" — we always have prev close for the windowed
    # estimator path (the first bar's contribution is NaN by formula).
    for verdict in verdicts:
        if verdict.is_hard_error:
            emitter.data_quality_rejected(
                rule=verdict.rule or "data_quality_failure",
                details={"bar_index": verdict.bar_index},
            )

    fallback = select_estimator(
        requested=spec.estimator,
        verdict=_window_fallback_verdict(verdicts),
        has_prev_close=True,
    )
    if fallback.selected and fallback.selected != spec.estimator:
        emitter.estimator_fallback_applied(
            from_=spec.estimator, to=fallback.selected, reason=fallback.reason or ""
        )
    resolved_estimator = fallback.selected or spec.estimator
    emitter.estimator_selected(resolved=resolved_estimator)

    # Per-bar quality-flag events (debounced inside the emitter).
    for v in verdicts:
        if v.flags:
            emitter.quality_flag_set(bar_index=v.bar_index, flags=list(v.flags))

    o, h, lo, c, c_prev = _log_columns(bars)
    # Build a spec-shaped object pointing at the resolved estimator so
    # the downstream windowed code dispatches correctly. (Re-creating a
    # frozen dataclass is the smallest change at this layer.)
    from dataclasses import replace as _dc_replace

    resolved_spec = _dc_replace(spec, estimator=resolved_estimator)
    primary_var, component_series = _windowed_for_estimator(resolved_spec, o, h, lo, c, c_prev)

    # Compose for close-to-close risk variance per research plan §3.1.
    # When the estimator already targets c2c (CtC, GK-YZ, YZ before the
    # recomposition below), the per-bar value already includes the
    # opening jump term; leave it untouched. The components dict still
    # surfaces the gap component when computable.
    if spec.target == "close_to_close_risk_variance" and (
        "cont" in component_series and "overnight_gap" in component_series
    ):
        cont = component_series["cont"]
        overnight = component_series["overnight_gap"]
        primary_var = [
            NAN if math.isnan(cont[i]) or math.isnan(overnight[i]) else cont[i] + overnight[i]
            for i in range(len(cont))
        ]

    vol_per_bar = [math.sqrt(v) if not math.isnan(v) and v >= 0 else NAN for v in primary_var]
    var_ann, vol_ann = _maybe_annualize(primary_var, spec)

    estimator_used = [resolved_estimator] * bars.height
    quality_flags: list[list[str]] = [list(v.flags) for v in verdicts]
    signature_values: list[float] | None = None

    if intra_bar_returns is not None:
        signature_values, noisy_indices = _build_volatility_signature(
            n_bars=bars.height,
            intra_bar_returns=intra_bar_returns,
            emitter=emitter,
            ratio_threshold=spec.quality_policy.rv_noise_ratio_threshold,
        )
        for bar_index in sorted(noisy_indices):
            if FLAG_NOISY_RV_TARGET not in quality_flags[bar_index]:
                quality_flags[bar_index].append(FLAG_NOISY_RV_TARGET)
            emitter.quality_flag_set(bar_index=bar_index, flags=[FLAG_NOISY_RV_TARGET])

    components_out: dict[str, VolComponent] = {}
    source_map: dict[str, ComponentSource] = {
        "cont": "derived",
        "overnight_gap": "daily_ohlc",
    }
    for name, series in component_series.items():
        components_out[name] = VolComponent(
            value=pl.Series(name, series, dtype=pl.Float64),
            unit="per_bar_variance",
            source=source_map.get(name, "derived"),
            valid_from=valid_from.alias("valid_from"),
            quality_flags=pl.Series("quality_flags", quality_flags, dtype=pl.List(pl.Utf8)),
        )

    # Volatility-signature emission (research plan §3.6, §5.3). Only
    # built when the caller supplies intra-bar minute returns; bars
    # without minute data carry NaN; bars whose 1m RV trips the §5.3
    # noise gate fall back to the 5m sampling and emit
    # ``rv_noise_gate_fired``.
    if signature_values is not None:
        components_out["volatility_signature"] = VolComponent(
            value=pl.Series("volatility_signature", signature_values, dtype=pl.Float64),
            unit="per_bar_variance",
            source="minute_rv",
            valid_from=valid_from.alias("valid_from"),
            quality_flags=pl.Series("quality_flags", quality_flags, dtype=pl.List(pl.Utf8)),
        )

    return VolEstimate(
        var_per_bar=pl.Series("var_per_bar", primary_var, dtype=pl.Float64),
        vol_per_bar=pl.Series("vol_per_bar", vol_per_bar, dtype=pl.Float64),
        var_annualized=(
            pl.Series("var_annualized", var_ann, dtype=pl.Float64) if var_ann else None
        ),
        vol_annualized=(
            pl.Series("vol_annualized", vol_ann, dtype=pl.Float64) if vol_ann else None
        ),
        components=components_out,
        quality_flags=pl.Series("quality_flags", quality_flags, dtype=pl.List(pl.Utf8)),
        estimator_used=pl.Series("estimator_used", estimator_used, dtype=pl.Utf8),
        valid_from=valid_from.alias("valid_from"),
    )


__all__ = ["estimate_variance"]
