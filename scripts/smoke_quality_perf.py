"""Performance smoke for the volatility data-quality / fallback path.

Times :func:`estimate_variance` over a 252-bar window × 500 synthetic
symbols using the Yang-Zhang spec with the canonical close-to-close
risk-variance target. Records wall-clock and per-call statistics into
the operator's per-iteration ``artifacts/`` directory.

**This script does NOT gate the phase.** It exists so the operator can
spot-check that the data-quality / fallback / logging hookups have not
introduced a perf regression. Run:

    uv run python scripts/smoke_quality_perf.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime, timedelta

import polars as pl

from liq.features.volatility import (
    TimingPolicy,
    VolCalendarPolicy,
    VolEstimatorSpec,
    VolQualityPolicy,
    estimate_variance,
)

N_BARS = 252
N_SYMBOLS = 500
SEED = 20240603


def _make_bars(symbol_index: int) -> pl.DataFrame:
    """Build a deterministic 252-bar OHLC frame for ``symbol_index``.

    Uses a simple modular-arithmetic walk so successive symbols produce
    distinct but bounded OHLC sequences without an RNG dependency
    (RNG would make perf measurements harder to reproduce)."""
    base_price = 50.0 + (symbol_index * 0.137) % 200
    base = datetime(2023, 1, 3, 20, 0, tzinfo=UTC)
    rows = []
    price = base_price
    for i in range(N_BARS):
        # A bounded triangular wave keeps prices > 0 and HL constraints valid.
        wave = ((symbol_index + i) % 23) / 23.0
        o = price
        h = price * (1.0 + 0.005 + 0.02 * wave)
        lo = price * (1.0 - 0.005 - 0.02 * wave)
        c = price * (1.0 + 0.01 * (wave - 0.5))
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 1_000_000 + (i * 73 + symbol_index * 31) % 100_000,
            }
        )
        price = c
    return pl.DataFrame(rows)


def _spec() -> VolEstimatorSpec:
    return VolEstimatorSpec(
        estimator="yang_zhang",
        target="close_to_close_risk_variance",
        window=21,
        min_periods=21,
        output_unit="per_bar_variance",
        price_basis="point_in_time_split_adjusted",
        calendar_policy=VolCalendarPolicy(periods_per_year=252),
        timing_policy=TimingPolicy(
            bar_timestamp_semantics="end_time",
            data_latency_policy="close_plus_0",
            corporate_action_policy="point_in_time_adjusted",
        ),
        quality_policy=VolQualityPolicy(
            high_low_outlier_method="past_rolling_z",
            high_low_outlier_threshold=4.0,
            rv_noise_ratio_threshold=2.0,
            estimator_dispersion_threshold=1.5,
            min_nonzero_volume_fraction=0.95,
        ),
        rv_spec=None,
    )


def main() -> int:
    spec = _spec()
    per_call: list[float] = []
    started = time.perf_counter()
    for sym in range(N_SYMBOLS):
        bars = _make_bars(sym)
        call_start = time.perf_counter()
        _ = estimate_variance(bars, spec)
        per_call.append(time.perf_counter() - call_start)
    wall = time.perf_counter() - started

    per_call.sort()
    n = len(per_call)
    summary = {
        "n_symbols": N_SYMBOLS,
        "n_bars_per_symbol": N_BARS,
        "wall_clock_s": round(wall, 4),
        "per_call_s": {
            "min": round(per_call[0], 6),
            "p50": round(per_call[n // 2], 6),
            "p95": round(per_call[int(n * 0.95)], 6),
            "p99": round(per_call[int(n * 0.99)], 6),
            "max": round(per_call[-1], 6),
            "mean": round(sum(per_call) / n, 6),
        },
        "throughput_calls_per_s": round(N_SYMBOLS / wall, 2),
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(
        f"OK: {N_SYMBOLS} symbols × {N_BARS} bars in {wall:.2f}s "
        f"({summary['throughput_calls_per_s']} calls/s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
