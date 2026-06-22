from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from liq.features.mean_reversion.vol import trailing_range_vol

# Plain non-OHLCV float strategy for point-in-time rolling behavior.
_FLOAT = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


def _paired_high_low(n: int, raw_a: list[float], raw_b: list[float]) -> dict[str, list[float]]:
    """Coerce two non-OHLCV float sequences into a valid (high, low) pair."""
    take = min(n, len(raw_a), len(raw_b))
    high = [max(raw_a[i], raw_b[i]) for i in range(take)]
    low = [min(raw_a[i], raw_b[i]) for i in range(take)]
    return {"high": high, "low": low}


@given(
    raw_a=st.lists(_FLOAT, min_size=4, max_size=40),
    raw_b=st.lists(_FLOAT, min_size=4, max_size=40),
    lookback=st.integers(min_value=1, max_value=8),
)
def test_trailing_range_vol_is_strictly_pit(
    raw_a: list[float], raw_b: list[float], lookback: int
) -> None:
    """Non-OHLCV property: output at index `i` must equal the helper run on the
    truncated prefix `seq[:i+1]` — i.e. the trailing-range volatility at `i`
    never depends on any input bar strictly later than `i`. This is the PIT
    invariant required for point-in-time rolling features.
    """
    n = min(len(raw_a), len(raw_b))
    if n < lookback:
        return  # Hypothesis shrunk past the warmup; nothing meaningful to check.
    seq = _paired_high_low(n, raw_a, raw_b)

    full = trailing_range_vol(seq, lookback=lookback)

    for i in range(lookback - 1, n):
        truncated = trailing_range_vol(
            {"high": seq["high"][: i + 1], "low": seq["low"][: i + 1]},
            lookback=lookback,
        )
        assert not math.isnan(truncated[-1])
        assert truncated[-1] == pytest.approx(full[i], rel=1e-9, abs=1e-9)


def test_trailing_range_vol_uses_rolling_mean_high_low_range() -> None:
    bars = pl.DataFrame(
        {
            "high": [10.0, 13.0, 15.0, 18.0],
            "low": [8.0, 10.0, 11.0, 13.0],
        }
    )

    out = trailing_range_vol(bars, lookback=3)

    assert math.isnan(out[0])
    assert math.isnan(out[1])
    assert out[2] == pytest.approx((2.0 + 3.0 + 4.0) / 3.0)
    assert out[3] == pytest.approx((3.0 + 4.0 + 5.0) / 3.0)


def test_trailing_range_vol_accepts_mapping_inputs() -> None:
    out = trailing_range_vol({"high": [5.0, 8.0], "low": [4.0, 6.0]}, lookback=1)

    assert out == pytest.approx([1.0, 2.0])


def test_trailing_range_vol_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="high"):
        trailing_range_vol({"low": [1.0]}, lookback=1)
