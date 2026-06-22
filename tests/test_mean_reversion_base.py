from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liq.features.mean_reversion.base import roll_extreme_midrange, roll_mean_midrange

# Plain non-OHLCV float strategy for point-in-time rolling behavior.
_FLOAT = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


def _paired_high_low(raw_a: list[float], raw_b: list[float]) -> dict[str, list[float]] | None:
    """Coerce two non-OHLCV float sequences into a valid (high, low) pair."""
    n = min(len(raw_a), len(raw_b))
    if n == 0:
        return None
    high = [max(raw_a[i], raw_b[i]) for i in range(n)]
    low = [min(raw_a[i], raw_b[i]) for i in range(n)]
    return {"high": high, "low": low}


def _assert_nan(value: float) -> None:
    assert math.isnan(value), f"expected NaN, got {value}"


@given(
    raw_a=st.lists(_FLOAT, min_size=4, max_size=40),
    raw_b=st.lists(_FLOAT, min_size=4, max_size=40),
    lookback=st.integers(min_value=1, max_value=8),
)
def test_roll_extreme_midrange_is_strictly_pit(
    raw_a: list[float], raw_b: list[float], lookback: int
) -> None:
    """Non-OHLCV property: the rolling-extreme midrange at index `i` depends
    only on inputs at indices `[i - lookback + 1, i]`; truncating future inputs
    yields the same value at `i`."""
    seq = _paired_high_low(raw_a, raw_b)
    if seq is None or len(seq["high"]) < lookback:
        return

    full = roll_extreme_midrange(seq, lookback=lookback)
    n = len(seq["high"])

    for i in range(lookback - 1, n):
        truncated = roll_extreme_midrange(
            {"high": seq["high"][: i + 1], "low": seq["low"][: i + 1]},
            lookback=lookback,
        )
        assert not math.isnan(truncated[-1])
        assert truncated[-1] == pytest.approx(full[i], rel=1e-9, abs=1e-9)


@given(
    raw_a=st.lists(_FLOAT, min_size=4, max_size=40),
    raw_b=st.lists(_FLOAT, min_size=4, max_size=40),
    lookback=st.integers(min_value=1, max_value=8),
)
def test_roll_mean_midrange_is_strictly_pit(
    raw_a: list[float], raw_b: list[float], lookback: int
) -> None:
    """Non-OHLCV property: the rolling-mean midrange at index `i` depends only
    on inputs at indices `[i - lookback + 1, i]`."""
    seq = _paired_high_low(raw_a, raw_b)
    if seq is None or len(seq["high"]) < lookback:
        return

    full = roll_mean_midrange(seq, lookback=lookback)
    n = len(seq["high"])

    for i in range(lookback - 1, n):
        truncated = roll_mean_midrange(
            {"high": seq["high"][: i + 1], "low": seq["low"][: i + 1]},
            lookback=lookback,
        )
        assert not math.isnan(truncated[-1])
        assert truncated[-1] == pytest.approx(full[i], rel=1e-9, abs=1e-9)


class TestRollExtremeMidrange:
    def test_uses_rolling_high_low_extremes(self) -> None:
        bars = {
            "high": [10.0, 12.0, 11.0, 15.0],
            "low": [8.0, 9.0, 7.0, 14.0],
        }

        out = roll_extreme_midrange(bars, lookback=3)

        _assert_nan(out[0])
        _assert_nan(out[1])
        assert out[2] == pytest.approx((12.0 + 7.0) / 2.0)
        assert out[3] == pytest.approx((15.0 + 7.0) / 2.0)

    def test_nan_inputs_keep_output_nan(self) -> None:
        out = roll_extreme_midrange(
            {"high": [10.0, float("nan"), 12.0], "low": [8.0, 7.0, 9.0]},
            lookback=2,
        )

        _assert_nan(out[0])
        _assert_nan(out[1])
        _assert_nan(out[2])

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback must be positive"):
            roll_extreme_midrange({"high": [1.0], "low": [1.0]}, lookback=0)


class TestRollMeanMidrange:
    def test_uses_rolling_mean_of_bar_midranges(self) -> None:
        bars = {
            "high": [10.0, 12.0, 14.0, 16.0],
            "low": [8.0, 8.0, 10.0, 12.0],
        }

        out = roll_mean_midrange(bars, lookback=2)

        _assert_nan(out[0])
        assert out[1] == pytest.approx((9.0 + 10.0) / 2.0)
        assert out[2] == pytest.approx((10.0 + 12.0) / 2.0)
        assert out[3] == pytest.approx((12.0 + 14.0) / 2.0)

    def test_mismatched_bar_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            roll_mean_midrange({"high": [1.0, 2.0], "low": [1.0]}, lookback=2)
