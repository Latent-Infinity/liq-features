from datetime import datetime, timezone, timedelta

from liq.features.indicators.zigzag import zigzag_pivots


def test_zigzag_basic_reversals() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [ts0 + timedelta(minutes=i) for i in range(6)]
    prices = [100, 102, 105, 102, 99, 104]

    pivots = zigzag_pivots(timestamps, prices, pct=0.02, symbol="TEST")

    assert len(pivots) == 2
    assert pivots[0].direction == "short"
    assert pivots[1].direction == "long"
    assert pivots[0].timestamp == timestamps[2]
    assert pivots[1].timestamp == timestamps[4]


def test_zigzag_requires_matching_lengths() -> None:
    try:
        zigzag_pivots([datetime.now(timezone.utc)], [], pct=0.01, symbol="X")
    except ValueError:
        return
    assert False, "Expected ValueError for mismatched lengths"
