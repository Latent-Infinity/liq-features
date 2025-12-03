from datetime import datetime, timezone

import polars as pl

from liq.features.alignment import align_higher_timeframe


def test_align_higher_timeframe_shift_and_fill() -> None:
    base = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            ],
            "open": [1, 2, 3],
        }
    )
    higher = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            ],
            "high_tf_value": [10, 20],
        }
    )
    aligned = align_higher_timeframe(base, higher, shift_periods=1)
    # Using completed higher bar (dropping most recent), all rows see last completed bar
    assert aligned["high_tf_value"].to_list() == [10, 10, 10]
