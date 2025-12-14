"""Tests for liq.features.alignment module."""

from datetime import datetime, timezone

import polars as pl

from liq.features.alignment import align_higher_timeframe


class TestAlignHigherTimeframe:
    """Tests for align_higher_timeframe function."""

    def test_basic_alignment(self) -> None:
        """Test basic alignment works correctly."""
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
        assert aligned["high_tf_value"].to_list() == [10, 10, 10]

    def test_empty_base_returns_base(self) -> None:
        """Test empty base DataFrame returns base."""
        base = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "open": pl.Series([], dtype=pl.Float64),
            }
        )
        higher = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "value": [10],
            }
        )
        result = align_higher_timeframe(base, higher)
        assert result.is_empty()

    def test_empty_higher_returns_base(self) -> None:
        """Test empty higher DataFrame returns base."""
        base = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "open": [1],
            }
        )
        higher = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )
        result = align_higher_timeframe(base, higher)
        assert len(result) == 1
        assert result["open"].to_list() == [1]

    def test_with_custom_timestamp_col(self) -> None:
        """Test with custom timestamp column name."""
        base = pl.DataFrame(
            {
                "ts": [
                    datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                ],
                "value": [1, 2],
            }
        )
        higher = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "high_val": [100],
            }
        )
        aligned = align_higher_timeframe(base, higher, timestamp_col="ts")
        assert len(aligned) == 2
        assert aligned["high_val"].to_list() == [100, 100]
