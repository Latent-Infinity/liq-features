"""Tests for liq.features.aggregation module."""

from datetime import UTC, datetime

import polars as pl
import pytest

from liq.features.aggregation import Aggregator, aggregate_to_timeframe


class TestAggregatorCreation:
    """Tests for Aggregator instantiation."""

    def test_create_aggregator(self) -> None:
        """Test basic aggregator creation."""
        agg = Aggregator(source_timeframe="1min", target_timeframe="1h")

        assert agg.source_timeframe == "1min"
        assert agg.target_timeframe == "1h"

    def test_invalid_source_timeframe(self) -> None:
        """Test invalid source timeframe raises error."""
        with pytest.raises(ValueError, match="Unknown source timeframe"):
            Aggregator(source_timeframe="invalid", target_timeframe="1h")

    def test_invalid_target_timeframe(self) -> None:
        """Test invalid target timeframe raises error."""
        with pytest.raises(ValueError, match="Unknown target timeframe"):
            Aggregator(source_timeframe="1min", target_timeframe="invalid")

    def test_target_must_be_larger(self) -> None:
        """Test target timeframe must be larger than source."""
        with pytest.raises(ValueError, match="must be larger"):
            Aggregator(source_timeframe="1h", target_timeframe="1min")

    def test_equal_timeframes_invalid(self) -> None:
        """Test equal timeframes are invalid."""
        with pytest.raises(ValueError, match="must be larger"):
            Aggregator(source_timeframe="1h", target_timeframe="1h")


class TestAggregatorAggregate:
    """Tests for Aggregator.aggregate method."""

    def test_aggregate_minute_to_hourly(self, sample_minute_df: pl.DataFrame) -> None:
        """Test aggregating minute data to hourly."""
        agg = Aggregator(source_timeframe="1min", target_timeframe="1h")

        result = agg.aggregate(sample_minute_df)

        # 60 minutes should become 1 hour
        assert len(result) == 1
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns

    def test_ohlc_aggregation_rules(self) -> None:
        """Test OHLC aggregation follows correct rules."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 2, 0, tzinfo=UTC),
            ],
            "open": [100.0, 102.0, 101.0],
            "high": [105.0, 103.0, 104.0],
            "low": [99.0, 100.0, 98.0],
            "close": [102.0, 101.0, 103.0],
        })

        agg = Aggregator(source_timeframe="1min", target_timeframe="1h")
        # include_incomplete=True to test aggregation logic with partial data
        result = agg.aggregate(df, include_incomplete=True)

        assert result["open"][0] == 100.0  # First open
        assert result["high"][0] == 105.0  # Max high
        assert result["low"][0] == 98.0    # Min low
        assert result["close"][0] == 103.0  # Last close

    def test_volume_aggregation(self) -> None:
        """Test volume is summed."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, 0, tzinfo=UTC),
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1500.0],
        })

        agg = Aggregator(source_timeframe="1min", target_timeframe="1h")
        # include_incomplete=True to test volume aggregation with partial data
        result = agg.aggregate(df, include_incomplete=True)

        assert result["volume"][0] == 2500.0  # Sum of volumes

    def test_missing_columns_raises_error(self) -> None:
        """Test missing columns raise error."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)],
            "open": [100.0],
        })

        agg = Aggregator(source_timeframe="1min", target_timeframe="1h")

        with pytest.raises(ValueError, match="Missing required columns"):
            agg.aggregate(df)

    def test_empty_dataframe(self) -> None:
        """Test empty DataFrame returns empty."""
        df = pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
        })

        agg = Aggregator(source_timeframe="1min", target_timeframe="1h")
        result = agg.aggregate(df)

        assert result.is_empty()

    def test_custom_timestamp_column(self) -> None:
        """Test custom timestamp column name."""
        df = pl.DataFrame({
            "ts": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, 0, tzinfo=UTC),
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        })

        agg = Aggregator(
            source_timeframe="1min",
            target_timeframe="1h",
            timestamp_col="ts",
        )
        result = agg.aggregate(df)

        assert "ts" in result.columns


class TestAggregateToTimeframe:
    """Tests for aggregate_to_timeframe convenience function."""

    def test_convenience_function(self) -> None:
        """Test convenience function works like Aggregator."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, 0, tzinfo=UTC),
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        })

        # include_incomplete=True to test with partial data
        result = aggregate_to_timeframe(df, "1min", "1h", include_incomplete=True)

        assert len(result) == 1
        assert result["open"][0] == 100.0
