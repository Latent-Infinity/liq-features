"""Tests for liq.features.alignment module."""

from datetime import UTC, datetime

import polars as pl

from liq.features.alignment import align_feature_frame, align_feature_frames, align_higher_timeframe


class TestAlignHigherTimeframe:
    """Tests for align_higher_timeframe function."""

    def test_basic_alignment(self) -> None:
        """Test basic alignment works correctly."""
        base = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
                ],
                "open": [1, 2, 3],
            }
        )
        higher = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
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
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "value": [10],
            }
        )
        result = align_higher_timeframe(base, higher)
        assert result.is_empty()

    def test_empty_higher_returns_base(self) -> None:
        """Test empty higher DataFrame returns base."""
        base = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
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
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                ],
                "value": [1, 2],
            }
        )
        higher = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "high_val": [100],
            }
        )
        aligned = align_higher_timeframe(base, higher, timestamp_col="ts")
        assert len(aligned) == 2
        assert aligned["high_val"].to_list() == [100, 100]


class TestAlignFeatureFrame:
    def test_aligns_symbol_aware_features(self) -> None:
        base = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                ],
                "symbol": ["BTC_USDT", "BTC_USDT", "ETH_USDT"],
                "close": [100.0, 101.0, 200.0],
            }
        )
        features = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                ],
                "symbol": ["BTC_USDT", "ETH_USDT"],
                "quote_mid": [99.5, 199.5],
            }
        )

        aligned = align_feature_frame(base, features)

        assert aligned["quote_mid"].to_list() == [99.5, 99.5, 199.5]

    def test_prefixes_and_shifts_feature_columns(self) -> None:
        base = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                ],
                "symbol": ["BTC_USDT", "BTC_USDT"],
            }
        )
        features = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                ],
                "symbol": ["BTC_USDT", "BTC_USDT"],
                "quote_mid": [100.0, 101.0],
            }
        )

        aligned = align_feature_frame(base, features, shift_periods=1, prefix="qs_")

        assert aligned["qs_quote_mid"].to_list() == [100.0, 100.0]


class TestAlignFeatureFrames:
    def test_aligns_multiple_feature_frames(self) -> None:
        base = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 1, tzinfo=UTC)],
                "symbol": ["BTC_USDT"],
            }
        )
        quote_features = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "symbol": ["BTC_USDT"],
                "quote_mid": [100.0],
            }
        )
        trade_features = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "symbol": ["BTC_USDT"],
                "trade_vwap": [100.1],
            }
        )

        aligned = align_feature_frames(
            base,
            {
                "quotes": quote_features,
                "trades": trade_features,
            },
        )

        assert aligned["quotes_quote_mid"].to_list() == [100.0]
        assert aligned["trades_trade_vwap"].to_list() == [100.1]
