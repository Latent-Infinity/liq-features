"""Tests for liq.features.microstructure module."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from liq.features.microstructure import (
    build_funding_features,
    build_open_interest_features,
    build_order_book_features,
    build_quote_features,
    build_trade_bar_features,
    corwin_schultz_spread,
)


class TestCorwinSchultzSpread:
    """Tests for corwin_schultz_spread function."""

    def test_positive_spread(self) -> None:
        df = pl.DataFrame(
            {
                "high": pl.Series("high", [2.0, 2.2, 2.4], dtype=pl.Float64),
                "low": pl.Series("low", [1.8, 1.9, 2.0], dtype=pl.Float64),
                "open": pl.Series("open", [2.0, 2.0, 2.0], dtype=pl.Float64),
                "close": pl.Series("close", [2.0, 2.0, 2.0], dtype=pl.Float64),
            }
        )
        spread = corwin_schultz_spread(df)
        assert spread >= 0

    def test_single_row_returns_zero(self) -> None:
        df = pl.DataFrame({"high": [2.0], "low": [1.8]})
        assert corwin_schultz_spread(df) == 0.0

    def test_empty_returns_zero(self) -> None:
        df = pl.DataFrame(
            {"high": pl.Series([], dtype=pl.Float64), "low": pl.Series([], dtype=pl.Float64)}
        )
        assert corwin_schultz_spread(df) == 0.0

    def test_equal_high_low_returns_zero(self) -> None:
        df = pl.DataFrame({"high": [2.0, 2.0, 2.0], "low": [2.0, 2.0, 2.0]})
        assert corwin_schultz_spread(df) == 0.0

    def test_returns_float(self) -> None:
        df = pl.DataFrame({"high": [2.0, 2.2, 2.4, 2.3], "low": [1.8, 1.9, 2.0, 2.1]})
        assert isinstance(corwin_schultz_spread(df), float)


class TestBuildQuoteFeatures:
    def test_build_quote_features_adds_expected_columns(self) -> None:
        ts0 = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts0, ts0 + timedelta(minutes=1)],
                "symbol": ["BTC_USDT", "BTC_USDT"],
                "bid": [100.0, 101.0],
                "ask": [100.2, 101.3],
                "bid_size": [5.0, 6.0],
                "ask_size": [4.0, 3.0],
            }
        )

        out = build_quote_features(df)

        for col in [
            "quote_mid",
            "quoted_spread",
            "quoted_spread_bps",
            "mid_return",
            "spread_change",
            "quote_imbalance",
            "microprice",
            "microprice_edge_bps",
        ]:
            assert col in out.columns
        assert out["quote_mid"].to_list()[0] == pytest.approx(100.1)
        assert out["quoted_spread"].to_list()[0] == pytest.approx(0.2)
        assert out["quote_imbalance"].to_list()[0] == pytest.approx((5.0 - 4.0) / 9.0)

    def test_build_quote_features_requires_bid_ask(self) -> None:
        with pytest.raises(ValueError, match="Missing required columns"):
            build_quote_features(pl.DataFrame({"timestamp": [], "bid": []}))


class TestBuildTradeBarFeatures:
    def test_build_trade_bar_features_aggregates_windows(self) -> None:
        ts0 = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts0, ts0 + timedelta(seconds=10), ts0 + timedelta(minutes=1)],
                "symbol": ["BTC_USDT", "BTC_USDT", "BTC_USDT"],
                "price": [100.0, 101.0, 102.0],
                "quantity": [1.0, 2.0, 1.0],
                "side": ["buy", "sell", "buy"],
            }
        )

        out = build_trade_bar_features(df, every="1m")

        assert out.height == 2
        assert "trade_vwap" in out.columns
        assert "trade_imbalance" in out.columns
        first = out.row(0, named=True)
        assert first["trade_count"] == 2
        assert first["trade_volume"] == pytest.approx(3.0)
        assert first["trade_vwap"] == pytest.approx((100.0 * 1.0 + 101.0 * 2.0) / 3.0)
        assert first["trade_imbalance"] == pytest.approx((1.0 - 2.0) / 3.0)

    def test_build_trade_bar_features_requires_side(self) -> None:
        with pytest.raises(ValueError, match="Missing required columns"):
            build_trade_bar_features(
                pl.DataFrame(
                    {
                        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                        "price": [100.0],
                        "quantity": [1.0],
                    }
                )
            )


class TestBuildOrderBookFeatures:
    def test_build_order_book_features_collapses_snapshots(self) -> None:
        ts0 = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts0, ts0, ts0, ts0],
                "symbol": ["BTC_USDT"] * 4,
                "snapshot_id": [1, 1, 1, 1],
                "side": ["bid", "bid", "ask", "ask"],
                "price": [100.0, 99.5, 100.2, 100.5],
                "size": [2.0, 1.0, 1.5, 0.5],
                "level": [0, 1, 0, 1],
            }
        )

        out = build_order_book_features(df, depth=2)

        assert out.height == 1
        row = out.row(0, named=True)
        assert row["best_bid"] == pytest.approx(100.0)
        assert row["best_ask"] == pytest.approx(100.2)
        assert row["quoted_spread"] == pytest.approx(0.2)
        assert row["depth_imbalance"] == pytest.approx((3.0 - 2.0) / 5.0)

    def test_build_order_book_features_requires_positive_depth(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "snapshot_id": [1],
                "side": ["bid"],
                "price": [100.0],
                "size": [1.0],
                "level": [0],
            }
        )
        with pytest.raises(ValueError, match="depth must be positive"):
            build_order_book_features(df, depth=0)


class TestBuildFundingFeatures:
    def test_build_funding_features_adds_rate_state(self) -> None:
        ts0 = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts0, ts0 + timedelta(hours=8)],
                "symbol": ["BTC_USDT", "BTC_USDT"],
                "funding_rate": [0.0001, -0.0002],
                "mark_price": [100.0, 105.0],
            }
        )

        out = build_funding_features(df)

        assert out["funding_rate_bps"].to_list() == pytest.approx([1.0, -2.0])
        assert out["funding_direction"].to_list() == [1, -1]
        assert out["funding_rate_change"].to_list()[0] == pytest.approx(0.0)
        assert out["mark_price_return"].to_list()[1] == pytest.approx(0.05)


class TestBuildOpenInterestFeatures:
    def test_build_open_interest_features_adds_deltas(self) -> None:
        ts0 = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts0, ts0 + timedelta(minutes=5)],
                "symbol": ["BTC_USDT", "BTC_USDT"],
                "open_interest": [1000.0, 1100.0],
                "open_interest_value": [100000.0, 121000.0],
                "circulating_supply": [10000.0, 10000.0],
            }
        )

        out = build_open_interest_features(df)

        assert out["open_interest_change"].to_list() == pytest.approx([0.0, 100.0])
        assert out["open_interest_pct_change"].to_list()[1] == pytest.approx(0.1)
        assert out["open_interest_value_pct_change"].to_list()[1] == pytest.approx(0.21)
        assert out["oi_to_supply"].to_list() == pytest.approx([0.1, 0.11])
