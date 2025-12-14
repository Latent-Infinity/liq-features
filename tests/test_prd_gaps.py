"""Tests for PRD gap closure features.

This module tests features required by the PRD but not yet implemented:
- Auto-registration of hardcoded indicators
- configure_defaults() for global parameter overrides
- compute_indicators() for batch computation
- cache_stats() for cache introspection
- include_incomplete parameter in Aggregator
"""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest


class TestAutoRegistration:
    """Tests for automatic indicator registration on import."""

    def test_get_indicator_without_explicit_imports(self) -> None:
        """Test that get_indicator works without explicitly importing indicator modules.

        This tests the critical gap where indicators/__init__.py must import
        momentum and trend modules to trigger @register_indicator decorators.
        """
        # Use a fresh import to simulate real usage
        from liq.features.indicators import get_indicator

        # These should work without importing momentum.py or trend.py explicitly
        RSI = get_indicator("rsi")
        assert RSI.name == "rsi"

        EMA = get_indicator("ema")
        assert EMA.name == "ema"

    def test_all_prd_indicators_registered(self) -> None:
        """Test all 8 PRD hardcoded indicators are auto-registered."""
        from liq.features.indicators import get_indicator

        prd_indicators = ["rsi", "macd", "stochastic", "ema", "sma", "bbands", "adx", "atr"]

        for name in prd_indicators:
            indicator_cls = get_indicator(name)
            assert indicator_cls.name == name, f"Expected {name} to be registered"

    def test_list_indicators_includes_hardcoded_without_explicit_imports(self) -> None:
        """Test list_indicators includes hardcoded indicators on first import."""
        from liq.features.indicators import list_indicators

        indicators = list_indicators()
        names = [ind["name"] for ind in indicators]

        # All PRD hardcoded indicators should be present
        assert "rsi" in names
        assert "macd" in names
        assert "ema" in names
        assert "atr" in names

        # Check at least some are marked as hardcoded
        hardcoded = [ind for ind in indicators if ind.get("source") == "hardcoded"]
        assert len(hardcoded) >= 8, f"Expected at least 8 hardcoded indicators, got {len(hardcoded)}"


class TestConfigureDefaults:
    """Tests for configure_defaults() global parameter override."""

    def test_configure_defaults_overrides_rsi_period(self) -> None:
        """Test configure_defaults changes RSI default period."""
        from liq.features import configure_defaults
        from liq.features.indicators import get_indicator

        # Override RSI default
        configure_defaults({"rsi": {"period": 21}})

        RSI = get_indicator("rsi")
        rsi = RSI()
        assert rsi.params["period"] == 21

        # Reset to original
        configure_defaults({"rsi": {"period": 14}})

    def test_configure_defaults_overrides_macd_params(self) -> None:
        """Test configure_defaults changes MACD default parameters."""
        from liq.features import configure_defaults
        from liq.features.indicators import get_indicator

        configure_defaults({
            "macd": {"fast_period": 8, "slow_period": 21, "signal_period": 5}
        })

        MACD = get_indicator("macd")
        macd = MACD()
        assert macd.params["fast_period"] == 8
        assert macd.params["slow_period"] == 21
        assert macd.params["signal_period"] == 5

        # Reset
        configure_defaults({
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        })

    def test_explicit_params_override_configured_defaults(self) -> None:
        """Test that explicit params take precedence over configured defaults."""
        from liq.features import configure_defaults
        from liq.features.indicators import get_indicator

        configure_defaults({"rsi": {"period": 21}})

        RSI = get_indicator("rsi")
        rsi = RSI(params={"period": 7})  # Explicit override
        assert rsi.params["period"] == 7

        # Reset
        configure_defaults({"rsi": {"period": 14}})

    def test_reset_defaults(self) -> None:
        """Test reset_defaults restores original values."""
        from liq.features import configure_defaults, reset_defaults
        from liq.features.indicators import get_indicator

        # Get original
        RSI = get_indicator("rsi")
        original = RSI.default_params.copy()

        # Override
        configure_defaults({"rsi": {"period": 50}})

        # Reset
        reset_defaults()

        RSI = get_indicator("rsi")
        rsi = RSI()
        assert rsi.params["period"] == original["period"]


class TestComputeIndicators:
    """Tests for compute_indicators() batch function."""

    @pytest.fixture
    def sample_bars(self) -> pl.DataFrame:
        """Create sample OHLCV bars for testing."""
        from datetime import timedelta
        import random
        random.seed(42)

        n = 100
        base_price = 100.0
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base_time + timedelta(minutes=i) for i in range(n)]

        prices = []
        price = base_price
        for _ in range(n):
            change = random.uniform(-0.5, 0.5)
            price += change
            prices.append(price)

        return pl.DataFrame({
            "ts": timestamps,
            "timestamp": timestamps,
            "open": [p - random.uniform(0, 0.2) for p in prices],
            "high": [p + random.uniform(0, 0.3) for p in prices],
            "low": [p - random.uniform(0, 0.3) for p in prices],
            "close": prices,
            "volume": [random.uniform(1000, 5000) for _ in range(n)],
        })

    def test_compute_indicators_returns_merged_dataframe(
        self, sample_bars: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test compute_indicators returns merged DataFrame with all columns."""
        from liq.features import compute_indicators
        from liq.store.parquet import ParquetStore

        storage = ParquetStore(str(tmp_path))

        result = compute_indicators(
            bars=sample_bars,
            symbol="EUR_USD",
            timeframe="1m",
            indicators=[
                ("rsi", {"period": 14}),
                ("ema", {"period": 10}),
            ],
            storage=storage,
        )

        assert isinstance(result, pl.DataFrame)
        assert "ts" in result.columns or "timestamp" in result.columns
        # Check indicator columns are present
        assert any("rsi" in col.lower() for col in result.columns)
        assert any("ema" in col.lower() or "value" in col.lower() for col in result.columns)

    def test_compute_indicators_uses_cache(
        self, sample_bars: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test compute_indicators caches results."""
        from liq.features import compute_indicators
        from liq.store.parquet import ParquetStore

        storage = ParquetStore(str(tmp_path))

        # First call
        result1 = compute_indicators(
            bars=sample_bars,
            symbol="EUR_USD",
            timeframe="1m",
            indicators=[("rsi", {"period": 14})],
            storage=storage,
        )

        # Second call should use cache
        result2 = compute_indicators(
            bars=sample_bars,
            symbol="EUR_USD",
            timeframe="1m",
            indicators=[("rsi", {"period": 14})],
            storage=storage,
        )

        # Results should be identical
        assert result1.shape == result2.shape

    def test_compute_indicators_error_on_unknown(
        self, sample_bars: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test compute_indicators raises on unknown indicator."""
        from liq.features import compute_indicators
        from liq.store.parquet import ParquetStore

        storage = ParquetStore(str(tmp_path))

        with pytest.raises(ValueError, match="Unknown indicator"):
            compute_indicators(
                bars=sample_bars,
                symbol="EUR_USD",
                timeframe="1m",
                indicators=[("unknown_xyz", {})],
                storage=storage,
            )


class TestCacheStats:
    """Tests for cache_stats() function."""

    def test_cache_stats_returns_dataframe(self, tmp_path: Path) -> None:
        """Test cache_stats returns DataFrame with expected columns."""
        from liq.features import cache_stats
        from liq.store.parquet import ParquetStore

        storage = ParquetStore(str(tmp_path))

        result = cache_stats(storage)

        assert isinstance(result, pl.DataFrame)
        expected_cols = {"indicator", "timeframe", "params_id", "row_count", "size_mb"}
        assert expected_cols.issubset(set(result.columns))

    def test_cache_stats_empty_when_no_cache(self, tmp_path: Path) -> None:
        """Test cache_stats returns empty DataFrame when no cache exists."""
        from liq.features import cache_stats
        from liq.store.parquet import ParquetStore

        storage = ParquetStore(str(tmp_path))

        result = cache_stats(storage)

        assert len(result) == 0

    def test_cache_stats_counts_after_computation(self, tmp_path: Path) -> None:
        """Test cache_stats shows entries after indicator computation."""
        from datetime import timedelta
        from liq.features import cache_stats
        from liq.features.indicators import get_indicator
        from liq.store.parquet import ParquetStore

        storage = ParquetStore(str(tmp_path))

        # Compute an indicator
        df = pl.DataFrame({
            "ts": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(50)],
            "close": [100.0 + i * 0.1 for i in range(50)],
            "high": [101.0 + i * 0.1 for i in range(50)],
            "low": [99.0 + i * 0.1 for i in range(50)],
        })

        RSI = get_indicator("rsi")
        rsi = RSI(storage=storage, params={"period": 14})
        rsi.compute(df, symbol="EUR_USD", timeframe="1m")

        result = cache_stats(storage)

        assert len(result) >= 1


class TestIncompleteBarHandling:
    """Tests for include_incomplete parameter in Aggregator."""

    @pytest.fixture
    def minute_bars(self) -> pl.DataFrame:
        """Create minute bars that don't align to full hours."""
        from datetime import timedelta

        # 75 minutes of data = 1 complete hour + 15 minutes partial
        timestamps = [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
            for i in range(75)
        ]

        return pl.DataFrame({
            "timestamp": timestamps,
            "open": [100.0 + i * 0.01 for i in range(75)],
            "high": [100.5 + i * 0.01 for i in range(75)],
            "low": [99.5 + i * 0.01 for i in range(75)],
            "close": [100.2 + i * 0.01 for i in range(75)],
            "volume": [1000.0 for _ in range(75)],
        })

    def test_include_incomplete_true_includes_partial_bar(
        self, minute_bars: pl.DataFrame
    ) -> None:
        """Test include_incomplete=True includes the partial final bar."""
        from liq.features.aggregation import Aggregator

        agg = Aggregator(
            source_timeframe="1m",
            target_timeframe="1h",
        )

        result = agg.aggregate(minute_bars, include_incomplete=True)

        # Should have 2 bars: 00:00-01:00 (complete) and 01:00-02:00 (partial)
        assert len(result) == 2

    def test_include_incomplete_false_excludes_partial_bar(
        self, minute_bars: pl.DataFrame
    ) -> None:
        """Test include_incomplete=False excludes the partial final bar."""
        from liq.features.aggregation import Aggregator

        agg = Aggregator(
            source_timeframe="1m",
            target_timeframe="1h",
        )

        result = agg.aggregate(minute_bars, include_incomplete=False)

        # Should only have 1 bar: 00:00-01:00 (complete)
        assert len(result) == 1

    def test_include_incomplete_default_is_false(
        self, minute_bars: pl.DataFrame
    ) -> None:
        """Test include_incomplete defaults to False per PRD."""
        from liq.features.aggregation import Aggregator

        agg = Aggregator(
            source_timeframe="1m",
            target_timeframe="1h",
        )

        # Call without explicit parameter
        result = agg.aggregate(minute_bars)

        # Should behave like include_incomplete=False
        assert len(result) == 1

    def test_all_complete_bars(self) -> None:
        """Test aggregation when all bars form complete periods."""
        from datetime import timedelta
        from liq.features.aggregation import Aggregator

        # Exactly 2 hours of data
        timestamps = [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
            for i in range(120)
        ]

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * 120,
            "high": [101.0] * 120,
            "low": [99.0] * 120,
            "close": [100.5] * 120,
            "volume": [1000.0] * 120,
        })

        agg = Aggregator(source_timeframe="1m", target_timeframe="1h")

        result_include = agg.aggregate(df, include_incomplete=True)
        result_exclude = agg.aggregate(df, include_incomplete=False)

        # Both should have same count when all bars are complete
        assert len(result_include) == len(result_exclude) == 2

    def test_only_partial_period(self) -> None:
        """Test aggregation with only partial period data."""
        from datetime import timedelta
        from liq.features.aggregation import Aggregator

        # Only 30 minutes of data
        timestamps = [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
            for i in range(30)
        ]

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.5] * 30,
            "volume": [1000.0] * 30,
        })

        agg = Aggregator(source_timeframe="1m", target_timeframe="1h")

        result_include = agg.aggregate(df, include_incomplete=True)
        result_exclude = agg.aggregate(df, include_incomplete=False)

        assert len(result_include) == 1  # Partial bar included
        assert len(result_exclude) == 0  # No complete bars

    def test_convenience_function_supports_include_incomplete(self) -> None:
        """Test aggregate_to_timeframe supports include_incomplete parameter."""
        from datetime import timedelta
        from liq.features.aggregation import aggregate_to_timeframe

        # 75 minutes
        timestamps = [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
            for i in range(75)
        ]

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * 75,
            "high": [101.0] * 75,
            "low": [99.0] * 75,
            "close": [100.5] * 75,
            "volume": [1000.0] * 75,
        })

        result = aggregate_to_timeframe(
            df, "1m", "1h", include_incomplete=False
        )

        assert len(result) == 1
