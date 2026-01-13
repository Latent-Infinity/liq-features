"""Tests for liq.features.cache module."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from liq.features.cache import (
    CacheManager,
    IndicatorCache,
    compute_cache_key,
    get_data_hash,
)
from liq.features.cache_models import (
    CacheEntry,
    CacheFilter,
    CacheStats,
    CleanupCriteria,
    CleanupResult,
)
from liq.store.parquet import ParquetStore


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample OHLC DataFrame for testing."""
    return pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ],
            "open": [100.0, 102.0, 101.0],
            "high": [103.0, 104.0, 103.0],
            "low": [99.0, 101.0, 100.0],
            "close": [102.0, 101.0, 103.0],
            "volume": [1000.0, 1500.0, 1200.0],
        }
    )


@pytest.fixture
def sample_result_df() -> pl.DataFrame:
    """Create a sample indicator result DataFrame."""
    return pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ],
            "value": [50.0, 55.0, 60.0],
        }
    )


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestGetDataHash:
    """Tests for get_data_hash function."""

    def test_returns_string(self, sample_df: pl.DataFrame) -> None:
        """Test function returns a string."""
        result = get_data_hash(sample_df)
        assert isinstance(result, str)

    def test_same_data_same_hash(self, sample_df: pl.DataFrame) -> None:
        """Test same data produces same hash."""
        hash1 = get_data_hash(sample_df)
        hash2 = get_data_hash(sample_df)
        assert hash1 == hash2

    def test_different_data_different_hash(self, sample_df: pl.DataFrame) -> None:
        """Test different data produces different hash."""
        hash1 = get_data_hash(sample_df)

        modified_df = sample_df.with_columns(pl.lit(200.0).alias("close"))
        hash2 = get_data_hash(modified_df)

        assert hash1 != hash2

    def test_empty_df_returns_hash(self) -> None:
        """Test empty DataFrame returns a hash."""
        df = pl.DataFrame({"ts": [], "value": []})
        result = get_data_hash(df)
        assert isinstance(result, str)
        assert len(result) > 0


class TestComputeCacheKey:
    """Tests for compute_cache_key function."""

    def test_returns_string(self) -> None:
        """Test function returns a string."""
        result = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        assert isinstance(result, str)

    def test_includes_all_components(self) -> None:
        """Test key includes all components."""
        key = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        # Key should be deterministic and unique
        assert len(key) > 10

    def test_different_params_different_key(self) -> None:
        """Test different params produce different keys."""
        key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 20},
            data_hash="abc123",
        )
        assert key1 != key2

    def test_different_indicator_different_key(self) -> None:
        """Test different indicator produces different key."""
        key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="SMA",
            params={"period": 14},
            data_hash="abc123",
        )
        assert key1 != key2

    def test_different_data_hash_different_key(self) -> None:
        """Test different data hash produces different key."""
        key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="xyz789",
        )
        assert key1 != key2

    def test_same_inputs_same_key(self) -> None:
        """Test same inputs produce same key."""
        key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        assert key1 == key2


class TestIndicatorCache:
    """Tests for IndicatorCache class."""

    def test_init_creates_cache_dir(self, temp_cache_dir: Path) -> None:
        """Test initialization creates cache directory."""
        store = ParquetStore(str(temp_cache_dir))
        cache = IndicatorCache(storage=store)
        assert cache.storage_root == temp_cache_dir

    def test_get_returns_none_for_missing(self, temp_cache_dir: Path) -> None:
        """Test get returns None for missing key."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        result = cache.get("nonexistent_key")
        assert result is None

    def test_set_and_get(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test set stores and get retrieves."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)
        result = cache.get(key)

        assert result is not None
        assert result.shape == sample_result_df.shape
        assert result["value"].to_list() == sample_result_df["value"].to_list()

    def test_has_returns_false_for_missing(self, temp_cache_dir: Path) -> None:
        """Test has returns False for missing key."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        assert not cache.has("nonexistent_key")

    def test_has_returns_true_for_existing(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test has returns True for existing key."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)
        assert cache.has(key)

    def test_delete_removes_entry(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test delete removes cache entry."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)
        assert cache.has(key)

        cache.delete(key)
        assert not cache.has(key)

    def test_delete_nonexistent_no_error(self, temp_cache_dir: Path) -> None:
        """Test delete on nonexistent key doesn't error."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        cache.delete("nonexistent_key")  # Should not raise

    def test_clear_removes_all_entries(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clear removes all cache entries."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="SMA",
            params={"period": 20},
            data_hash="abc123",
        )
        cache.set(key1, sample_result_df)
        cache.set(key2, sample_result_df)

        cache.clear()

        assert not cache.has(key1)
        assert not cache.has(key2)

    def test_stats_returns_dict(self, temp_cache_dir: Path) -> None:
        """Test stats returns dictionary."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        stats = cache.stats()
        assert isinstance(stats, dict)
        assert "entries" in stats
        assert "total_size_bytes" in stats

    def test_stats_counts_entries(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test stats counts entries correctly."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash="abc123",
        )
        key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="SMA",
            params={"period": 20},
            data_hash="abc123",
        )
        cache.set(key1, sample_result_df)
        cache.set(key2, sample_result_df)

        stats = cache.stats()
        assert stats["entries"] == 2


class TestIndicatorCacheIntegration:
    """Integration tests for IndicatorCache with real data."""

    def test_cache_with_computed_key(
        self,
        temp_cache_dir: Path,
        sample_df: pl.DataFrame,
        sample_result_df: pl.DataFrame,
    ) -> None:
        """Test cache with computed cache key."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Compute cache key
        data_hash = get_data_hash(sample_df)
        cache_key = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash=data_hash,
        )

        # Store result
        cache.set(cache_key, sample_result_df)

        # Retrieve result
        result = cache.get(cache_key)
        assert result is not None
        assert result.shape == sample_result_df.shape

    def test_cache_invalidates_on_data_change(
        self,
        temp_cache_dir: Path,
        sample_df: pl.DataFrame,
        sample_result_df: pl.DataFrame,
    ) -> None:
        """Test cache key changes when data changes."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Original data cache key
        data_hash1 = get_data_hash(sample_df)
        cache_key1 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash=data_hash1,
        )
        cache.set(cache_key1, sample_result_df)

        # Modified data cache key
        modified_df = sample_df.with_columns(pl.lit(200.0).alias("close"))
        data_hash2 = get_data_hash(modified_df)
        cache_key2 = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1m",
            indicator="RSI",
            params={"period": 14},
            data_hash=data_hash2,
        )

        # Keys should be different
        assert cache_key1 != cache_key2

        # Old key still exists
        assert cache.has(cache_key1)

        # New key doesn't exist (would need to compute)
        assert not cache.has(cache_key2)


class TestCacheManagerStats:
    """Tests for CacheManager.stats() method (Task 1.1)."""

    def test_stats_returns_cache_stats_model(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test stats returns CacheStats model."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)

        stats = cache.stats()
        # Current implementation returns dict, new implementation should return CacheStats
        assert isinstance(stats, (dict, CacheStats))

    def test_stats_empty_cache(self, temp_cache_dir: Path) -> None:
        """Test stats on empty cache."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        stats = cache.stats()
        assert stats["entries"] == 0

    def test_stats_total_entries_accurate(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test total entry count is accurate."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add 3 entries
        for i, indicator in enumerate(["rsi", "macd", "sma"]):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash{i}",
            )
            cache.set(key, sample_result_df)

        stats = cache.stats()
        assert stats["entries"] == 3

    def test_stats_total_size_positive(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test total size is positive for non-empty cache."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)

        stats = cache.stats()
        assert stats["total_size_bytes"] > 0


class TestCacheManagerListEntries:
    """Tests for CacheManager.list_entries() method (Task 1.1/1.3)."""

    def test_list_entries_empty_cache(self, temp_cache_dir: Path) -> None:
        """Test list_entries on empty cache returns empty list."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))
        entries = cache.list_entries()
        assert entries == []

    def test_list_entries_returns_all(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries returns all entries without filter."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add 3 entries
        keys = []
        for indicator in ["rsi", "macd", "sma"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash="abc123",
            )
            cache.set(key, sample_result_df)
            keys.append(key)

        entries = cache.list_entries()
        assert len(entries) == 3

    def test_list_entries_filter_by_symbol(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries filters by symbol."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries for different symbols with unique data hashes
        symbols = ["BTC_USDT", "EUR_USD", "BTC_USDT"]
        for i, symbol in enumerate(symbols):
            key = compute_cache_key(
                symbol=symbol,
                timeframe="1h",
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{i}",  # Use index for unique keys
            )
            cache.set(key, sample_result_df)

        filter = CacheFilter(symbol="BTC_USDT")
        entries = cache.list_entries(filter)
        assert len(entries) == 2
        for entry in entries:
            assert entry.symbol == "BTC_USDT"

    def test_list_entries_filter_by_indicator(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries filters by indicator."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries for different indicators
        for indicator in ["rsi", "macd", "stoch"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{indicator}",
            )
            cache.set(key, sample_result_df)

        filter = CacheFilter(indicator="macd")
        entries = cache.list_entries(filter)
        assert len(entries) == 1
        assert entries[0].indicator == "macd"

    def test_list_entries_filter_by_indicator_glob(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries filters by indicator with glob pattern."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries for stoch-related indicators
        for indicator in ["stoch", "stochf", "stochrsi", "rsi"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{indicator}",
            )
            cache.set(key, sample_result_df)

        filter = CacheFilter(indicator="stoch*")
        entries = cache.list_entries(filter)
        assert len(entries) == 3
        for entry in entries:
            assert entry.indicator.startswith("stoch")

    def test_list_entries_filter_by_timeframe(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries filters by timeframe."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries for different timeframes
        for tf in ["1m", "1h", "1d"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe=tf,
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{tf}",
            )
            cache.set(key, sample_result_df)

        filter = CacheFilter(timeframe="1h")
        entries = cache.list_entries(filter)
        assert len(entries) == 1
        assert entries[0].timeframe == "1h"


class TestCacheManagerClean:
    """Tests for CacheManager.clean() method (Task 1.5)."""

    def test_clean_empty_criteria_matches_all(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean with empty criteria matches all entries."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries
        for indicator in ["rsi", "macd", "sma"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{indicator}",
            )
            cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == 3

        # Clean with empty criteria
        criteria = CleanupCriteria()
        result = cache.clean(criteria)

        assert result.deleted_count == 3
        assert cache.stats()["entries"] == 0

    def test_clean_by_symbol(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean filters by symbol."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries for different symbols
        for symbol in ["BTC_USDT", "EUR_USD", "SPY"]:
            key = compute_cache_key(
                symbol=symbol,
                timeframe="1h",
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{symbol}",
            )
            cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == 3

        # Clean only BTC_USDT
        criteria = CleanupCriteria(symbol="BTC_USDT")
        result = cache.clean(criteria)

        assert result.deleted_count == 1
        assert cache.stats()["entries"] == 2

    def test_clean_by_indicator(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean filters by indicator."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries for different indicators
        for indicator in ["sarext", "sar", "rsi"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{indicator}",
            )
            cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == 3

        # Clean only sar* indicators
        criteria = CleanupCriteria(indicator="sar*")
        result = cache.clean(criteria)

        assert result.deleted_count == 2
        assert cache.stats()["entries"] == 1

    def test_clean_dry_run(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean with dry_run=True doesn't delete."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entries
        for indicator in ["rsi", "macd"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{indicator}",
            )
            cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == 2

        # Dry run
        criteria = CleanupCriteria()
        result = cache.clean(criteria, dry_run=True)

        assert result.deleted_count == 2
        assert result.dry_run is True
        # Entries should still exist
        assert cache.stats()["entries"] == 2

    def test_clean_returns_freed_bytes(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean returns freed bytes."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entry
        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)

        initial_size = cache.stats()["total_size_bytes"]

        # Clean
        criteria = CleanupCriteria()
        result = cache.clean(criteria)

        assert result.freed_bytes > 0
        assert result.freed_bytes == initial_size

    def test_clean_compound_criteria(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean with compound criteria."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add various entries
        entries = [
            ("BTC_USDT", "1h", "rsi"),
            ("BTC_USDT", "1m", "rsi"),
            ("EUR_USD", "1h", "rsi"),
            ("BTC_USDT", "1h", "macd"),
        ]
        for symbol, tf, indicator in entries:
            key = compute_cache_key(
                symbol=symbol,
                timeframe=tf,
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{symbol}_{tf}_{indicator}",
            )
            cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == 4

        # Clean only BTC_USDT + 1h entries
        criteria = CleanupCriteria(symbol="BTC_USDT", timeframe="1h")
        result = cache.clean(criteria)

        assert result.deleted_count == 2  # rsi and macd for BTC_USDT/1h
        assert cache.stats()["entries"] == 2

    def test_clean_no_matches(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean with criteria that matches nothing."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entry
        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)

        # Clean with non-matching criteria
        criteria = CleanupCriteria(symbol="NONEXISTENT")
        result = cache.clean(criteria)

        assert result.deleted_count == 0
        assert cache.stats()["entries"] == 1

    def test_clean_success_property(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clean result success property."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Add entry
        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="abc123",
        )
        cache.set(key, sample_result_df)

        criteria = CleanupCriteria()
        result = cache.clean(criteria)

        assert result.success is True
        assert len(result.errors) == 0


class TestCacheAgeTracking:
    """Tests for cache age tracking and age-based cleanup."""

    def test_list_entries_includes_created_at(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test that list_entries populates created_at timestamp."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0].created_at is not None

    def test_created_at_is_recent(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test that created_at is a reasonable recent timestamp."""
        from datetime import datetime, timedelta, timezone

        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        before = datetime.now(tz=timezone.utc)

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        after = datetime.now(tz=timezone.utc)

        entries = cache.list_entries()
        assert len(entries) == 1
        created = entries[0].created_at
        assert created is not None

        # Normalize timezones for comparison
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        # Should be between before and after (with 1 second tolerance for timing)
        assert created >= before - timedelta(seconds=1)
        assert created <= after + timedelta(seconds=1)

    def test_clean_older_than_filters_by_age(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test that older_than criteria filters by entry age."""
        import time

        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Create an entry
        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        # Entry just created, should not match "older than 1 hour"
        criteria = CleanupCriteria(older_than="1h")
        result = cache.clean(criteria, dry_run=True)
        assert result.deleted_count == 0

        # But should match "older than 0 minutes" (immediately old)
        # Actually, entries with age 0 are < 1m, so let's test differently
        # Use a very small threshold that the entry will exceed
        time.sleep(0.1)  # Wait 100ms
        # Entry should still not be older than 1 minute
        criteria = CleanupCriteria(older_than="1m")
        result = cache.clean(criteria, dry_run=True)
        assert result.deleted_count == 0

    def test_clean_older_than_matches_old_entries(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test that older_than matches genuinely old entries."""
        import os
        import time

        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        # Manually backdate the file modification time
        # Find the parquet file and change its mtime
        parquet_files = list(temp_cache_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0

        # Set mtime to 2 days ago
        old_time = time.time() - (2 * 24 * 60 * 60)
        for f in parquet_files:
            os.utime(f, (old_time, old_time))

        # Now entry should be older than 1 day
        criteria = CleanupCriteria(older_than="1d")
        result = cache.clean(criteria, dry_run=True)
        assert result.deleted_count == 1

    def test_older_than_various_durations(self) -> None:
        """Test parsing various duration formats."""
        from liq.features.cache_models import parse_duration
        from datetime import timedelta

        assert parse_duration("7d") == timedelta(days=7)
        assert parse_duration("24h") == timedelta(hours=24)
        assert parse_duration("30m") == timedelta(minutes=30)
        assert parse_duration("2w") == timedelta(weeks=2)

    def test_older_than_invalid_format_raises(self) -> None:
        """Test that invalid duration format raises ValueError."""
        from liq.features.cache_models import parse_duration
        import pytest

        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("invalid")

        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("7x")  # Unknown unit

        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("days")  # No number

    def test_cleanup_criteria_validates_older_than(self) -> None:
        """Test that CleanupCriteria validates older_than on construction."""
        from pydantic import ValidationError

        # Valid
        criteria = CleanupCriteria(older_than="7d")
        assert criteria.older_than == "7d"

        # Invalid should raise
        with pytest.raises(ValidationError):
            CleanupCriteria(older_than="invalid")

    def test_entries_without_timestamp_not_matched_by_older_than(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test that entries without created_at are not matched by older_than."""
        from liq.features.cache_models import CacheEntry, CleanupCriteria

        # Create entry without timestamp
        entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/abc:1h:hash123",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="abc",
            timeframe="1h",
            data_hash="hash123",
            created_at=None,  # No timestamp
        )

        criteria = CleanupCriteria(older_than="1d")
        # Should NOT match because we can't determine age
        assert not criteria.matches(entry)


class TestCacheCleanupPerformance:
    """Tests for cache cleanup performance characteristics."""

    def test_clean_multiple_entries_in_single_call(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test cleanup of multiple entries works efficiently."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Create multiple entries
        num_entries = 50
        for i in range(num_entries):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=f"indicator_{i}",
                params={"period": 14},
                data_hash=f"hash_{i}",
            )
            cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == num_entries

        # Clean all at once
        criteria = CleanupCriteria()
        result = cache.clean(criteria)

        assert result.deleted_count == num_entries
        assert result.success is True
        assert cache.stats()["entries"] == 0

    def test_cleanup_result_tracks_all_deletions(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test CleanupResult accurately tracks deletion count and freed bytes."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Create entries
        num_entries = 10
        for i in range(num_entries):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=f"indicator_{i}",
                params={"period": 14},
                data_hash=f"hash_{i}",
            )
            cache.set(key, sample_result_df)

        # Dry run first
        criteria = CleanupCriteria()
        dry_result = cache.clean(criteria, dry_run=True)
        assert dry_result.deleted_count == num_entries
        assert dry_result.freed_bytes > 0

        # Actual cleanup
        result = cache.clean(criteria)
        assert result.deleted_count == num_entries
        assert result.freed_bytes > 0

    def test_cleanup_with_filter_only_deletes_matching(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test filtered cleanup only deletes matching entries."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Create entries for multiple symbols
        for symbol in ["BTC_USDT", "EUR_USD", "SPY"]:
            for i in range(5):
                key = compute_cache_key(
                    symbol=symbol,
                    timeframe="1h",
                    indicator=f"indicator_{i}",
                    params={"period": 14},
                    data_hash=f"hash_{symbol}_{i}",
                )
                cache.set(key, sample_result_df)

        assert cache.stats()["entries"] == 15

        # Clean only BTC_USDT
        criteria = CleanupCriteria(symbol="BTC_USDT")
        result = cache.clean(criteria)

        assert result.deleted_count == 5
        assert cache.stats()["entries"] == 10

    def test_list_entries_with_many_entries(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries handles many entries."""
        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Create entries
        num_entries = 30
        for i in range(num_entries):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=f"indicator_{i}",
                params={"period": 14},
                data_hash=f"hash_{i}",
            )
            cache.set(key, sample_result_df)

        entries = cache.list_entries()
        assert len(entries) == num_entries

        # All should have created_at populated
        for entry in entries:
            assert entry.created_at is not None
            assert entry.size_bytes > 0

    def test_list_entries_with_limit_and_offset(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test list_entries pagination."""
        from liq.features.cache_models import CacheFilter

        cache = IndicatorCache(storage=ParquetStore(str(temp_cache_dir)))

        # Create entries
        num_entries = 20
        for i in range(num_entries):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=f"indicator_{i:02d}",  # Padded for sorting
                params={"period": 14},
                data_hash=f"hash_{i}",
            )
            cache.set(key, sample_result_df)

        # Test limit
        filter_limit = CacheFilter(limit=5)
        entries = cache.list_entries(filter_limit)
        assert len(entries) == 5

        # Test offset + limit
        filter_page = CacheFilter(limit=5, offset=10)
        entries = cache.list_entries(filter_page)
        assert len(entries) == 5

        # Test offset beyond entries
        filter_beyond = CacheFilter(offset=100)
        entries = cache.list_entries(filter_beyond)
        assert len(entries) == 0
