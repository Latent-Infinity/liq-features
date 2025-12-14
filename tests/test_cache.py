"""Tests for liq.features.cache module."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from liq.features.cache import (
    IndicatorCache,
    compute_cache_key,
    get_data_hash,
)


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
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        assert temp_cache_dir.exists()

    def test_get_returns_none_for_missing(self, temp_cache_dir: Path) -> None:
        """Test get returns None for missing key."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        result = cache.get("nonexistent_key")
        assert result is None

    def test_set_and_get(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test set stores and get retrieves."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)

        cache.set("test_key", sample_result_df)
        result = cache.get("test_key")

        assert result is not None
        assert result.shape == sample_result_df.shape
        assert result["value"].to_list() == sample_result_df["value"].to_list()

    def test_has_returns_false_for_missing(self, temp_cache_dir: Path) -> None:
        """Test has returns False for missing key."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        assert not cache.has("nonexistent_key")

    def test_has_returns_true_for_existing(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test has returns True for existing key."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        cache.set("test_key", sample_result_df)
        assert cache.has("test_key")

    def test_delete_removes_entry(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test delete removes cache entry."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        cache.set("test_key", sample_result_df)
        assert cache.has("test_key")

        cache.delete("test_key")
        assert not cache.has("test_key")

    def test_delete_nonexistent_no_error(self, temp_cache_dir: Path) -> None:
        """Test delete on nonexistent key doesn't error."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        cache.delete("nonexistent_key")  # Should not raise

    def test_clear_removes_all_entries(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test clear removes all cache entries."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        cache.set("key1", sample_result_df)
        cache.set("key2", sample_result_df)

        cache.clear()

        assert not cache.has("key1")
        assert not cache.has("key2")

    def test_stats_returns_dict(self, temp_cache_dir: Path) -> None:
        """Test stats returns dictionary."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        stats = cache.stats()
        assert isinstance(stats, dict)
        assert "entries" in stats
        assert "total_size_bytes" in stats

    def test_stats_counts_entries(
        self, temp_cache_dir: Path, sample_result_df: pl.DataFrame
    ) -> None:
        """Test stats counts entries correctly."""
        cache = IndicatorCache(cache_dir=temp_cache_dir)
        cache.set("key1", sample_result_df)
        cache.set("key2", sample_result_df)

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
        cache = IndicatorCache(cache_dir=temp_cache_dir)

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
        cache = IndicatorCache(cache_dir=temp_cache_dir)

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
