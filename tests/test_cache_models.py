"""Tests for cache management data models."""

from datetime import datetime, timedelta, timezone

import pytest

from liq.features.cache_models import (
    CacheEntry,
    CacheFilter,
    CacheStats,
    CleanupCriteria,
    CleanupResult,
    parse_duration,
)


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_parse_days(self) -> None:
        """Test parsing days."""
        assert parse_duration("7d") == timedelta(days=7)
        assert parse_duration("1d") == timedelta(days=1)
        assert parse_duration("30d") == timedelta(days=30)

    def test_parse_hours(self) -> None:
        """Test parsing hours."""
        assert parse_duration("24h") == timedelta(hours=24)
        assert parse_duration("1h") == timedelta(hours=1)
        assert parse_duration("48h") == timedelta(hours=48)

    def test_parse_minutes(self) -> None:
        """Test parsing minutes."""
        assert parse_duration("30m") == timedelta(minutes=30)
        assert parse_duration("1m") == timedelta(minutes=1)
        assert parse_duration("60m") == timedelta(minutes=60)

    def test_parse_weeks(self) -> None:
        """Test parsing weeks."""
        assert parse_duration("1w") == timedelta(weeks=1)
        assert parse_duration("2w") == timedelta(weeks=2)

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert parse_duration("7D") == timedelta(days=7)
        assert parse_duration("24H") == timedelta(hours=24)

    def test_with_whitespace(self) -> None:
        """Test handling of whitespace."""
        assert parse_duration(" 7d ") == timedelta(days=7)

    def test_invalid_format(self) -> None:
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("invalid")

    def test_no_number(self) -> None:
        """Test missing number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("d")

    def test_no_unit(self) -> None:
        """Test missing unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("7")

    def test_unknown_unit(self) -> None:
        """Test unknown unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("7x")


class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_from_key_valid(self) -> None:
        """Test parsing a valid cache key."""
        key = "BTC_USDT/indicators/rsi/a1b2c3d4:1h:abcd1234"
        entry = CacheEntry.from_key(key, size_bytes=1024)

        assert entry is not None
        assert entry.key == key
        assert entry.symbol == "BTC_USDT"
        assert entry.indicator == "rsi"
        assert entry.params_hash == "a1b2c3d4"
        assert entry.timeframe == "1h"
        assert entry.data_hash == "abcd1234"
        assert entry.size_bytes == 1024

    def test_from_key_with_timestamp(self) -> None:
        """Test parsing with creation timestamp."""
        key = "EUR_USD/indicators/macd/f1e2d3c4:15m:fedcba09"
        now = datetime.now(tz=timezone.utc)
        entry = CacheEntry.from_key(key, size_bytes=2048, created_at=now)

        assert entry is not None
        assert entry.created_at == now

    def test_from_key_invalid_format(self) -> None:
        """Test invalid key format returns None."""
        assert CacheEntry.from_key("invalid") is None
        assert CacheEntry.from_key("no/indicators/here") is None
        assert CacheEntry.from_key("sym/indicators/ind/no_colons") is None

    def test_from_key_missing_symbol(self) -> None:
        """Test key without symbol returns None."""
        assert CacheEntry.from_key("indicators/rsi/a1b2:1h:abcd") is None

    def test_from_key_wrong_colon_count(self) -> None:
        """Test params key with wrong number of colons returns None."""
        assert CacheEntry.from_key("SYM/indicators/rsi/a1b2:1h") is None
        assert CacheEntry.from_key("SYM/indicators/rsi/a:b:c:d") is None


class TestCacheStats:
    """Tests for CacheStats model."""

    def test_empty_stats(self) -> None:
        """Test default empty stats."""
        stats = CacheStats()
        assert stats.total_entries == 0
        assert stats.total_size_bytes == 0
        assert stats.by_symbol == {}
        assert stats.by_indicator == {}

    def test_add_entry(self) -> None:
        """Test adding entries updates stats."""
        stats = CacheStats()
        entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
            size_bytes=1024,
        )
        stats.add_entry(entry)

        assert stats.total_entries == 1
        assert stats.total_size_bytes == 1024
        assert stats.by_symbol == {"BTC_USDT": 1}
        assert stats.by_indicator == {"rsi": 1}
        assert stats.by_timeframe == {"1h": 1}

    def test_add_multiple_entries(self) -> None:
        """Test adding multiple entries accumulates correctly."""
        stats = CacheStats()

        entry1 = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
            size_bytes=1024,
        )
        entry2 = CacheEntry(
            key="BTC_USDT/indicators/macd/c:1h:d",
            symbol="BTC_USDT",
            indicator="macd",
            params_hash="c",
            timeframe="1h",
            data_hash="d",
            size_bytes=2048,
        )
        entry3 = CacheEntry(
            key="EUR_USD/indicators/rsi/e:1m:f",
            symbol="EUR_USD",
            indicator="rsi",
            params_hash="e",
            timeframe="1m",
            data_hash="f",
            size_bytes=4096,
        )

        stats.add_entry(entry1)
        stats.add_entry(entry2)
        stats.add_entry(entry3)

        assert stats.total_entries == 3
        assert stats.total_size_bytes == 7168
        assert stats.by_symbol == {"BTC_USDT": 2, "EUR_USD": 1}
        assert stats.by_indicator == {"rsi": 2, "macd": 1}
        assert stats.by_timeframe == {"1h": 2, "1m": 1}

    def test_size_properties(self) -> None:
        """Test size conversion properties."""
        stats = CacheStats(total_size_bytes=1073741824)  # 1 GB
        assert stats.total_size_mb == 1024.0
        assert stats.total_size_gb == 1.0


class TestCleanupCriteria:
    """Tests for CleanupCriteria model."""

    def test_empty_criteria(self) -> None:
        """Test empty criteria matches all."""
        criteria = CleanupCriteria()
        assert criteria.is_empty

        entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        assert criteria.matches(entry)

    def test_symbol_filter(self) -> None:
        """Test filtering by symbol."""
        criteria = CleanupCriteria(symbol="BTC_USDT")
        assert not criteria.is_empty

        btc_entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        eur_entry = CacheEntry(
            key="EUR_USD/indicators/rsi/a:1h:b",
            symbol="EUR_USD",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )

        assert criteria.matches(btc_entry)
        assert not criteria.matches(eur_entry)

    def test_indicator_filter_exact(self) -> None:
        """Test filtering by exact indicator name."""
        criteria = CleanupCriteria(indicator="rsi")

        rsi_entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        macd_entry = CacheEntry(
            key="BTC_USDT/indicators/macd/a:1h:b",
            symbol="BTC_USDT",
            indicator="macd",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )

        assert criteria.matches(rsi_entry)
        assert not criteria.matches(macd_entry)

    def test_indicator_filter_glob(self) -> None:
        """Test filtering by indicator glob pattern."""
        criteria = CleanupCriteria(indicator="sar*")

        sar_entry = CacheEntry(
            key="BTC_USDT/indicators/sar/a:1h:b",
            symbol="BTC_USDT",
            indicator="sar",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        sarext_entry = CacheEntry(
            key="BTC_USDT/indicators/sarext/a:1h:b",
            symbol="BTC_USDT",
            indicator="sarext",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        rsi_entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )

        assert criteria.matches(sar_entry)
        assert criteria.matches(sarext_entry)
        assert not criteria.matches(rsi_entry)

    def test_timeframe_filter(self) -> None:
        """Test filtering by timeframe."""
        criteria = CleanupCriteria(timeframe="1m")

        entry_1m = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1m:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1m",
            data_hash="b",
        )
        entry_1h = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )

        assert criteria.matches(entry_1m)
        assert not criteria.matches(entry_1h)

    def test_older_than_filter(self) -> None:
        """Test filtering by age."""
        criteria = CleanupCriteria(older_than="7d")
        now = datetime.now(tz=timezone.utc)

        old_entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
            created_at=now - timedelta(days=10),
        )
        new_entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:c",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="c",
            created_at=now - timedelta(days=1),
        )
        no_timestamp_entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:d",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="d",
        )

        assert criteria.matches(old_entry, now=now)
        assert not criteria.matches(new_entry, now=now)
        assert not criteria.matches(no_timestamp_entry, now=now)  # No timestamp = don't match

    def test_compound_criteria(self) -> None:
        """Test combining multiple criteria."""
        criteria = CleanupCriteria(symbol="BTC_USDT", indicator="rsi", timeframe="1h")

        matching = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        wrong_symbol = CacheEntry(
            key="EUR_USD/indicators/rsi/a:1h:b",
            symbol="EUR_USD",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        wrong_indicator = CacheEntry(
            key="BTC_USDT/indicators/macd/a:1h:b",
            symbol="BTC_USDT",
            indicator="macd",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        wrong_timeframe = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1m:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1m",
            data_hash="b",
        )

        assert criteria.matches(matching)
        assert not criteria.matches(wrong_symbol)
        assert not criteria.matches(wrong_indicator)
        assert not criteria.matches(wrong_timeframe)

    def test_invalid_older_than_validation(self) -> None:
        """Test that invalid older_than raises validation error."""
        with pytest.raises(ValueError):
            CleanupCriteria(older_than="invalid")


class TestCleanupResult:
    """Tests for CleanupResult model."""

    def test_empty_result(self) -> None:
        """Test default empty result."""
        result = CleanupResult()
        assert result.deleted_count == 0
        assert result.freed_bytes == 0
        assert result.errors == []
        assert result.success

    def test_add_deletion(self) -> None:
        """Test recording deletions."""
        result = CleanupResult()
        entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
            size_bytes=1024,
        )
        result.add_deletion(entry)

        assert result.deleted_count == 1
        assert result.freed_bytes == 1024

    def test_add_error(self) -> None:
        """Test recording errors."""
        result = CleanupResult()
        result.add_error("Failed to delete entry")

        assert not result.success
        assert len(result.errors) == 1
        assert "Failed to delete" in result.errors[0]

    def test_freed_properties(self) -> None:
        """Test freed space conversion properties."""
        result = CleanupResult(freed_bytes=1073741824)  # 1 GB
        assert result.freed_mb == 1024.0
        assert result.freed_gb == 1.0

    def test_dry_run_flag(self) -> None:
        """Test dry run flag."""
        result = CleanupResult(dry_run=True)
        assert result.dry_run


class TestCacheFilter:
    """Tests for CacheFilter model."""

    def test_empty_filter(self) -> None:
        """Test empty filter matches all."""
        filter = CacheFilter()
        entry = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        assert filter.matches(entry)

    def test_symbol_filter(self) -> None:
        """Test filtering by symbol."""
        filter = CacheFilter(symbol="BTC_USDT")
        btc = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        eur = CacheEntry(
            key="EUR_USD/indicators/rsi/a:1h:b",
            symbol="EUR_USD",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        assert filter.matches(btc)
        assert not filter.matches(eur)

    def test_indicator_glob(self) -> None:
        """Test indicator glob pattern."""
        filter = CacheFilter(indicator="stoch*")
        stoch = CacheEntry(
            key="BTC_USDT/indicators/stoch/a:1h:b",
            symbol="BTC_USDT",
            indicator="stoch",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        stochf = CacheEntry(
            key="BTC_USDT/indicators/stochf/a:1h:b",
            symbol="BTC_USDT",
            indicator="stochf",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        rsi = CacheEntry(
            key="BTC_USDT/indicators/rsi/a:1h:b",
            symbol="BTC_USDT",
            indicator="rsi",
            params_hash="a",
            timeframe="1h",
            data_hash="b",
        )
        assert filter.matches(stoch)
        assert filter.matches(stochf)
        assert not filter.matches(rsi)

    def test_limit_offset(self) -> None:
        """Test limit and offset are stored correctly."""
        filter = CacheFilter(limit=10, offset=5)
        assert filter.limit == 10
        assert filter.offset == 5
