"""Tests for cache CLI commands."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from typer.testing import CliRunner

from liq.features.cache import IndicatorCache, compute_cache_key
from liq.features.cli import app
from liq.store.parquet import ParquetStore

runner = CliRunner()


@pytest.fixture
def sample_result_df() -> pl.DataFrame:
    """Sample indicator result dataframe."""
    return pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 15, 10, tzinfo=UTC),
                datetime(2024, 1, 15, 11, tzinfo=UTC),
                datetime(2024, 1, 15, 12, tzinfo=UTC),
            ],
            "value": [50.0, 55.0, 60.0],
        }
    )


class TestCacheStatsCommand:
    """Tests for 'cache stats' command."""

    def test_stats_empty_cache(self, tmp_path: Path) -> None:
        """Test stats with empty cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = runner.invoke(app, ["cache", "stats", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_stats_with_entries(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test stats shows entry count."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        for i in range(3):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{i}",
            )
            cache.set(key, sample_result_df)

        result = runner.invoke(app, ["cache", "stats", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "3" in result.output


class TestCacheListCommand:
    """Tests for 'cache list' command."""

    def test_list_empty_cache(self, tmp_path: Path) -> None:
        """Test list with empty cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = runner.invoke(app, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "No cache entries" in result.output

    def test_list_with_entries(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test list shows entries."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        result = runner.invoke(app, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "BTC_USDT" in result.output
        assert "rsi" in result.output

    def test_list_filter_by_symbol(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test list filters by symbol."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        for symbol in ["BTC_USDT", "EUR_USD"]:
            key = compute_cache_key(
                symbol=symbol,
                timeframe="1h",
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{symbol}",
            )
            cache.set(key, sample_result_df)

        result = runner.invoke(
            app, ["cache", "list", "--cache-dir", str(cache_dir), "--symbol", "BTC_USDT"]
        )
        assert result.exit_code == 0
        assert "BTC_USDT" in result.output
        assert "EUR_USD" not in result.output


class TestCacheCleanCommand:
    """Tests for 'cache clean' command."""

    def test_clean_requires_filter_or_all(self, tmp_path: Path) -> None:
        """Test clean requires --all or filter."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = runner.invoke(app, ["cache", "clean", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_clean_dry_run(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test clean --dry-run doesn't delete."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        result = runner.invoke(
            app, ["cache", "clean", "--cache-dir", str(cache_dir), "--all", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "1" in result.output
        # Entry should still exist
        assert cache.has(key)

    def test_clean_with_force(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test clean --force skips confirmation."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        result = runner.invoke(
            app, ["cache", "clean", "--cache-dir", str(cache_dir), "--all", "--force"]
        )
        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Entry should be gone
        assert not cache.has(key)

    def test_clean_by_symbol(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test clean --symbol filter."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        for symbol in ["BTC_USDT", "EUR_USD"]:
            key = compute_cache_key(
                symbol=symbol,
                timeframe="1h",
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{symbol}",
            )
            cache.set(key, sample_result_df)

        result = runner.invoke(
            app, ["cache", "clean", "--cache-dir", str(cache_dir), "--symbol", "BTC_USDT", "--force"]
        )
        assert result.exit_code == 0
        assert "1" in result.output  # Deleted 1 entry
        # EUR_USD should still exist
        eur_key = compute_cache_key(
            symbol="EUR_USD",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_EUR_USD",
        )
        assert cache.has(eur_key)

    def test_clean_by_indicator(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test clean --indicator filter."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        for indicator in ["sarext", "rsi"]:
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator=indicator,
                params={"period": 14},
                data_hash=f"hash_{indicator}",
            )
            cache.set(key, sample_result_df)

        result = runner.invoke(
            app, ["cache", "clean", "--cache-dir", str(cache_dir), "--indicator", "sarext", "--force"]
        )
        assert result.exit_code == 0
        assert "1" in result.output

    def test_clean_no_matches(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test clean with no matches."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        result = runner.invoke(
            app, ["cache", "clean", "--cache-dir", str(cache_dir), "--symbol", "NONEXISTENT"]
        )
        assert result.exit_code == 0
        assert "No entries match" in result.output

    def test_clean_confirmation_cancelled(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test clean cancelled by user."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)

        # Simulate user typing "n" to cancel
        result = runner.invoke(
            app, ["cache", "clean", "--cache-dir", str(cache_dir), "--all"],
            input="n\n"
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        # Entry should still exist
        assert cache.has(key)


class TestLegacyIndicatorCacheCommand:
    """Tests for legacy 'indicator-cache' command."""

    def test_cache_status_empty(self, tmp_path: Path) -> None:
        """Test cache status with empty cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = runner.invoke(app, ["indicator-cache", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "0" in result.output or "entries" in result.output.lower()

    def test_cache_status_with_entries(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test cache status shows entry count."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        for i in range(2):
            key = compute_cache_key(
                symbol="BTC_USDT",
                timeframe="1h",
                indicator="rsi",
                params={"period": 14},
                data_hash=f"hash_{i}",
            )
            cache.set(key, sample_result_df)

        result = runner.invoke(app, ["indicator-cache", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "2" in result.output

    def test_cache_clear(self, tmp_path: Path, sample_result_df: pl.DataFrame) -> None:
        """Test cache clear command."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(storage=ParquetStore(str(cache_dir)))

        key = compute_cache_key(
            symbol="BTC_USDT",
            timeframe="1h",
            indicator="rsi",
            params={"period": 14},
            data_hash="hash_abc",
        )
        cache.set(key, sample_result_df)
        assert cache.has(key)

        result = runner.invoke(app, ["indicator-cache", "--cache-dir", str(cache_dir), "--clear"])
        assert result.exit_code == 0
        assert not cache.has(key)
