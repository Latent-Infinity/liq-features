"""Indicator result caching using xxhash for fast key generation.

This module provides caching for computed indicator results to avoid
redundant calculations. Cache keys incorporate data hash, indicator name,
and parameters to ensure correctness.

Design Principles:
    - SRP: Focused solely on caching indicator results
    - DIP: Uses liq-store storage abstraction
    - KISS: Simple key-value storage with Parquet files

Note:
    This module requires xxhash for fast hashing (optional dependency).
    Install with: pip install xxhash

Example:
    >>> from liq.features.cache import IndicatorCache, compute_cache_key, get_data_hash
    >>> cache = IndicatorCache()
    >>>
    >>> # Compute cache key
    >>> data_hash = get_data_hash(df)
    >>> key = compute_cache_key("EUR_USD", "1m", "RSI", {"period": 14}, data_hash)
    >>>
    >>> # Check cache
    >>> if cache.has(key):
    ...     result = cache.get(key)
    ... else:
    ...     result = indicator.compute(df)
    ...     cache.set(key, result)
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from multiprocessing import current_process
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import polars as pl
from liq.store import key_builder
from liq.store.config import create_parquet_store_from_env
from liq.store.parquet import ParquetStore
from liq.store.protocols import TimeSeriesStore

from liq.features.params import hash_params

if TYPE_CHECKING:
    from liq.features.cache_models import (
        CacheEntry,
        CacheFilter,
        CacheStats,
        CleanupCriteria,
        CleanupResult,
    )

logger = logging.getLogger(__name__)
try:
    import threading

    _INDEX_LOCK = threading.Lock()
except Exception:
    _INDEX_LOCK = None

# Try to import xxhash for faster hashing
try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    xxhash = None  # type: ignore[assignment]


@runtime_checkable
class CacheManager(Protocol):
    """Protocol for cache management operations.

    Extends basic cache operations with management capabilities:
    - Statistics collection
    - Cleanup with criteria
    - Entry listing with filters

    This protocol allows different cache implementations while
    maintaining a consistent management interface.
    """

    def stats(self) -> CacheStats:
        """Collect cache statistics.

        Returns:
            CacheStats with entry counts, sizes, and breakdowns.
        """
        ...

    def clean(
        self,
        criteria: CleanupCriteria,
        *,
        dry_run: bool = False,
    ) -> CleanupResult:
        """Clean cache entries matching criteria.

        Args:
            criteria: Criteria for selecting entries to delete.
            dry_run: If True, report what would be deleted without deleting.

        Returns:
            CleanupResult with deletion counts and any errors.
        """
        ...

    def list_entries(
        self,
        filters: CacheFilter | None = None,
    ) -> list[CacheEntry]:
        """List cache entries matching filters.

        Args:
            filters: Optional filters to apply.

        Returns:
            List of matching cache entries.
        """
        ...


def get_data_hash(df: pl.DataFrame) -> str:
    """Compute a hash of DataFrame contents for cache invalidation.

    Uses xxhash if available for performance, falls back to SHA256.

    Args:
        df: DataFrame to hash

    Returns:
        Hex string hash of the data
    """
    # Serialize DataFrame to bytes
    data_bytes = df.write_ipc(None).getvalue()

    if HAS_XXHASH:
        return xxhash.xxh64(data_bytes).hexdigest()
    else:
        return hashlib.sha256(data_bytes).hexdigest()[:16]


def compute_cache_key(
    symbol: str,
    timeframe: str,
    indicator: str,
    params: dict[str, Any],
    data_hash: str,
) -> str:
    """Compute a unique cache key for an indicator result.

    The key incorporates all factors that affect the result:
    - Symbol and timeframe identify the data source
    - Indicator name identifies the computation
    - Parameters affect the computation
    - Data hash ensures invalidation when data changes

    Args:
        symbol: Trading symbol (e.g., "EUR_USD")
        timeframe: Data timeframe (e.g., "1m", "1h")
        indicator: Indicator name (e.g., "RSI")
        params: Indicator parameters (e.g., {"period": 14})
        data_hash: Hash of the input data

    Returns:
        Unique cache key string
    """
    params_id = hash_params(params)
    params_key = f"{params_id}:{timeframe}:{data_hash}"
    return key_builder.indicators(symbol, indicator.lower(), params_key)


class IndicatorCache:
    """Store-backed cache for indicator computation results."""

    def __init__(
        self,
        storage: TimeSeriesStore | None = None,
        *,
        data_root: Path | str | None = None,
    ) -> None:
        """Initialize the indicator cache.

        Args:
            storage: Optional TimeSeriesStore to use for caching.
            data_root: Optional data root for ParquetStore when storage is not provided.
        """
        if storage is None:
            if data_root is not None:
                storage = ParquetStore(str(Path(data_root).expanduser()))
            else:
                storage = create_parquet_store_from_env()
        self._storage = storage
        self._index_key = "cache/indicators/index"
        self._index_enabled = os.getenv("LIQ_FEATURES_INDEX", "off").lower() != "off"
        self._lockless_prev: bool | None = None
        self._lockless_rebuild: bool = True

    def _load_index(self) -> pl.DataFrame:
        if not self._index_enabled:
            return pl.DataFrame(
                schema={
                    "key": pl.Utf8,
                    "symbol": pl.Utf8,
                    "timeframe": pl.Utf8,
                    "indicator": pl.Utf8,
                    "params_hash": pl.Utf8,
                    "data_hash": pl.Utf8,
                }
            )
        if self._storage.exists(self._index_key):
            return self._storage.read(self._index_key)
        index_df = pl.DataFrame(
            schema={
                "key": pl.Utf8,
                "symbol": pl.Utf8,
                "timeframe": pl.Utf8,
                "indicator": pl.Utf8,
                "params_hash": pl.Utf8,
                "data_hash": pl.Utf8,
            }
        )
        return self._rebuild_index(index_df)

    def _rebuild_index(self, empty_index: pl.DataFrame) -> pl.DataFrame:
        rows: list[dict[str, str]] = []
        for key in self._storage.list_keys():
            meta = _meta_from_key(key)
            if meta is not None:
                rows.append(meta)
        if not rows:
            return empty_index
        index_df = pl.DataFrame(rows)
        self._write_index(index_df)
        return index_df

    def rebuild_index(self) -> pl.DataFrame:
        """Rebuild the cache index from storage keys."""
        prev_enabled = self._index_enabled
        self._index_enabled = True
        empty_index = pl.DataFrame(
            schema={
                "key": pl.Utf8,
                "symbol": pl.Utf8,
                "timeframe": pl.Utf8,
                "indicator": pl.Utf8,
                "params_hash": pl.Utf8,
                "data_hash": pl.Utf8,
            }
        )
        rebuilt = self._rebuild_index(empty_index)
        self._index_enabled = prev_enabled
        return rebuilt

    def _write_index(self, index_df: pl.DataFrame) -> None:
        if not self._index_enabled:
            return
        self._storage.write(self._index_key, index_df, mode="overwrite")

    @property
    def storage(self) -> TimeSeriesStore:
        """Get the underlying storage instance."""
        return self._storage

    @property
    def storage_root(self) -> Path | None:
        """Best-effort access to the storage root when available."""
        root = getattr(self._storage, "data_root", None)
        if isinstance(root, Path):
            return root
        if isinstance(root, str):
            return Path(root)
        return None

    def has(self, key: str) -> bool:
        """Check if a cache entry exists.

        Args:
            key: Cache key to check

        Returns:
            True if entry exists, False otherwise
        """
        return self._storage.exists(key)

    def get(self, key: str) -> pl.DataFrame | None:
        """Retrieve a cached result.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached DataFrame or None if not found
        """
        if not self._storage.exists(key):
            return None
        df = self._storage.read(key)
        if df.is_empty():
            return None
        return df

    def set(
        self,
        key: str,
        df: pl.DataFrame,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        indicator: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Store a result in the cache.

        Args:
            key: Cache key
            df: DataFrame to cache
            symbol: Optional symbol for index metadata
            timeframe: Optional timeframe for index metadata
            indicator: Optional indicator name for index metadata
            params: Optional params dict for index metadata
        """
        self._storage.write(key, df, mode="overwrite")

        if not self._index_enabled or current_process().name != "MainProcess":
            return

        meta = _meta_from_key(key)

        if meta is not None:
            try:
                if _INDEX_LOCK:
                    _INDEX_LOCK.acquire()
                index_df = self._load_index()
                index_df = pl.concat([index_df, pl.DataFrame([meta])], how="vertical")
                index_df = index_df.unique(subset=["key"], keep="last")
                self._write_index(index_df)
            except Exception as exc:
                logger.warning("Failed to update indicator cache index: %s", exc)
            finally:
                if _INDEX_LOCK:
                    _INDEX_LOCK.release()

    def delete(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: Cache key to delete
        """
        self._storage.delete(key)
        if self._index_enabled and self._storage.exists(self._index_key):
            index_df = self._load_index()
            if "key" in index_df.columns:
                index_df = index_df.filter(pl.col("key") != key)
                self._write_index(index_df)

    def delete_by_indicator(self, pattern: str) -> int:
        """Delete all cache entries matching an indicator pattern.

        Args:
            pattern: Indicator name pattern (case-insensitive, supports wildcards).
                     Examples: "sar", "sarext", "sar*" (matches sar and sarext)

        Returns:
            Number of entries deleted
        """
        import fnmatch

        pattern_lower = pattern.lower()
        if not self._index_enabled:
            return 0
        index_df = self._load_index()
        if index_df.is_empty():
            return 0
        indicators = index_df["indicator"].to_list()
        matches = [
            fnmatch.fnmatch(ind.lower(), pattern_lower)
            for ind in indicators
        ]
        keys = index_df.filter(pl.Series(matches))["key"].to_list()
        for key in keys:
            self._storage.delete(key)
        index_df = index_df.filter(~index_df["key"].is_in(keys))
        self._write_index(index_df)
        return len(keys)

    def list_indicators(self) -> dict[str, int]:
        """List all cached indicators with counts.

        Returns:
            Dict mapping indicator names to entry counts
        """
        from collections import Counter

        if not self._index_enabled:
            return {}
        index_df = self._load_index()
        if index_df.is_empty():
            return {}
        indicators = Counter(index_df["indicator"].to_list())
        return dict(indicators)

    def clear(self) -> None:
        """Clear all cache entries."""
        index_df = self._load_index() if self._index_enabled else pl.DataFrame()
        keys = index_df["key"].to_list() if not index_df.is_empty() else [
            k for k in self._storage.list_keys() if "/indicators/" in k
        ]
        for key in keys:
            self._storage.delete(key)
        if self._index_enabled and self._storage.exists(self._index_key):
            self._write_index(index_df.head(0))

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats:
                - entries: Number of cached entries
                - total_size_bytes: Total size of cache in bytes
        """
        index_df = self._load_index() if self._index_enabled else pl.DataFrame()
        keys = index_df["key"].to_list() if not index_df.is_empty() else [
            k for k in self._storage.list_keys() if "/indicators/" in k
        ]
        total_size = 0
        for key in keys:
            try:
                df = self._storage.read(key)
                total_size += int(df.estimated_size())
            except Exception:
                continue

        return {
            "entries": len(keys),
            "total_size_bytes": total_size,
        }

    def query_index(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        indicator: str | None = None,
        params_hash: str | None = None,
        data_hash: str | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Query the cache index with optional filters."""
        if not self._index_enabled:
            return pl.DataFrame()
        index_df = self._load_index()
        if index_df.is_empty():
            return index_df

        df = index_df
        if symbol:
            df = df.filter(pl.col("symbol") == symbol)
        if timeframe:
            df = df.filter(pl.col("timeframe") == timeframe)
        if params_hash:
            df = df.filter(pl.col("params_hash") == params_hash)
        if data_hash:
            df = df.filter(pl.col("data_hash") == data_hash)
        if indicator:
            matches = _match_pattern(df["indicator"].to_list(), indicator)
            df = df.filter(pl.Series(matches))
        if limit:
            df = df.head(limit)
        return df

    def _get_entry_created_at(self, key: str) -> datetime | None:
        """Get the creation timestamp for a cache entry.

        Uses file modification time as a proxy for creation time.

        Args:
            key: Cache key.

        Returns:
            Creation timestamp, or None if unavailable.
        """
        from datetime import datetime, timezone

        try:
            # Get the storage root and construct the path
            root = self.storage_root
            if root is None:
                return None

            key_path = root / key
            if not key_path.exists():
                return None

            # Find the oldest parquet file in the key directory
            parquet_files = list(key_path.rglob("*.parquet"))
            if not parquet_files:
                # Try direct file
                if key_path.is_file():
                    mtime = key_path.stat().st_mtime
                    return datetime.fromtimestamp(mtime, tz=timezone.utc)
                return None

            # Use the oldest file's mtime as creation time
            oldest_mtime = min(f.stat().st_mtime for f in parquet_files)
            return datetime.fromtimestamp(oldest_mtime, tz=timezone.utc)
        except Exception:
            return None

    def list_entries(
        self,
        filters: CacheFilter | None = None,
    ) -> list[CacheEntry]:
        """List cache entries matching filters.

        Args:
            filters: Optional filters to apply.

        Returns:
            List of matching cache entries.
        """
        from liq.features.cache_models import CacheEntry, CacheFilter

        # Get all keys
        keys = self._storage.list_keys()
        indicator_keys = [k for k in keys if "/indicators/" in k]

        entries: list[CacheEntry] = []
        for key in indicator_keys:
            entry = CacheEntry.from_key(key)
            if entry is None:
                continue

            # Get size and created_at
            try:
                df = self._storage.read(key)
                entry.size_bytes = int(df.estimated_size())
            except Exception:
                entry.size_bytes = 0

            entry.created_at = self._get_entry_created_at(key)

            # Apply filters
            if filters is not None and not filters.matches(entry):
                continue

            entries.append(entry)

        # Apply limit and offset from filter
        if filters is not None:
            if filters.offset > 0:
                entries = entries[filters.offset:]
            if filters.limit is not None:
                entries = entries[:filters.limit]

        return entries

    def clean(
        self,
        criteria: CleanupCriteria,
        *,
        dry_run: bool = False,
    ) -> CleanupResult:
        """Clean cache entries matching criteria.

        Args:
            criteria: Criteria for selecting entries to delete.
            dry_run: If True, report what would be deleted without deleting.

        Returns:
            CleanupResult with deletion counts and any errors.
        """
        from liq.features.cache_models import CacheEntry, CleanupCriteria, CleanupResult

        result = CleanupResult(dry_run=dry_run)

        # Get all keys
        keys = self._storage.list_keys()
        indicator_keys = [k for k in keys if "/indicators/" in k]

        for key in indicator_keys:
            entry = CacheEntry.from_key(key)
            if entry is None:
                continue

            # Get size and created_at for accurate reporting and age filtering
            try:
                df = self._storage.read(key)
                entry.size_bytes = int(df.estimated_size())
            except Exception:
                entry.size_bytes = 0

            entry.created_at = self._get_entry_created_at(key)

            # Check if entry matches criteria
            if not criteria.is_empty and not criteria.matches(entry):
                continue

            # Delete or just count
            if not dry_run:
                try:
                    self._storage.delete(key)
                    result.add_deletion(entry)
                except Exception as exc:
                    result.add_error(f"Failed to delete {key}: {exc}")
            else:
                result.add_deletion(entry)

        # Update index if we deleted anything
        if not dry_run and result.deleted_count > 0 and self._index_enabled:
            self.rebuild_index()

        return result

    def lockless_index(self, rebuild_on_exit: bool = True) -> "IndicatorCache":
        """Return self configured for lockless indexing in a with block."""
        self._lockless_prev = self._index_enabled
        self._lockless_rebuild = rebuild_on_exit
        self._index_enabled = False
        return self

    def __enter__(self) -> "IndicatorCache":
        return self.lockless_index(rebuild_on_exit=True)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        prev = self._lockless_prev if self._lockless_prev is not None else True
        rebuild = self._lockless_rebuild
        self._lockless_prev = None
        self._lockless_rebuild = True
        self._index_enabled = prev
        if rebuild:
            self.rebuild_index()


def _indicator_from_key(key: str) -> str | None:
    """Extract indicator name from storage key."""
    parts = key.split("/")
    for idx, part in enumerate(parts):
        if part == "indicators" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _meta_from_key(key: str) -> dict[str, str] | None:
    """Parse a cache key into metadata for indexing."""
    parts = key.split("/")
    if len(parts) < 4:
        return None
    if "indicators" not in parts:
        return None
    idx = parts.index("indicators")
    if idx + 2 >= len(parts):
        return None
    symbol = parts[idx - 1] if idx > 0 else None
    indicator = parts[idx + 1]
    params_key = parts[idx + 2]
    if symbol is None:
        return None
    if params_key.count(":") != 2:
        return None
    params_hash, timeframe, data_hash = params_key.split(":")
    return {
        "key": key,
        "symbol": symbol,
        "timeframe": timeframe,
        "indicator": indicator,
        "params_hash": params_hash,
        "data_hash": data_hash,
    }


def _match_pattern(values: list[str], pattern: str) -> list[bool]:
    import fnmatch

    pattern_lower = pattern.lower()
    return [fnmatch.fnmatch(value.lower(), pattern_lower) for value in values]
