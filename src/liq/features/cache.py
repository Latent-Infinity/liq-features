"""Indicator result caching using xxhash for fast key generation.

This module provides caching for computed indicator results to avoid
redundant calculations. Cache keys incorporate data hash, indicator name,
and parameters to ensure correctness.

Design Principles:
    - SRP: Focused solely on caching indicator results
    - DIP: Uses file-based storage abstraction
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
import json
from pathlib import Path
from typing import Any

import polars as pl

# Try to import xxhash for faster hashing
try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    xxhash = None  # type: ignore[assignment]


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
    # Normalize params to sorted JSON for deterministic hashing
    params_str = json.dumps(params, sort_keys=True)

    # Combine all components
    key_parts = [symbol, timeframe, indicator, params_str, data_hash]
    key_string = "|".join(key_parts)

    # Hash the combined key for shorter, uniform keys
    if HAS_XXHASH:
        return xxhash.xxh64(key_string.encode()).hexdigest()
    else:
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]


class IndicatorCache:
    """File-based cache for indicator computation results.

    Stores computed indicator results as Parquet files for fast retrieval.
    Uses xxhash-based cache keys for efficient lookup.

    Attributes:
        cache_dir: Directory where cache files are stored
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize the indicator cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.liq/cache/indicators
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".liq" / "cache" / "indicators"
        elif isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self._cache_dir

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        return self._cache_dir / f"{key}.parquet"

    def has(self, key: str) -> bool:
        """Check if a cache entry exists.

        Args:
            key: Cache key to check

        Returns:
            True if entry exists, False otherwise
        """
        return self._key_to_path(key).exists()

    def get(self, key: str) -> pl.DataFrame | None:
        """Retrieve a cached result.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached DataFrame or None if not found
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            return pl.read_parquet(path)
        except Exception:
            # Corrupted cache entry
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, df: pl.DataFrame) -> None:
        """Store a result in the cache.

        Args:
            key: Cache key
            df: DataFrame to cache
        """
        path = self._key_to_path(key)
        df.write_parquet(path)

    def delete(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: Cache key to delete
        """
        path = self._key_to_path(key)
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all cache entries."""
        for path in self._cache_dir.glob("*.parquet"):
            path.unlink(missing_ok=True)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats:
                - entries: Number of cached entries
                - total_size_bytes: Total size of cache in bytes
        """
        entries = list(self._cache_dir.glob("*.parquet"))
        total_size = sum(p.stat().st_size for p in entries)

        return {
            "entries": len(entries),
            "total_size_bytes": total_size,
        }
