"""Data models for indicator cache management.

Provides Pydantic models for cache statistics, cleanup criteria, and results.
These models enable type-safe cache management operations with validation.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CacheEntry(BaseModel):
    """Represents a single cache entry.

    Attributes:
        key: Full cache key path.
        symbol: Trading symbol (e.g., BTC_USDT).
        indicator: Indicator name (e.g., rsi, macd).
        params_hash: Hash of indicator parameters.
        timeframe: Data timeframe (e.g., 1m, 1h).
        data_hash: Hash of input data.
        size_bytes: Size of cached data in bytes.
        created_at: When the entry was created (if known).
    """

    key: str
    symbol: str
    indicator: str
    params_hash: str
    timeframe: str
    data_hash: str
    size_bytes: int = 0
    created_at: datetime | None = None

    @classmethod
    def from_key(cls, key: str, size_bytes: int = 0, created_at: datetime | None = None) -> CacheEntry | None:
        """Parse a cache key into a CacheEntry.

        Args:
            key: Full cache key path.
            size_bytes: Size of cached data.
            created_at: Creation timestamp.

        Returns:
            CacheEntry if key is valid, None otherwise.
        """
        parts = key.split("/")
        if len(parts) < 4 or "indicators" not in parts:
            return None

        idx = parts.index("indicators")
        if idx + 2 >= len(parts) or idx < 1:
            return None

        symbol = parts[idx - 1]
        indicator = parts[idx + 1]
        params_key = parts[idx + 2]

        if params_key.count(":") != 2:
            return None

        params_hash, timeframe, data_hash = params_key.split(":")

        return cls(
            key=key,
            symbol=symbol,
            indicator=indicator,
            params_hash=params_hash,
            timeframe=timeframe,
            data_hash=data_hash,
            size_bytes=size_bytes,
            created_at=created_at,
        )


class CacheStats(BaseModel):
    """Statistics about cache usage.

    Attributes:
        total_entries: Total number of cache entries.
        total_size_bytes: Total size of all entries in bytes.
        by_symbol: Entry count per symbol.
        by_indicator: Entry count per indicator.
        by_timeframe: Entry count per timeframe.
        size_by_symbol: Size in bytes per symbol.
        size_by_indicator: Size in bytes per indicator.
    """

    total_entries: int = 0
    total_size_bytes: int = 0
    by_symbol: dict[str, int] = Field(default_factory=dict)
    by_indicator: dict[str, int] = Field(default_factory=dict)
    by_timeframe: dict[str, int] = Field(default_factory=dict)
    size_by_symbol: dict[str, int] = Field(default_factory=dict)
    size_by_indicator: dict[str, int] = Field(default_factory=dict)

    @property
    def total_size_mb(self) -> float:
        """Total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def total_size_gb(self) -> float:
        """Total size in gigabytes."""
        return self.total_size_bytes / (1024 * 1024 * 1024)

    def add_entry(self, entry: CacheEntry) -> None:
        """Add an entry to the statistics.

        Args:
            entry: Cache entry to add.
        """
        self.total_entries += 1
        self.total_size_bytes += entry.size_bytes

        self.by_symbol[entry.symbol] = self.by_symbol.get(entry.symbol, 0) + 1
        self.by_indicator[entry.indicator] = self.by_indicator.get(entry.indicator, 0) + 1
        self.by_timeframe[entry.timeframe] = self.by_timeframe.get(entry.timeframe, 0) + 1

        self.size_by_symbol[entry.symbol] = self.size_by_symbol.get(entry.symbol, 0) + entry.size_bytes
        self.size_by_indicator[entry.indicator] = (
            self.size_by_indicator.get(entry.indicator, 0) + entry.size_bytes
        )


def parse_duration(duration_str: str) -> timedelta:
    """Parse a duration string into a timedelta.

    Supports formats: 7d, 24h, 30m, 1w, 2w

    Args:
        duration_str: Duration string.

    Returns:
        Parsed timedelta.

    Raises:
        ValueError: If format is invalid.
    """
    pattern = r"^(\d+)(d|h|m|w)$"
    match = re.match(pattern, duration_str.strip().lower())
    if not match:
        raise ValueError(
            f"Invalid duration format: {duration_str}. "
            "Expected format: <number><unit> where unit is d(days), h(hours), m(minutes), w(weeks)"
        )

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "d":
        return timedelta(days=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "w":
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Unknown duration unit: {unit}")


class CleanupCriteria(BaseModel):
    """Criteria for selecting cache entries to clean.

    All criteria are combined with AND logic. Empty criteria matches all entries.

    Attributes:
        older_than: Delete entries older than this duration (e.g., "7d", "24h").
        symbol: Filter by symbol (exact match).
        indicator: Filter by indicator (supports glob patterns like "sar*").
        timeframe: Filter by timeframe (exact match).
        data_hash: Filter by data hash (exact match).
    """

    older_than: str | None = None
    symbol: str | None = None
    indicator: str | None = None
    timeframe: str | None = None
    data_hash: str | None = None

    @field_validator("older_than")
    @classmethod
    def validate_older_than(cls, v: str | None) -> str | None:
        """Validate older_than duration format."""
        if v is not None:
            parse_duration(v)  # Raises ValueError if invalid
        return v

    def matches(self, entry: CacheEntry, now: datetime | None = None) -> bool:
        """Check if an entry matches this criteria.

        Args:
            entry: Cache entry to check.
            now: Current time for age comparison.

        Returns:
            True if entry matches all criteria.
        """
        import fnmatch

        if self.symbol is not None and entry.symbol != self.symbol:
            return False

        if self.indicator is not None:
            if not fnmatch.fnmatch(entry.indicator.lower(), self.indicator.lower()):
                return False

        if self.timeframe is not None and entry.timeframe != self.timeframe:
            return False

        if self.data_hash is not None and entry.data_hash != self.data_hash:
            return False

        if self.older_than is not None:
            if entry.created_at is None:
                # Can't determine age, don't match
                return False
            if now is None:
                now = datetime.now(tz=entry.created_at.tzinfo)
            age = now - entry.created_at
            threshold = parse_duration(self.older_than)
            if age < threshold:
                return False

        return True

    @property
    def is_empty(self) -> bool:
        """Check if criteria is empty (matches all entries)."""
        return all(
            v is None
            for v in [self.older_than, self.symbol, self.indicator, self.timeframe, self.data_hash]
        )


class CleanupResult(BaseModel):
    """Result of a cache cleanup operation.

    Attributes:
        deleted_count: Number of entries deleted.
        freed_bytes: Bytes freed by deletion.
        errors: List of errors encountered during cleanup.
        dry_run: Whether this was a dry run (no actual deletions).
    """

    deleted_count: int = 0
    freed_bytes: int = 0
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False

    @property
    def freed_mb(self) -> float:
        """Freed space in megabytes."""
        return self.freed_bytes / (1024 * 1024)

    @property
    def freed_gb(self) -> float:
        """Freed space in gigabytes."""
        return self.freed_bytes / (1024 * 1024 * 1024)

    @property
    def success(self) -> bool:
        """Whether cleanup completed without errors."""
        return len(self.errors) == 0

    def add_deletion(self, entry: CacheEntry) -> None:
        """Record a successful deletion.

        Args:
            entry: Deleted cache entry.
        """
        self.deleted_count += 1
        self.freed_bytes += entry.size_bytes

    def add_error(self, message: str) -> None:
        """Record an error.

        Args:
            message: Error message.
        """
        self.errors.append(message)


class CacheFilter(BaseModel):
    """Filter for listing cache entries.

    Attributes:
        symbol: Filter by symbol.
        indicator: Filter by indicator (supports glob patterns).
        timeframe: Filter by timeframe.
        limit: Maximum number of entries to return.
        offset: Number of entries to skip.
    """

    symbol: str | None = None
    indicator: str | None = None
    timeframe: str | None = None
    limit: int | None = None
    offset: int = 0

    def matches(self, entry: CacheEntry) -> bool:
        """Check if an entry matches this filter.

        Args:
            entry: Cache entry to check.

        Returns:
            True if entry matches filter.
        """
        import fnmatch

        if self.symbol is not None and entry.symbol != self.symbol:
            return False

        if self.indicator is not None:
            if not fnmatch.fnmatch(entry.indicator.lower(), self.indicator.lower()):
                return False

        if self.timeframe is not None and entry.timeframe != self.timeframe:
            return False

        return True
