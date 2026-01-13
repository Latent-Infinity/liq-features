"""Custom exceptions for indicator cache operations.

Provides a hierarchy of exceptions for handling cache errors
with contextual information for debugging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CacheError(Exception):
    """Base exception for cache operations.

    All cache-specific exceptions inherit from this class,
    allowing callers to catch all cache errors with a single except.

    Attributes:
        message: Human-readable error message.
        context: Additional context dictionary.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize cache error.

        Args:
            message: Human-readable error message.
            context: Additional context for debugging.
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        if not self.context:
            return self.message
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} ({context_str})"


class CacheNotFoundError(CacheError):
    """Raised when a cache entry is not found.

    This occurs when:
    - Requesting a key that doesn't exist
    - Entry was deleted between check and access

    Attributes:
        key: The cache key that was not found.
    """

    def __init__(
        self,
        message: str,
        key: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize cache not found error.

        Args:
            message: Human-readable error message.
            key: Cache key that was not found.
            context: Additional context for debugging.
        """
        self.key = key
        full_context = {"key": key}
        if context:
            full_context.update(context)
        super().__init__(message, full_context)


class CacheCorruptionError(CacheError):
    """Raised when a cache entry is corrupted.

    This occurs when:
    - Parquet file is malformed or truncated
    - Data schema doesn't match expected format
    - File was modified externally

    Attributes:
        key: The cache key with corrupted data.
        path: File path if known.
        reason: Description of the corruption.
    """

    def __init__(
        self,
        message: str,
        key: str,
        path: Path | str | None = None,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize cache corruption error.

        Args:
            message: Human-readable error message.
            key: Cache key with corrupted data.
            path: File path if known.
            reason: Description of the corruption.
            context: Additional context for debugging.
        """
        self.key = key
        self.path = Path(path) if isinstance(path, str) else path
        self.reason = reason
        full_context = {"key": key}
        if path is not None:
            full_context["path"] = str(path)
        if reason is not None:
            full_context["reason"] = reason
        if context:
            full_context.update(context)
        super().__init__(message, full_context)


class CacheCleanupError(CacheError):
    """Raised when cache cleanup fails.

    This occurs when:
    - Permission denied during deletion
    - File is locked by another process
    - Storage backend is unavailable

    Attributes:
        key: The cache key that failed to delete (if single key).
        failed_keys: List of keys that failed (if batch operation).
        operation: The operation that failed.
    """

    def __init__(
        self,
        message: str,
        key: str | None = None,
        failed_keys: list[str] | None = None,
        operation: str = "delete",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize cache cleanup error.

        Args:
            message: Human-readable error message.
            key: Single cache key that failed (for single deletion).
            failed_keys: List of keys that failed (for batch operations).
            operation: The operation that failed.
            context: Additional context for debugging.
        """
        self.key = key
        self.failed_keys = failed_keys or []
        self.operation = operation
        full_context = {"operation": operation}
        if key is not None:
            full_context["key"] = key
        if failed_keys:
            full_context["failed_count"] = len(failed_keys)
        if context:
            full_context.update(context)
        super().__init__(message, full_context)


class CacheConfigurationError(CacheError):
    """Raised when cache configuration is invalid.

    This occurs when:
    - Invalid compression level specified
    - Invalid data root path
    - Missing required environment variables

    Attributes:
        parameter: The parameter that is invalid.
        value: The invalid value.
        valid_range: Description of valid values.
    """

    def __init__(
        self,
        message: str,
        parameter: str,
        value: Any,
        valid_range: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize cache configuration error.

        Args:
            message: Human-readable error message.
            parameter: Parameter name that is invalid.
            value: Invalid value provided.
            valid_range: Description of valid values.
            context: Additional context for debugging.
        """
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range
        full_context = {"parameter": parameter, "value": value}
        if valid_range is not None:
            full_context["valid_range"] = valid_range
        if context:
            full_context.update(context)
        super().__init__(message, full_context)
