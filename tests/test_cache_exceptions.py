"""Tests for cache exception hierarchy."""

from pathlib import Path

import pytest

from liq.features.cache_exceptions import (
    CacheCleanupError,
    CacheConfigurationError,
    CacheCorruptionError,
    CacheError,
    CacheNotFoundError,
)


class TestCacheError:
    """Tests for base CacheError."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = CacheError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.context == {}

    def test_message_with_context(self) -> None:
        """Test error message with context."""
        error = CacheError("Failed", context={"key": "abc", "size": 100})
        assert "key=abc" in str(error)
        assert "size=100" in str(error)
        assert error.context == {"key": "abc", "size": 100}

    def test_inheritance(self) -> None:
        """Test that CacheError is an Exception."""
        error = CacheError("test")
        assert isinstance(error, Exception)


class TestCacheNotFoundError:
    """Tests for CacheNotFoundError."""

    def test_basic(self) -> None:
        """Test basic not found error."""
        error = CacheNotFoundError("Entry not found", key="BTC_USDT/indicators/rsi/abc")
        assert error.key == "BTC_USDT/indicators/rsi/abc"
        assert "key=" in str(error)
        assert isinstance(error, CacheError)

    def test_with_context(self) -> None:
        """Test not found error with additional context."""
        error = CacheNotFoundError(
            "Entry not found",
            key="test_key",
            context={"operation": "get"},
        )
        assert error.key == "test_key"
        assert "operation=get" in str(error)

    def test_catch_as_cache_error(self) -> None:
        """Test that CacheNotFoundError can be caught as CacheError."""
        with pytest.raises(CacheError):
            raise CacheNotFoundError("Not found", key="test")


class TestCacheCorruptionError:
    """Tests for CacheCorruptionError."""

    def test_basic(self) -> None:
        """Test basic corruption error."""
        error = CacheCorruptionError("Data corrupted", key="test_key")
        assert error.key == "test_key"
        assert error.path is None
        assert error.reason is None
        assert isinstance(error, CacheError)

    def test_with_path(self) -> None:
        """Test corruption error with file path."""
        error = CacheCorruptionError(
            "Data corrupted",
            key="test_key",
            path="/path/to/file.parquet",
        )
        assert error.path == Path("/path/to/file.parquet")
        assert "path=" in str(error)

    def test_with_path_object(self) -> None:
        """Test corruption error with Path object."""
        error = CacheCorruptionError(
            "Data corrupted",
            key="test_key",
            path=Path("/path/to/file.parquet"),
        )
        assert error.path == Path("/path/to/file.parquet")

    def test_with_reason(self) -> None:
        """Test corruption error with reason."""
        error = CacheCorruptionError(
            "Data corrupted",
            key="test_key",
            reason="Invalid parquet schema",
        )
        assert error.reason == "Invalid parquet schema"
        assert "reason=Invalid parquet schema" in str(error)

    def test_full_context(self) -> None:
        """Test corruption error with all fields."""
        error = CacheCorruptionError(
            "Data corrupted",
            key="test_key",
            path="/path/to/file.parquet",
            reason="Truncated file",
            context={"size": 0},
        )
        assert "key=test_key" in str(error)
        assert "path=" in str(error)
        assert "reason=Truncated file" in str(error)
        assert "size=0" in str(error)


class TestCacheCleanupError:
    """Tests for CacheCleanupError."""

    def test_single_key_failure(self) -> None:
        """Test cleanup error for single key."""
        error = CacheCleanupError("Failed to delete", key="test_key")
        assert error.key == "test_key"
        assert error.failed_keys == []
        assert error.operation == "delete"
        assert isinstance(error, CacheError)

    def test_batch_failure(self) -> None:
        """Test cleanup error for batch operation."""
        failed = ["key1", "key2", "key3"]
        error = CacheCleanupError("Batch cleanup failed", failed_keys=failed)
        assert error.key is None
        assert error.failed_keys == failed
        assert "failed_count=3" in str(error)

    def test_custom_operation(self) -> None:
        """Test cleanup error with custom operation."""
        error = CacheCleanupError("Failed", operation="rebuild_index")
        assert error.operation == "rebuild_index"
        assert "operation=rebuild_index" in str(error)

    def test_with_context(self) -> None:
        """Test cleanup error with additional context."""
        error = CacheCleanupError(
            "Failed to delete",
            key="test_key",
            context={"reason": "Permission denied"},
        )
        assert "reason=Permission denied" in str(error)


class TestCacheConfigurationError:
    """Tests for CacheConfigurationError."""

    def test_basic(self) -> None:
        """Test basic configuration error."""
        error = CacheConfigurationError(
            "Invalid compression level",
            parameter="compression_level",
            value=25,
        )
        assert error.parameter == "compression_level"
        assert error.value == 25
        assert error.valid_range is None
        assert isinstance(error, CacheError)

    def test_with_valid_range(self) -> None:
        """Test configuration error with valid range."""
        error = CacheConfigurationError(
            "Invalid compression level",
            parameter="compression_level",
            value=25,
            valid_range="1-22",
        )
        assert error.valid_range == "1-22"
        assert "valid_range=1-22" in str(error)

    def test_full_context(self) -> None:
        """Test configuration error with full context."""
        error = CacheConfigurationError(
            "Invalid path",
            parameter="data_root",
            value="/nonexistent",
            valid_range="existing directory path",
            context={"env_var": "LIQ_DATA_ROOT"},
        )
        assert "parameter=data_root" in str(error)
        assert "value=/nonexistent" in str(error)
        assert "env_var=LIQ_DATA_ROOT" in str(error)


class TestExceptionHierarchy:
    """Tests for exception hierarchy behavior."""

    def test_all_inherit_from_cache_error(self) -> None:
        """Test all exceptions inherit from CacheError."""
        errors = [
            CacheNotFoundError("test", key="k"),
            CacheCorruptionError("test", key="k"),
            CacheCleanupError("test"),
            CacheConfigurationError("test", parameter="p", value="v"),
        ]
        for error in errors:
            assert isinstance(error, CacheError)
            assert isinstance(error, Exception)

    def test_catch_all_cache_errors(self) -> None:
        """Test catching all cache errors with single except."""
        errors_raised = []

        for exc_class in [CacheNotFoundError, CacheCorruptionError, CacheCleanupError]:
            try:
                if exc_class == CacheNotFoundError:
                    raise CacheNotFoundError("test", key="k")
                elif exc_class == CacheCorruptionError:
                    raise CacheCorruptionError("test", key="k")
                else:
                    raise CacheCleanupError("test")
            except CacheError as e:
                errors_raised.append(type(e).__name__)

        assert len(errors_raised) == 3
        assert "CacheNotFoundError" in errors_raised
        assert "CacheCorruptionError" in errors_raised
        assert "CacheCleanupError" in errors_raised
