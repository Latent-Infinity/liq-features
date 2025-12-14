"""Tests for liq.features.volatility module."""

import polars as pl

from liq.features.volatility import garman_klass, yang_zhang


class TestYangZhang:
    """Tests for yang_zhang function."""

    def test_positive_volatility(self) -> None:
        """Test returns positive volatility for valid data."""
        df = pl.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
            }
        )
        vol = yang_zhang(df)
        assert vol > 0

    def test_single_row_returns_zero(self) -> None:
        """Test single row returns zero."""
        df = pl.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
            }
        )
        vol = yang_zhang(df)
        assert vol == 0.0

    def test_empty_returns_zero(self) -> None:
        """Test empty DataFrame returns zero."""
        df = pl.DataFrame(
            {
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )
        vol = yang_zhang(df)
        assert vol == 0.0

    def test_returns_float(self) -> None:
        """Test function returns a float."""
        df = pl.DataFrame(
            {
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [2.0, 3.0, 4.0, 5.0],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.5, 2.5, 3.5, 4.5],
            }
        )
        vol = yang_zhang(df)
        assert isinstance(vol, float)


class TestGarmanKlass:
    """Tests for garman_klass function."""

    def test_positive_volatility(self) -> None:
        """Test returns positive volatility for valid data."""
        df = pl.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
            }
        )
        vol = garman_klass(df)
        assert vol > 0

    def test_single_row_returns_zero(self) -> None:
        """Test single row returns zero."""
        df = pl.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
            }
        )
        vol = garman_klass(df)
        assert vol == 0.0

    def test_empty_returns_zero(self) -> None:
        """Test empty DataFrame returns zero."""
        df = pl.DataFrame(
            {
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )
        vol = garman_klass(df)
        assert vol == 0.0

    def test_returns_float(self) -> None:
        """Test function returns a float."""
        df = pl.DataFrame(
            {
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [2.0, 3.0, 4.0, 5.0],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.5, 2.5, 3.5, 4.5],
            }
        )
        vol = garman_klass(df)
        assert isinstance(vol, float)
