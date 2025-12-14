"""Tests for liq.features.microstructure module."""

import polars as pl

from liq.features.microstructure import corwin_schultz_spread


class TestCorwinSchultzSpread:
    """Tests for corwin_schultz_spread function."""

    def test_positive_spread(self) -> None:
        """Test spread is non-negative."""
        df = pl.DataFrame(
            {
                "high": pl.Series("high", [2.0, 2.2, 2.4], dtype=pl.Float64),
                "low": pl.Series("low", [1.8, 1.9, 2.0], dtype=pl.Float64),
                "open": pl.Series("open", [2.0, 2.0, 2.0], dtype=pl.Float64),
                "close": pl.Series("close", [2.0, 2.0, 2.0], dtype=pl.Float64),
            }
        )
        spread = corwin_schultz_spread(df)
        assert spread >= 0

    def test_single_row_returns_zero(self) -> None:
        """Test single row returns zero."""
        df = pl.DataFrame(
            {
                "high": [2.0],
                "low": [1.8],
            }
        )
        spread = corwin_schultz_spread(df)
        assert spread == 0.0

    def test_empty_returns_zero(self) -> None:
        """Test empty DataFrame returns zero."""
        df = pl.DataFrame(
            {
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
            }
        )
        spread = corwin_schultz_spread(df)
        assert spread == 0.0

    def test_equal_high_low_returns_zero(self) -> None:
        """Test equal high and low returns zero (gamma = 0)."""
        df = pl.DataFrame(
            {
                "high": [2.0, 2.0, 2.0],
                "low": [2.0, 2.0, 2.0],
            }
        )
        spread = corwin_schultz_spread(df)
        assert spread == 0.0

    def test_returns_float(self) -> None:
        """Test function returns a float."""
        df = pl.DataFrame(
            {
                "high": [2.0, 2.2, 2.4, 2.3],
                "low": [1.8, 1.9, 2.0, 2.1],
            }
        )
        spread = corwin_schultz_spread(df)
        assert isinstance(spread, float)
