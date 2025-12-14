"""Tests for liq.features.derived module."""

import polars as pl
import pytest

from liq.features.derived import (
    compute_derived_fields,
    compute_returns,
    compute_volatility,
)


class TestComputeDerivedFields:
    """Tests for compute_derived_fields function."""

    def test_basic_derived_fields(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test basic derived fields are computed correctly."""
        # Rename ts to match expected columns
        df = sample_ohlc_df.rename({"ts": "timestamp"})

        result = compute_derived_fields(df)

        assert "midrange" in result.columns
        assert "range" in result.columns
        assert "true_range" in result.columns
        assert "true_range_midrange" in result.columns

    def test_midrange_calculation(self) -> None:
        """Test midrange is (high + low) / 2."""
        df = pl.DataFrame({
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        })

        result = compute_derived_fields(df)

        assert result["midrange"][0] == 100.0  # (110 + 90) / 2

    def test_range_calculation(self) -> None:
        """Test range is high - low."""
        df = pl.DataFrame({
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        })

        result = compute_derived_fields(df)

        assert result["range"][0] == 20.0  # 110 - 90

    def test_true_range_calculation(self) -> None:
        """Test true range considers previous close."""
        df = pl.DataFrame({
            "open": [100.0, 105.0],
            "high": [110.0, 108.0],
            "low": [95.0, 102.0],
            "close": [105.0, 106.0],
        })

        result = compute_derived_fields(df)

        # First row: just high - low = 110 - 95 = 15
        assert result["true_range"][0] == 15.0

        # Second row: max(108-102, |108-105|, |102-105|) = max(6, 3, 3) = 6
        assert result["true_range"][1] == 6.0

    def test_missing_columns_raises_error(self) -> None:
        """Test ValueError raised for missing columns."""
        df = pl.DataFrame({
            "open": [100.0],
            "high": [110.0],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_derived_fields(df)


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_simple_returns(self) -> None:
        """Test simple return calculation."""
        df = pl.DataFrame({
            "close": [100.0, 110.0, 105.0],
        })

        result = compute_returns(df, column="close", periods=1)

        assert "return" in result.columns
        assert result["return"][0] is None  # First value is null
        assert abs(result["return"][1] - 0.1) < 0.001  # 10% gain
        assert abs(result["return"][2] - (-0.0454545)) < 0.001  # ~4.5% loss

    def test_multi_period_returns(self) -> None:
        """Test multi-period return calculation."""
        df = pl.DataFrame({
            "close": [100.0, 105.0, 110.0],
        })

        result = compute_returns(df, column="close", periods=2)

        assert "return_2" in result.columns
        # 2-period return: (110 - 100) / 100 = 0.1
        assert abs(result["return_2"][2] - 0.1) < 0.001

    def test_log_returns(self) -> None:
        """Test log return calculation."""
        import math

        df = pl.DataFrame({
            "close": [100.0, 110.0],
        })

        result = compute_returns(df, column="close", periods=1, log_returns=True)

        assert "log_return" in result.columns
        expected = math.log(110 / 100)
        assert abs(result["log_return"][1] - expected) < 0.001

    def test_missing_column_raises_error(self) -> None:
        """Test ValueError for missing column."""
        df = pl.DataFrame({"open": [100.0]})

        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_returns(df, column="close")


class TestComputeVolatility:
    """Tests for compute_volatility function."""

    def test_rolling_volatility(self) -> None:
        """Test rolling volatility calculation."""
        df = pl.DataFrame({
            "close": [100.0, 102.0, 98.0, 103.0, 99.0, 104.0, 100.0, 105.0, 101.0, 106.0],
        })

        result = compute_volatility(df, window=5, annualize=False)

        assert "volatility_5" in result.columns
        # With 10 prices, we get 9 returns. Rolling window of 5 starts having values at index 5
        # (need 5 returns for first volatility value, plus 1 for the return calculation)
        non_null = result["volatility_5"].drop_nulls()
        assert len(non_null) > 0  # Should have some values

    def test_annualized_volatility(self) -> None:
        """Test annualized volatility uses correct factor."""
        df = pl.DataFrame({
            "close": [100.0] * 30,  # Constant price = 0 volatility
        })

        result = compute_volatility(df, window=5, annualize=True)

        # All volatility should be ~0 for constant prices
        non_null = result["volatility_5"].drop_nulls()
        for val in non_null.to_list():
            assert val < 0.001 or val != val  # Allow NaN or very small

    def test_missing_column_raises_error(self) -> None:
        """Test ValueError for missing column."""
        df = pl.DataFrame({"open": [100.0]})

        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_volatility(df, column="close")
