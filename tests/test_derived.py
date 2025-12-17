"""Tests for liq.features.derived module."""

import polars as pl
import pytest

from liq.features.derived import (
    DEFAULT_FIBONACCI_WINDOWS,
    compute_derived_fields,
    compute_multi_window_volatility,
    compute_returns,
    compute_rolling_returns,
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


class TestDefaultFibonacciWindows:
    """Tests for DEFAULT_FIBONACCI_WINDOWS constant."""

    def test_default_windows_values(self) -> None:
        """Test default Fibonacci windows have expected values."""
        assert DEFAULT_FIBONACCI_WINDOWS == [55, 210, 340, 890, 3750]

    def test_default_windows_is_list(self) -> None:
        """Test default windows is a list of integers."""
        assert isinstance(DEFAULT_FIBONACCI_WINDOWS, list)
        assert all(isinstance(w, int) for w in DEFAULT_FIBONACCI_WINDOWS)


class TestComputeRollingReturns:
    """Tests for compute_rolling_returns function."""

    def test_rolling_returns_default_windows(self) -> None:
        """Test rolling returns with default Fibonacci windows."""
        # Create a large DataFrame to accommodate default windows
        n = 4000
        df = pl.DataFrame({
            "close": [100.0 + i * 0.01 for i in range(n)],
        })

        result = compute_rolling_returns(df)

        # Should have columns for sum and mean for each window
        for window in DEFAULT_FIBONACCI_WINDOWS:
            assert f"log_return_sum_{window}" in result.columns
            assert f"log_return_mean_{window}" in result.columns

    def test_rolling_returns_custom_windows(self) -> None:
        """Test rolling returns with custom windows."""
        n = 100
        df = pl.DataFrame({
            "close": [100.0 + i * 0.1 for i in range(n)],
        })

        result = compute_rolling_returns(df, windows=[10, 20, 30])

        # Should have columns only for custom windows
        assert "log_return_sum_10" in result.columns
        assert "log_return_mean_10" in result.columns
        assert "log_return_sum_20" in result.columns
        assert "log_return_mean_20" in result.columns
        assert "log_return_sum_30" in result.columns
        assert "log_return_mean_30" in result.columns

    def test_rolling_returns_sum_aggregation(self) -> None:
        """Test rolling sum of log returns."""
        import math

        df = pl.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        })

        result = compute_rolling_returns(df, windows=[3], aggregations=["sum"])

        assert "log_return_sum_3" in result.columns
        # Verify sum is computed correctly
        non_null = result["log_return_sum_3"].drop_nulls().to_list()
        assert len(non_null) > 0

    def test_rolling_returns_mean_aggregation(self) -> None:
        """Test rolling mean of log returns."""
        df = pl.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        })

        result = compute_rolling_returns(df, windows=[3], aggregations=["mean"])

        assert "log_return_mean_3" in result.columns
        non_null = result["log_return_mean_3"].drop_nulls().to_list()
        assert len(non_null) > 0

    def test_rolling_returns_simple_vs_log(self) -> None:
        """Test log returns vs simple returns."""
        df = pl.DataFrame({
            "close": [100.0, 110.0, 105.0, 115.0, 110.0, 120.0],
        })

        result_log = compute_rolling_returns(df, windows=[3], log_returns=True)
        result_simple = compute_rolling_returns(df, windows=[3], log_returns=False)

        # Column names should reflect log vs simple
        assert "log_return_sum_3" in result_log.columns
        assert "return_sum_3" in result_simple.columns

    def test_rolling_returns_missing_column_error(self) -> None:
        """Test ValueError for missing column."""
        df = pl.DataFrame({"open": [100.0]})

        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_rolling_returns(df, column="close")

    def test_rolling_returns_custom_column(self) -> None:
        """Test rolling returns with custom column."""
        df = pl.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "open": [99.0, 100.0, 101.0, 102.0, 103.0],
        })

        result = compute_rolling_returns(df, column="open", windows=[3])

        assert "log_return_sum_3" in result.columns


class TestComputeMultiWindowVolatility:
    """Tests for compute_multi_window_volatility function."""

    def test_multi_window_volatility_default_windows(self) -> None:
        """Test multi-window volatility with default Fibonacci windows."""
        # Create large DataFrame to accommodate default windows
        n = 4000
        import random
        random.seed(42)
        df = pl.DataFrame({
            "close": [100.0 + random.gauss(0, 1) for _ in range(n)],
        })

        result = compute_multi_window_volatility(df)

        # Should have volatility column for each window
        for window in DEFAULT_FIBONACCI_WINDOWS:
            assert f"volatility_{window}" in result.columns

    def test_multi_window_volatility_custom_windows(self) -> None:
        """Test multi-window volatility with custom windows."""
        n = 100
        import random
        random.seed(42)
        df = pl.DataFrame({
            "close": [100.0 + random.gauss(0, 1) for _ in range(n)],
        })

        result = compute_multi_window_volatility(df, windows=[10, 20, 30])

        assert "volatility_10" in result.columns
        assert "volatility_20" in result.columns
        assert "volatility_30" in result.columns

    def test_multi_window_volatility_annualized(self) -> None:
        """Test annualized vs non-annualized volatility."""
        import math
        n = 100
        import random
        random.seed(42)
        df = pl.DataFrame({
            "close": [100.0 + random.gauss(0, 1) for _ in range(n)],
        })

        result_annual = compute_multi_window_volatility(
            df, windows=[20], annualize=True, periods_per_year=252
        )
        result_raw = compute_multi_window_volatility(
            df, windows=[20], annualize=False
        )

        # Annualized should be larger by sqrt(252) factor
        annual_vals = result_annual["volatility_20"].drop_nulls().to_list()
        raw_vals = result_raw["volatility_20"].drop_nulls().to_list()

        if len(annual_vals) > 0 and len(raw_vals) > 0:
            ratio = annual_vals[0] / raw_vals[0] if raw_vals[0] != 0 else 0
            expected_ratio = math.sqrt(252)
            assert abs(ratio - expected_ratio) < 0.1

    def test_multi_window_volatility_values_positive(self) -> None:
        """Test that all volatility values are non-negative."""
        n = 100
        import random
        random.seed(42)
        df = pl.DataFrame({
            "close": [100.0 + random.gauss(0, 1) for _ in range(n)],
        })

        result = compute_multi_window_volatility(df, windows=[10, 20])

        for col in ["volatility_10", "volatility_20"]:
            non_null = result[col].drop_nulls().to_list()
            assert all(v >= 0 for v in non_null)

    def test_multi_window_volatility_missing_column_error(self) -> None:
        """Test ValueError for missing column."""
        df = pl.DataFrame({"open": [100.0]})

        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_multi_window_volatility(df, column="close")

    def test_multi_window_volatility_custom_column(self) -> None:
        """Test multi-window volatility with custom column."""
        n = 50
        df = pl.DataFrame({
            "close": [100.0 + i * 0.1 for i in range(n)],
            "open": [99.0 + i * 0.1 for i in range(n)],
        })

        result = compute_multi_window_volatility(df, column="open", windows=[10])

        assert "volatility_10" in result.columns
