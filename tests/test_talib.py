"""Tests for liq.features.indicators.talib module."""

from datetime import UTC, datetime
from unittest.mock import patch

import polars as pl
import pytest

from liq.features.indicators.talib import (
    HAS_TALIB,
    _check_talib,
    _create_dynamic_indicator_class,
    _indicator_class_cache,
    clear_indicator_cache,
    format_outputs,
    get_available_indicators,
    get_dynamic_indicator,
    get_indicator_info,
    list_dynamic_indicators,
    map_inputs,
)


# Skip all tests if TA-Lib is not installed
pytestmark = pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")


@pytest.fixture
def sample_ohlc_df() -> pl.DataFrame:
    """Create a sample OHLC DataFrame for testing."""
    n_rows = 100
    base_price = 100.0

    timestamps = [
        datetime(2024, 1, 1, i // 24, i % 24, 0, tzinfo=UTC) for i in range(n_rows)
    ]

    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = base_price
    for i in range(n_rows):
        import math

        trend = math.sin(i / 10) * 2
        noise = (i % 7 - 3) * 0.5

        open_price = price
        close_price = price + trend + noise
        high_price = max(open_price, close_price) + abs(noise) + 1
        low_price = min(open_price, close_price) - abs(noise) - 0.5

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(1000 + i * 10)

        price = close_price

    return pl.DataFrame(
        {
            "ts": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


class TestCheckTalib:
    """Tests for _check_talib function."""

    def test_check_talib_does_not_raise_when_available(self) -> None:
        """Test _check_talib does not raise when TA-Lib is installed."""
        # If we're running this test, TA-Lib is available
        _check_talib()  # Should not raise


class TestCheckTalibNotInstalled:
    """Tests for _check_talib when TA-Lib is not available."""

    def test_check_talib_raises_import_error(self) -> None:
        """Test _check_talib raises ImportError when TA-Lib is not installed."""
        with patch("liq.features.indicators.talib.HAS_TALIB", False):
            with pytest.raises(ImportError, match="TA-Lib is required"):
                _check_talib()


class TestGetAvailableIndicators:
    """Tests for get_available_indicators function."""

    def test_returns_list(self) -> None:
        """Test function returns a list."""
        indicators = get_available_indicators()
        assert isinstance(indicators, list)

    def test_returns_lowercase_names(self) -> None:
        """Test all indicator names are lowercase."""
        indicators = get_available_indicators()
        for name in indicators:
            assert name == name.lower()

    def test_returns_sorted_list(self) -> None:
        """Test list is sorted alphabetically."""
        indicators = get_available_indicators()
        assert indicators == sorted(indicators)

    def test_contains_common_indicators(self) -> None:
        """Test list contains well-known indicators."""
        indicators = get_available_indicators()
        assert "rsi" in indicators
        assert "sma" in indicators
        assert "ema" in indicators
        assert "macd" in indicators
        assert "cci" in indicators

    def test_returns_many_indicators(self) -> None:
        """Test returns many indicators (TA-Lib has 150+)."""
        indicators = get_available_indicators()
        assert len(indicators) >= 100


class TestGetIndicatorInfo:
    """Tests for get_indicator_info function."""

    def test_returns_dict(self) -> None:
        """Test function returns a dictionary."""
        info = get_indicator_info("rsi")
        assert isinstance(info, dict)

    def test_contains_required_keys(self) -> None:
        """Test dict contains required keys."""
        info = get_indicator_info("sma")
        assert "name" in info
        assert "display_name" in info
        assert "group" in info
        assert "inputs" in info
        assert "parameters" in info
        assert "outputs" in info

    def test_case_insensitive(self) -> None:
        """Test indicator lookup is case-insensitive."""
        info1 = get_indicator_info("RSI")
        info2 = get_indicator_info("rsi")
        info3 = get_indicator_info("Rsi")

        assert info1["name"] == info2["name"] == info3["name"] == "RSI"

    def test_unknown_indicator_raises(self) -> None:
        """Test unknown indicator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown TA-Lib indicator"):
            get_indicator_info("not_a_real_indicator_xyz")

    def test_rsi_info(self) -> None:
        """Test RSI indicator info."""
        info = get_indicator_info("rsi")
        assert info["name"] == "RSI"
        assert "timeperiod" in info["parameters"]
        assert "real" in info["outputs"] or "RSI" in info["outputs"]

    def test_macd_info_has_multiple_outputs(self) -> None:
        """Test MACD indicator info has multiple outputs."""
        info = get_indicator_info("macd")
        assert info["name"] == "MACD"
        assert len(info["outputs"]) >= 3  # MACD, signal, histogram

    def test_bbands_info_has_multiple_outputs(self) -> None:
        """Test Bollinger Bands info has multiple outputs."""
        info = get_indicator_info("bbands")
        assert len(info["outputs"]) == 3  # upper, middle, lower


class TestCreateDynamicIndicatorClass:
    """Tests for _create_dynamic_indicator_class function."""

    def test_creates_class(self) -> None:
        """Test function creates a class."""
        IndicatorClass = _create_dynamic_indicator_class("rsi")
        assert isinstance(IndicatorClass, type)

    def test_class_has_name_attribute(self) -> None:
        """Test created class has name attribute."""
        IndicatorClass = _create_dynamic_indicator_class("sma")
        assert hasattr(IndicatorClass, "name")
        assert IndicatorClass.name == "sma"

    def test_class_has_default_params(self) -> None:
        """Test created class has default_params attribute."""
        IndicatorClass = _create_dynamic_indicator_class("ema")
        assert hasattr(IndicatorClass, "default_params")
        assert isinstance(IndicatorClass.default_params, dict)

    def test_class_has_docstring(self) -> None:
        """Test created class has docstring."""
        IndicatorClass = _create_dynamic_indicator_class("cci")
        assert IndicatorClass.__doc__ is not None
        assert "CCI" in IndicatorClass.__doc__

    def test_class_name_is_uppercase(self) -> None:
        """Test created class __name__ is uppercase."""
        IndicatorClass = _create_dynamic_indicator_class("rsi")
        assert IndicatorClass.__name__ == "RSI"

    def test_unknown_indicator_raises(self) -> None:
        """Test unknown indicator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown TA-Lib indicator"):
            _create_dynamic_indicator_class("fake_indicator_xyz")


class TestGetDynamicIndicator:
    """Tests for get_dynamic_indicator function."""

    def test_returns_class(self) -> None:
        """Test function returns a class."""
        IndicatorClass = get_dynamic_indicator("rsi")
        assert isinstance(IndicatorClass, type)

    def test_case_insensitive(self) -> None:
        """Test lookup is case-insensitive."""
        cls1 = get_dynamic_indicator("RSI")
        cls2 = get_dynamic_indicator("rsi")
        cls3 = get_dynamic_indicator("Rsi")

        # Should return same cached class
        assert cls1 is cls2 is cls3

    def test_caches_classes(self) -> None:
        """Test classes are cached."""
        # Clear cache first
        _indicator_class_cache.clear()

        cls1 = get_dynamic_indicator("sma")
        cls2 = get_dynamic_indicator("sma")

        assert cls1 is cls2
        assert "sma" in _indicator_class_cache

    def test_can_instantiate(self) -> None:
        """Test returned class can be instantiated."""
        RSI = get_dynamic_indicator("rsi")
        indicator = RSI()
        assert indicator is not None

    def test_can_instantiate_with_params(self) -> None:
        """Test returned class can be instantiated with custom params."""
        SMA = get_dynamic_indicator("sma")
        indicator = SMA(params={"timeperiod": 30})
        assert indicator.params["timeperiod"] == 30

    def test_unknown_indicator_raises(self) -> None:
        """Test unknown indicator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown TA-Lib indicator"):
            get_dynamic_indicator("not_real_indicator")


class TestDynamicIndicatorCompute:
    """Tests for dynamic indicator compute method."""

    def test_cci_compute_uses_prices(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test CCI indicator uses high/low/close prices."""
        CCI = get_dynamic_indicator("cci")
        indicator = CCI(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns

    def test_atr_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test ATR indicator computation."""
        ATR = get_dynamic_indicator("atr")
        indicator = ATR(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        # ATR should be positive
        values = result["value"].to_list()
        assert all(v >= 0 for v in values if v is not None)

    def test_adx_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test ADX indicator computation."""
        ADX = get_dynamic_indicator("adx")
        indicator = ADX(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert len(result) > 0

    def test_natr_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test NATR indicator computation."""
        NATR = get_dynamic_indicator("natr")
        indicator = NATR(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns

    def test_willr_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test Williams %R indicator computation (uses prices input)."""
        WILLR = get_dynamic_indicator("willr")
        indicator = WILLR(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        # Williams %R should be between -100 and 0
        values = result["value"].to_list()
        assert all(-100 <= v <= 0 for v in values if v is not None)

    def test_dx_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test DX indicator computation."""
        DX = get_dynamic_indicator("dx")
        indicator = DX(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert len(result) > 0

    def test_plus_di_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test Plus DI indicator computation."""
        PLUS_DI = get_dynamic_indicator("plus_di")
        indicator = PLUS_DI(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns

    def test_minus_di_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test Minus DI indicator computation."""
        MINUS_DI = get_dynamic_indicator("minus_di")
        indicator = MINUS_DI(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns


class TestListDynamicIndicators:
    """Tests for list_dynamic_indicators function."""

    def test_returns_list(self) -> None:
        """Test function returns a list."""
        indicators = list_dynamic_indicators()
        assert isinstance(indicators, list)

    def test_returns_list_of_dicts(self) -> None:
        """Test returns list of dictionaries."""
        indicators = list_dynamic_indicators()
        assert all(isinstance(ind, dict) for ind in indicators)

    def test_dicts_have_required_keys(self) -> None:
        """Test each dict has required keys."""
        indicators = list_dynamic_indicators()
        if indicators:
            ind = indicators[0]
            assert "name" in ind
            assert "display_name" in ind
            assert "group" in ind
            assert "inputs" in ind
            assert "parameters" in ind
            assert "outputs" in ind

    def test_returns_many_indicators(self) -> None:
        """Test returns many indicators."""
        indicators = list_dynamic_indicators()
        assert len(indicators) >= 100

    def test_names_are_lowercase(self) -> None:
        """Test all names are lowercase."""
        indicators = list_dynamic_indicators()
        for ind in indicators:
            assert ind["name"] == ind["name"].lower()


class TestClearIndicatorCache:
    """Tests for clear_indicator_cache function."""

    def test_clears_cache(self) -> None:
        """Test cache is cleared."""
        # Ensure something is in cache
        get_dynamic_indicator("rsi")
        assert len(_indicator_class_cache) > 0

        clear_indicator_cache()
        assert len(_indicator_class_cache) == 0

    def test_new_classes_after_clear(self) -> None:
        """Test new classes are created after clearing cache."""
        cls1 = get_dynamic_indicator("sma")
        clear_indicator_cache()
        cls2 = get_dynamic_indicator("sma")

        # Should be different instances (not same reference)
        assert cls1 is not cls2


class TestMapInputs:
    """Tests for map_inputs function."""

    def test_prices_list_mapping(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test mapping for 'prices' with list specification."""
        from collections import OrderedDict

        # MFI requires high, low, close, volume
        input_names = OrderedDict({"prices": ["high", "low", "close", "volume"]})
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="mfi")

        assert "high" in inputs
        assert "low" in inputs
        assert "close" in inputs
        assert "volume" in inputs
        assert len(inputs["high"]) == len(sample_ohlc_df)

    def test_prices_with_open(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test mapping for 'prices' that includes open."""
        from collections import OrderedDict

        # BOP requires open, high, low, close
        input_names = OrderedDict({"prices": ["open", "high", "low", "close"]})
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="bop")

        assert "open" in inputs
        assert "high" in inputs
        assert "low" in inputs
        assert "close" in inputs

    def test_price_mapping(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test mapping for 'price' input type."""
        from collections import OrderedDict

        input_names = OrderedDict({"price": "close"})
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="obv")

        assert "close" in inputs
        assert len(inputs["close"]) == len(sample_ohlc_df)

    def test_real_mapping(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test mapping for 'real' input type."""
        from collections import OrderedDict

        input_names = OrderedDict({"real": None})
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="rsi")

        assert "close" in inputs

    def test_real_with_custom_price_column(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test mapping uses custom price column for 'real' input."""
        from collections import OrderedDict

        # Add midrange column
        df = sample_ohlc_df.with_columns(
            ((pl.col("high") + pl.col("low")) / 2).alias("midrange")
        )
        input_names = OrderedDict({"real": None})

        inputs = map_inputs(df, input_names, price_column="midrange", indicator_name="rsi")

        # Should use midrange values (mapped to 'close' key)
        assert "close" in inputs
        # The values should match midrange column
        import numpy as np

        np.testing.assert_array_almost_equal(inputs["close"], df["midrange"].to_numpy())

    def test_missing_column_raises(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test ValueError raised for missing required column."""
        from collections import OrderedDict

        # Remove volume column
        df = sample_ohlc_df.drop("volume")
        input_names = OrderedDict({"prices": ["high", "low", "close", "volume"]})

        with pytest.raises(ValueError, match="requires 'volume' column"):
            map_inputs(df, input_names, indicator_name="mfi")

    def test_real1_uses_high(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test 'real1' input uses high column."""
        from collections import OrderedDict

        input_names = OrderedDict({"real0": None, "real1": None})
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="beta")

        assert "close" in inputs  # real0
        assert "high" in inputs  # real1

    def test_periods_skipped(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test 'periods' input is skipped (handled by params)."""
        from collections import OrderedDict

        input_names = OrderedDict({"real": None, "periods": None})
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="mavp")

        assert "close" in inputs
        assert "periods" not in inputs


class TestFormatOutputs:
    """Tests for format_outputs function."""

    def test_single_output(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test formatting single output."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        values = np.random.randn(len(sample_ohlc_df))

        result = format_outputs(ts_series, values, ["real"])

        assert "ts" in result.columns
        assert "value" in result.columns  # Single output uses 'value'
        assert len(result) <= len(sample_ohlc_df)  # May filter NaN

    def test_multi_output(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test formatting multiple outputs (like MACD)."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        n = len(sample_ohlc_df)
        result_arrays = [
            np.random.randn(n),  # MACD
            np.random.randn(n),  # Signal
            np.random.randn(n),  # Histogram
        ]

        result = format_outputs(ts_series, result_arrays, ["macd", "signal", "histogram"])

        assert "ts" in result.columns
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_tuple_return_type(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test handling tuple return type."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        n = len(sample_ohlc_df)
        result_arrays = (np.random.randn(n), np.random.randn(n))

        result = format_outputs(ts_series, result_arrays, ["aroondown", "aroonup"])

        assert "aroondown" in result.columns
        assert "aroonup" in result.columns

    def test_list_return_type(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test handling list return type (like AROON returns)."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        n = len(sample_ohlc_df)
        result_arrays = [np.random.randn(n), np.random.randn(n)]

        result = format_outputs(ts_series, result_arrays, ["aroondown", "aroonup"])

        assert "aroondown" in result.columns
        assert "aroonup" in result.columns

    def test_filters_nan_single_output(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test NaN filtering for single output."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        values = np.full(len(sample_ohlc_df), np.nan)
        values[50:] = 1.0  # Only last 50 rows have values

        result = format_outputs(ts_series, values, ["real"])

        assert len(result) == 50

    def test_filters_nan_multi_output(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test NaN filtering for multi-output."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        n = len(sample_ohlc_df)
        values1 = np.full(n, np.nan)
        values1[10:] = 1.0  # 90 valid
        values2 = np.full(n, np.nan)
        values2[20:] = 1.0  # 80 valid

        result = format_outputs(ts_series, [values1, values2], ["out1", "out2"])

        # Should have intersection of valid rows
        assert len(result) == 80


class TestVolumeBasedIndicators:
    """Tests for volume-based indicators (MFI, AD, ADOSC, OBV)."""

    def test_mfi_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test MFI (Money Flow Index) requires volume."""
        MFI = get_dynamic_indicator("mfi")
        indicator = MFI(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0  # MFI produces results

    def test_ad_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test AD (Accumulation/Distribution) requires volume."""
        AD = get_dynamic_indicator("ad")
        indicator = AD()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0

    def test_adosc_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test ADOSC (AD Oscillator) requires volume."""
        ADOSC = get_dynamic_indicator("adosc")
        indicator = ADOSC(params={"fastperiod": 3, "slowperiod": 10})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0

    def test_obv_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test OBV (On-Balance Volume) requires volume."""
        OBV = get_dynamic_indicator("obv")
        indicator = OBV()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0


class TestBOPIndicator:
    """Tests for BOP (Balance of Power) indicator requiring open price."""

    def test_bop_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test BOP requires open, high, low, close."""
        BOP = get_dynamic_indicator("bop")
        indicator = BOP()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        # BOP should be between -1 and 1
        values = result["value"].to_list()
        assert all(-1 <= v <= 1 for v in values if v is not None)


class TestMultiOutputIndicators:
    """Tests for multi-output indicators (AROON, STOCH, MACD, etc.)."""

    def test_aroon_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test AROON returns two outputs."""
        AROON = get_dynamic_indicator("aroon")
        indicator = AROON(params={"timeperiod": 14})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "aroondown" in result.columns
        assert "aroonup" in result.columns
        assert len(result) > 0

    def test_stoch_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test STOCH returns slowk and slowd."""
        STOCH = get_dynamic_indicator("stoch")
        indicator = STOCH()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "slowk" in result.columns
        assert "slowd" in result.columns
        # Values should be between 0 and 100
        for col in ["slowk", "slowd"]:
            values = result[col].to_list()
            assert all(0 <= v <= 100 for v in values if v is not None)

    def test_macd_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test MACD returns three outputs."""
        MACD = get_dynamic_indicator("macd")
        indicator = MACD()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        # MACD outputs: macd, macdsignal, macdhist
        assert len(result.columns) >= 3
        assert len(result) > 0

    def test_stochf_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test STOCHF (Fast Stochastic) returns two outputs."""
        STOCHF = get_dynamic_indicator("stochf")
        indicator = STOCHF()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "fastk" in result.columns
        assert "fastd" in result.columns

    def test_bbands_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test Bollinger Bands returns three outputs."""
        BBANDS = get_dynamic_indicator("bbands")
        indicator = BBANDS(params={"timeperiod": 20})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        # BBANDS outputs: upperband, middleband, lowerband
        assert len(result.columns) >= 3


class TestCandlestickPatterns:
    """Tests for candlestick pattern recognition indicators."""

    def test_cdl_doji_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test CDL_DOJI candlestick pattern."""
        CDLDOJI = get_dynamic_indicator("cdldoji")
        indicator = CDLDOJI()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        # Candlestick patterns return 100, -100, or 0
        values = result["value"].unique().to_list()
        assert all(v in [-100, 0, 100] for v in values)

    def test_cdl_hammer_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test CDL_HAMMER candlestick pattern."""
        CDLHAMMER = get_dynamic_indicator("cdlhammer")
        indicator = CDLHAMMER()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns

    def test_cdl_engulfing_compute(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test CDL_ENGULFING candlestick pattern."""
        CDLENGULFING = get_dynamic_indicator("cdlengulfing")
        indicator = CDLENGULFING()

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_indicator_info_has_input_names(self) -> None:
        """Test get_indicator_info includes input_names in response."""
        info = get_indicator_info("mfi")
        assert "input_names" in info
        assert isinstance(info["input_names"], dict)

    def test_prices_without_list_fallback(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test 'prices' input with non-list spec falls back to OHLC."""
        from collections import OrderedDict

        # Simulate non-list prices spec (edge case)
        input_names = OrderedDict({"prices": "close"})  # Not a list
        inputs = map_inputs(sample_ohlc_df, input_names, indicator_name="test")

        # Should fall back to high, low, close
        assert "high" in inputs
        assert "low" in inputs
        assert "close" in inputs

    def test_real1_without_high(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test 'real1' input falls back to close when high not available."""
        from collections import OrderedDict

        # Remove high column
        df = sample_ohlc_df.drop("high")
        input_names = OrderedDict({"real0": None, "real1": None})
        inputs = map_inputs(df, input_names, indicator_name="test")

        # real1 should fall back to close
        assert "close" in inputs

    def test_format_outputs_single_array_multi_output(
        self, sample_ohlc_df: pl.DataFrame
    ) -> None:
        """Test format_outputs handles single array for multi-output (edge case)."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        # Single array instead of list/tuple
        result_arrays = np.random.randn(len(sample_ohlc_df))

        result = format_outputs(ts_series, result_arrays, ["out1", "out2"])

        # Should still work with first output
        assert "ts" in result.columns
        assert "out1" in result.columns

    def test_two_input_indicator_beta(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test BETA indicator uses two inputs (close and high)."""
        BETA = get_dynamic_indicator("beta")
        indicator = BETA(params={"timeperiod": 5})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0

    def test_correlation_indicator(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test CORREL indicator uses two inputs."""
        CORREL = get_dynamic_indicator("correl")
        indicator = CORREL(params={"timeperiod": 30})

        result = indicator.compute(sample_ohlc_df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0

    def test_custom_price_column(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test using custom price column via _price_column param."""
        # Add midrange column
        df = sample_ohlc_df.with_columns(
            ((pl.col("high") + pl.col("low")) / 2).alias("midrange")
        )

        RSI = get_dynamic_indicator("rsi")
        indicator = RSI(params={"timeperiod": 14, "_price_column": "midrange"})

        result = indicator.compute(df)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0

    def test_get_indicator_info_unknown_raises(self) -> None:
        """Test get_indicator_info raises for unknown indicator."""
        with pytest.raises(ValueError, match="Unknown TA-Lib indicator"):
            get_indicator_info("not_a_real_indicator_xyz_123")

    def test_to_float64_already_float64(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test _to_float64 returns same array if already float64."""
        from liq.features.indicators.talib import _to_float64

        import numpy as np

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _to_float64(arr)

        # Should be same object (not a copy)
        assert result is arr

    def test_to_float64_converts_int(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test _to_float64 converts int array to float64."""
        from liq.features.indicators.talib import _to_float64

        import numpy as np

        arr = np.array([1, 2, 3], dtype=np.int64)
        result = _to_float64(arr)

        assert result.dtype == np.float64
        assert list(result) == [1.0, 2.0, 3.0]

    def test_list_dynamic_indicators(self) -> None:
        """Test list_dynamic_indicators returns all available indicators."""
        from liq.features.indicators.talib import list_dynamic_indicators

        indicators = list_dynamic_indicators()

        assert len(indicators) > 100  # TA-Lib has 150+ indicators
        # Check structure of returned items
        assert all("name" in ind for ind in indicators)
        assert all("display_name" in ind for ind in indicators)
        assert all("group" in ind for ind in indicators)
        assert all("inputs" in ind for ind in indicators)
        assert all("parameters" in ind for ind in indicators)
        assert all("outputs" in ind for ind in indicators)

        # Check some known indicators exist
        names = [ind["name"] for ind in indicators]
        assert "rsi" in names
        assert "sma" in names
        assert "macd" in names

    def test_map_inputs_unknown_column_error(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test map_inputs raises for unknown column requirement."""
        from collections import OrderedDict

        input_names: OrderedDict[str, Any] = OrderedDict({"unknown_column": None})

        with pytest.raises(ValueError, match="requires 'unknown_column' column"):
            map_inputs(sample_ohlc_df, input_names, indicator_name="test")

    def test_format_outputs_scalar_result(self, sample_ohlc_df: pl.DataFrame) -> None:
        """Test format_outputs handles scalar result for single output."""
        import numpy as np

        ts_series = sample_ohlc_df["ts"]
        # A numpy scalar value
        result_scalar = np.float64(42.0)

        result = format_outputs(ts_series, result_scalar, ["value"])

        assert "ts" in result.columns
        assert "value" in result.columns

    def test_price0_price1_fallback_to_price_column(
        self, sample_ohlc_df: pl.DataFrame
    ) -> None:
        """Test price0/price1 falls back to price column if spec column missing."""
        from collections import OrderedDict

        # Create df without 'high' - should fall back to price_column
        df_no_high = sample_ohlc_df.select(["ts", "close", "low", "volume"])

        # price0 with spec='missing' should fallback to price_column
        input_names: OrderedDict[str, Any] = OrderedDict({"price0": "missing"})
        inputs = map_inputs(df_no_high, input_names, price_column="close")

        # Should have close (the fallback)
        assert "close" in inputs
