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
    get_available_indicators,
    get_dynamic_indicator,
    get_indicator_info,
    list_dynamic_indicators,
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
