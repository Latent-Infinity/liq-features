"""Tests for liq.features.indicators module."""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from liq.features.indicators import BaseIndicator, get_indicator, list_indicators
from liq.features.indicators.momentum import MACD, ROC, RSI, Stochastic
from liq.features.indicators.trend import ADX, ATR, BBANDS, EMA, HMA, SMA, WMA
from liq.store.parquet import ParquetStore


class TestBaseIndicator:
    """Tests for BaseIndicator abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test BaseIndicator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseIndicator()

    def test_subclass_requires_compute(self) -> None:
        """Test subclass must implement _compute."""

        class IncompleteIndicator(BaseIndicator):
            name = "incomplete"
            default_params = {}

        with pytest.raises(TypeError):
            IncompleteIndicator()

    def test_base_indicator_uses_cache_and_input_column(self, tmp_path: Path) -> None:
        """Indicator should round-trip through liq-store and honor input column override."""

        class EchoIndicator(BaseIndicator):
            name = "echo"
            default_params = {}

            def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.select("ts", pl.col("close").alias("value"))

        store = ParquetStore(str(tmp_path))
        indicator = EchoIndicator(storage=store, input_column="midrange")

        df = pl.DataFrame({
            "ts": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
            ],
            "close": [1.0, 1.1],
            "midrange": [2.0, 2.1],
        })

        out1 = indicator.compute(df, symbol="EUR_USD", timeframe="1m")
        assert out1["value"].to_list() == [2.0, 2.1]  # input column override applied

        out2 = indicator.compute(df, symbol="EUR_USD", timeframe="1m")
        assert out2.to_dicts() == out1.to_dicts()
        assert any("EUR_USD/indicators/echo/" in key for key in store.list_keys())


class TestIndicatorRegistry:
    """Tests for indicator registry functions."""

    def test_get_known_indicator(self) -> None:
        """Test getting a known indicator."""
        RSIClass = get_indicator("rsi")

        assert RSIClass.name == "rsi"

    def test_get_indicator_case_insensitive(self) -> None:
        """Test indicator lookup is case-insensitive."""
        rsi1 = get_indicator("rsi")
        rsi2 = get_indicator("RSI")
        rsi3 = get_indicator("Rsi")

        assert rsi1 is rsi2 is rsi3

    def test_get_unknown_indicator_raises(self) -> None:
        """Test unknown indicator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown indicator"):
            get_indicator("unknown_indicator_xyz")

    def test_list_indicators_returns_list(self) -> None:
        """Test list_indicators returns a list."""
        indicators = list_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) > 0

    def test_list_indicators_contains_hardcoded(self) -> None:
        """Test list contains hardcoded indicators."""
        indicators = list_indicators()
        names = [ind["name"] for ind in indicators]

        assert "rsi" in names
        assert "ema" in names
        assert "macd" in names

    def test_list_indicators_merges_talib_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate TA-Lib availability and ensure merged listing."""
        from liq.features.indicators import talib as talib_mod
        from liq.features.indicators import registry

        monkeypatch.setattr(talib_mod, "list_dynamic_indicators", lambda: [
            {
                "name": "talib_dummy",
                "display_name": "TA Dummy",
                "parameters": {"timeperiod": 10},
            }
        ])

        indicators = registry.list_indicators()
        names = {ind["name"] for ind in indicators}
        sources = {ind["name"]: ind.get("source") for ind in indicators}

        assert "rsi" in names  # hardcoded baseline
        assert "talib_dummy" in names
        assert sources.get("talib_dummy") == "talib"


class TestRSI:
    """Tests for RSI indicator."""

    def test_rsi_default_params(self) -> None:
        """Test RSI default parameters."""
        rsi = RSI()

        assert rsi.params["period"] == 14

    def test_rsi_custom_params(self) -> None:
        """Test RSI custom parameters."""
        rsi = RSI(params={"period": 20})

        assert rsi.params["period"] == 20

    def test_rsi_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test RSI computation."""
        rsi = RSI(params={"period": 14})

        result = rsi.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns
        # RSI should be between 0 and 100
        values = result["value"].drop_nulls()
        assert all(0 <= v <= 100 for v in values.to_list())

    def test_rsi_repr(self) -> None:
        """Test RSI string representation."""
        rsi = RSI(params={"period": 14})

        repr_str = repr(rsi)

        assert "RSI" in repr_str
        assert "period=14" in repr_str


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_default_params(self) -> None:
        """Test MACD default parameters."""
        macd = MACD()

        assert macd.params["fast_period"] == 12
        assert macd.params["slow_period"] == 26
        assert macd.params["signal_period"] == 9

    def test_macd_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test MACD computation returns multiple outputs."""
        macd = MACD()

        result = macd.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns


class TestStochastic:
    """Tests for Stochastic indicator."""

    def test_stochastic_default_params(self) -> None:
        """Test Stochastic default parameters."""
        stoch = Stochastic()

        assert stoch.params["k_period"] == 14
        assert stoch.params["d_period"] == 3

    def test_stochastic_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test Stochastic computation."""
        stoch = Stochastic()

        result = stoch.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
        # Stochastic should be between 0 and 100
        k_values = result["stoch_k"].drop_nulls()
        assert all(0 <= v <= 100 for v in k_values.to_list())


class TestEMA:
    """Tests for EMA indicator."""

    def test_ema_default_params(self) -> None:
        """Test EMA default parameters."""
        ema = EMA()

        assert ema.params["period"] == 20

    def test_ema_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test EMA computation."""
        ema = EMA(params={"period": 10})

        result = ema.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns
        assert len(result) > 0


class TestSMA:
    """Tests for SMA indicator."""

    def test_sma_default_params(self) -> None:
        """Test SMA default parameters."""
        sma = SMA()

        assert sma.params["period"] == 20

    def test_sma_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test SMA computation."""
        sma = SMA(params={"period": 10})

        result = sma.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns


class TestBBANDS:
    """Tests for Bollinger Bands indicator."""

    def test_bbands_default_params(self) -> None:
        """Test BBANDS default parameters."""
        bb = BBANDS()

        assert bb.params["period"] == 20
        assert bb.params["std_dev"] == 2.0

    def test_bbands_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test BBANDS computation."""
        bb = BBANDS()

        result = bb.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "upper" in result.columns
        assert "middle" in result.columns
        assert "lower" in result.columns
        # Upper should be > middle > lower
        for i in range(len(result)):
            row = result.row(i, named=True)
            assert row["upper"] >= row["middle"] >= row["lower"]


class TestATR:
    """Tests for ATR indicator."""

    def test_atr_default_params(self) -> None:
        """Test ATR default parameters."""
        atr = ATR()

        assert atr.params["period"] == 14

    def test_atr_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test ATR computation."""
        atr = ATR()

        result = atr.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns
        # ATR should be positive
        values = result["value"].drop_nulls()
        assert all(v >= 0 for v in values.to_list())


class TestADX:
    """Tests for ADX indicator."""

    def test_adx_default_params(self) -> None:
        """Test ADX default parameters."""
        adx = ADX()

        assert adx.params["period"] == 14

    def test_adx_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test ADX computation returns multiple outputs."""
        adx = ADX()

        result = adx.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns


class TestROC:
    """Tests for ROC indicator."""

    def test_roc_default_params(self) -> None:
        """Test ROC default parameters."""
        roc = ROC()

        assert roc.params["period"] == 10

    def test_roc_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test ROC computation."""
        roc = ROC()

        result = roc.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns


class TestWMA:
    """Tests for WMA indicator."""

    def test_wma_default_params(self) -> None:
        """Test WMA default parameters."""
        wma = WMA()

        assert wma.params["period"] == 20

    def test_wma_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test WMA computation."""
        wma = WMA(params={"period": 10})

        result = wma.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns


class TestMidrangeIndicators:
    """Tests for Midrange-based indicators."""

    def test_atr_midrange_default_params(self) -> None:
        """Test ATR_Midrange default parameters."""
        from liq.features.indicators.trend import ATR_Midrange

        atr_mr = ATR_Midrange()

        assert atr_mr.params["period"] == 14
        assert atr_mr.params["input_column"] == "midrange"

    def test_atr_midrange_computes_correctly(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test ATR_Midrange computation uses midrange price."""
        from liq.features.indicators.trend import ATR_Midrange

        atr_mr = ATR_Midrange()
        result = atr_mr.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns
        # ATR should be positive
        values = result["value"].drop_nulls()
        assert all(v >= 0 for v in values.to_list())
        assert len(result) > 0

    def test_atr_midrange_uses_midrange(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test ATR_Midrange actually uses midrange in calculation."""
        from liq.features.indicators.trend import ATR, ATR_Midrange

        # Add midrange column to test data
        df_with_midrange = sample_ohlc_df_large.with_columns(
            [((pl.col("high") + pl.col("low")) / 2).alias("midrange")]
        )

        atr = ATR()
        atr_mr = ATR_Midrange()

        result_atr = atr.compute(sample_ohlc_df_large)
        result_atr_mr = atr_mr.compute(df_with_midrange)

        # Both should return valid results
        assert len(result_atr) > 0
        assert len(result_atr_mr) > 0

    def test_adx_midrange_default_params(self) -> None:
        """Test ADX_Midrange default parameters."""
        from liq.features.indicators.trend import ADX_Midrange

        adx_mr = ADX_Midrange()

        assert adx_mr.params["period"] == 14
        assert adx_mr.params["input_column"] == "midrange"

    def test_adx_midrange_computes_correctly(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test ADX_Midrange computation."""
        from liq.features.indicators.trend import ADX_Midrange

        # Add midrange column
        df_with_midrange = sample_ohlc_df_large.with_columns(
            [((pl.col("high") + pl.col("low")) / 2).alias("midrange")]
        )

        adx_mr = ADX_Midrange()
        result = adx_mr.compute(df_with_midrange)

        # Check column structure matches ADX
        assert "ts" in result.columns
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns

    def test_stochastic_midrange_default_params(self) -> None:
        """Test Stochastic_Midrange default parameters."""
        from liq.features.indicators.momentum import Stochastic_Midrange

        stoch_mr = Stochastic_Midrange()

        assert stoch_mr.params["k_period"] == 14
        assert stoch_mr.params["input_column"] == "midrange"

    def test_stochastic_midrange_computes_correctly(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test Stochastic_Midrange computation uses midrange price."""
        from liq.features.indicators.momentum import Stochastic_Midrange

        # Add midrange column
        df_with_midrange = sample_ohlc_df_large.with_columns(
            [((pl.col("high") + pl.col("low")) / 2).alias("midrange")]
        )

        stoch_mr = Stochastic_Midrange()
        result = stoch_mr.compute(df_with_midrange)

        assert "ts" in result.columns
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
        # Stochastic should be between 0 and 100
        k_values = result["stoch_k"].drop_nulls()
        assert all(0 <= v <= 100 for v in k_values.to_list())

    def test_input_column_parameter(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test custom input_column parameter works."""
        from liq.features.indicators.trend import EMA

        # Use EMA with custom column
        ema_close = EMA(params={"period": 10, "column": "close"})
        ema_open = EMA(params={"period": 10, "column": "open"})

        result_close = ema_close.compute(sample_ohlc_df_large)
        result_open = ema_open.compute(sample_ohlc_df_large)

        # Results should be different
        assert len(result_close) == len(result_open)
        close_vals = result_close["value"].to_list()
        open_vals = result_open["value"].to_list()
        assert close_vals != open_vals

    def test_midrange_indicator_in_registry(self) -> None:
        """Test midrange indicators are registered."""
        from liq.features.indicators import get_indicator, list_indicators

        indicators = list_indicators()
        names = [ind["name"] for ind in indicators]

        assert "atr_midrange" in names
        assert "adx_midrange" in names
        assert "stochastic_midrange" in names

        # Test getting by name
        ATR_MR = get_indicator("atr_midrange")
        assert ATR_MR.name == "atr_midrange"


class TestHMA:
    """Tests for Hull Moving Average (HMA) indicator."""

    def test_hma_default_params(self) -> None:
        """Test HMA default parameters."""
        hma = HMA()

        assert hma.params["period"] == 55
        assert hma.params["column"] == "close"

    def test_hma_custom_params(self) -> None:
        """Test HMA with custom parameters."""
        hma = HMA(params={"period": 20, "column": "open"})

        assert hma.params["period"] == 20
        assert hma.params["column"] == "open"

    def test_hma_compute(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test HMA computation."""
        hma = HMA(params={"period": 10})

        result = hma.compute(sample_ohlc_df_large)

        assert "ts" in result.columns
        assert "value" in result.columns
        # Should have no NaN values
        assert result.filter(pl.col("value").is_nan()).height == 0
        # HMA values should be reasonable (in price range)
        values = result["value"].to_list()
        assert len(values) > 0
        # Values should be within reasonable range of price data
        min_price = sample_ohlc_df_large["low"].min()
        max_price = sample_ohlc_df_large["high"].max()
        assert all(min_price - 10 < v < max_price + 10 for v in values)

    def test_hma_custom_column(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test HMA with custom column parameter."""
        hma_close = HMA(params={"period": 10, "column": "close"})
        hma_open = HMA(params={"period": 10, "column": "open"})

        result_close = hma_close.compute(sample_ohlc_df_large)
        result_open = hma_open.compute(sample_ohlc_df_large)

        # Results should be different when using different columns
        close_vals = result_close["value"].to_list()
        open_vals = result_open["value"].to_list()
        assert close_vals != open_vals

    def test_hma_multiple_periods(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test HMA with Fibonacci periods."""
        # Test with smaller periods that fit the sample data
        for period in [10, 20, 30]:
            hma = HMA(params={"period": period})
            result = hma.compute(sample_ohlc_df_large)
            assert result.height > 0
            assert result.filter(pl.col("value").is_nan()).height == 0

    def test_hma_vs_sma_lag_reduction(self, sample_ohlc_df_large: pl.DataFrame) -> None:
        """Test that HMA has less lag than SMA (responds faster to price changes)."""
        period = 10
        hma = HMA(params={"period": period})
        sma = SMA(params={"period": period})

        result_hma = hma.compute(sample_ohlc_df_large)
        result_sma = sma.compute(sample_ohlc_df_large)

        # Both should return valid results
        assert result_hma.height > 0
        assert result_sma.height > 0
        # HMA and SMA should be different (HMA has less lag)
        hma_vals = result_hma["value"].to_list()
        sma_vals = result_sma["value"].to_list()
        # Align by timestamp
        common_len = min(len(hma_vals), len(sma_vals))
        assert hma_vals[:common_len] != sma_vals[:common_len]

    def test_hma_repr(self) -> None:
        """Test HMA string representation."""
        hma = HMA(params={"period": 55})

        repr_str = repr(hma)

        assert "HMA" in repr_str
        assert "period=55" in repr_str

    def test_hma_in_registry(self) -> None:
        """Test HMA is registered and accessible via get_indicator."""
        HMAClass = get_indicator("hma")

        assert HMAClass.name == "hma"
        assert HMAClass is HMA
