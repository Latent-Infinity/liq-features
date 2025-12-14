"""Tests for liq.features.quantization module."""

import polars as pl

from liq.features.quantization import (
    INDICATOR_SCALES,
    dequantize_from_int,
    get_indicator_scale,
    quantize_to_int,
)


class TestGetIndicatorScale:
    """Tests for get_indicator_scale function."""

    def test_known_indicator(self) -> None:
        """Test known indicator returns correct scale."""
        scale, dtype, min_val, max_val = get_indicator_scale("rsi")

        assert scale == 100
        assert dtype == "int16"

    def test_unknown_indicator_uses_default(self) -> None:
        """Test unknown indicator returns default scale."""
        scale, dtype, min_val, max_val = get_indicator_scale("unknown_indicator")

        assert scale == 100_000
        assert dtype == "int64"

    def test_all_registered_indicators(self) -> None:
        """Test all registered indicators have valid scales."""
        for name in INDICATOR_SCALES:
            scale, dtype, min_val, max_val = get_indicator_scale(name)

            assert scale > 0
            assert dtype in ("int16", "int32", "int64")
            assert min_val < max_val


class TestQuantizeToInt:
    """Tests for quantize_to_int function."""

    def test_basic_quantization(self) -> None:
        """Test basic float to int quantization."""
        values = pl.Series("rsi", [50.0, 65.5, 30.25])

        quantized, scale = quantize_to_int(values, "rsi")

        assert scale == 100
        assert quantized[0] == 5000
        assert quantized[1] == 6550
        assert quantized[2] == 3025

    def test_custom_scale(self) -> None:
        """Test custom scale override."""
        values = pl.Series("test", [1.0, 2.5])

        quantized, scale = quantize_to_int(values, "unknown", custom_scale=1000)

        assert scale == 1000
        assert quantized[0] == 1000
        assert quantized[1] == 2500

    def test_handles_infinity(self) -> None:
        """Test infinity values are clipped."""
        values = pl.Series("test", [float("inf"), float("-inf"), 50.0])

        quantized, scale = quantize_to_int(values, "rsi")

        # Should not contain infinity
        assert quantized[2] == 5000  # Normal value preserved
        # Infinity values should be clipped to valid range
        assert all(abs(v) < 2**53 for v in quantized.to_list())

    def test_returns_int64(self) -> None:
        """Test output is Int64."""
        values = pl.Series("test", [1.0, 2.0, 3.0])

        quantized, _ = quantize_to_int(values, "rsi")

        assert quantized.dtype == pl.Int64


class TestDequantizeFromInt:
    """Tests for dequantize_from_int function."""

    def test_basic_dequantization(self) -> None:
        """Test basic int to float dequantization."""
        values = pl.Series("rsi", [5000, 6550, 3025])

        dequantized = dequantize_from_int(values, "rsi")

        assert dequantized[0] == 50.0
        assert dequantized[1] == 65.5
        assert dequantized[2] == 30.25

    def test_custom_scale(self) -> None:
        """Test custom scale override."""
        values = pl.Series("test", [1000, 2500])

        dequantized = dequantize_from_int(values, "unknown", custom_scale=1000)

        assert dequantized[0] == 1.0
        assert dequantized[1] == 2.5

    def test_roundtrip(self) -> None:
        """Test quantize then dequantize preserves values."""
        original = pl.Series("test", [50.0, 65.5, 30.25, 99.99])

        quantized, scale = quantize_to_int(original, "rsi")
        recovered = dequantize_from_int(quantized, "rsi")

        # Should be approximately equal (within scale precision)
        for i in range(len(original)):
            assert abs(original[i] - recovered[i]) < 0.01

    def test_returns_float64(self) -> None:
        """Test output is Float64."""
        values = pl.Series("test", [1000, 2000, 3000])

        dequantized = dequantize_from_int(values, "rsi")

        assert dequantized.dtype == pl.Float64
