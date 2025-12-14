"""Fixed-point quantization for indicator values.

This module provides efficient storage of indicator values by converting floats
to fixed-point integers. This approach provides 2-6x better compression with Zstandard
compared to storing float64 values.

Design Principles:
    - SRP: Only handles numeric quantization/dequantization
    - KISS: Simple conversion functions with clear semantics
    - DIP: Works with Polars Series abstractions
"""

import polars as pl

# Indicator scale registry (scale, dtype, min_val, max_val)
# Scale determines precision: 100 = 2 decimals, 1000 = 3 decimals, etc.
INDICATOR_SCALES: dict[str, tuple[int, str, int, int]] = {
    # Bounded percentages [0-100]
    "rsi": (100, "int16", -32768, 32767),
    "mfi": (100, "int16", 0, 32767),
    "stoch_k": (100, "int16", 0, 32767),
    "stoch_d": (100, "int16", 0, 32767),
    # Percent returns / oscillators
    "roc": (1_000_000, "int32", -2147483648, 2147483647),
    "ret": (1_000_000, "int32", -2147483648, 2147483647),
    # Moving averages / price-like
    "ema": (100_000, "int64", -9223372036854775808, 9223372036854775807),
    "sma": (100_000, "int64", -9223372036854775808, 9223372036854775807),
    "wma": (100_000, "int64", -9223372036854775808, 9223372036854775807),
    "vwap": (100_000, "int64", -9223372036854775808, 9223372036854775807),
    # MACD (price-like oscillator)
    "macd": (100_000, "int64", -9223372036854775808, 9223372036854775807),
    # Bollinger Bands (price-like)
    "bbands": (100_000, "int64", -9223372036854775808, 9223372036854775807),
    # ATR (price difference)
    "atr": (100_000, "int32", 0, 2147483647),
    # Volume indicators (already integers)
    "obv": (1, "int64", -9223372036854775808, 9223372036854775807),
}

_DEFAULT_SCALE = (100_000, "int64", -9223372036854775808, 9223372036854775807)


def get_indicator_scale(indicator: str) -> tuple[int, str, int, int]:
    """Get scale configuration for an indicator.

    Args:
        indicator: Indicator name (e.g., "rsi", "ema", "macd")

    Returns:
        Tuple of (scale, dtype, min_val, max_val)
        - scale: Multiplier for fixed-point conversion
        - dtype: Target integer dtype ("int16", "int32", "int64")
        - min_val: Minimum value for dtype
        - max_val: Maximum value for dtype

    Examples:
        >>> get_indicator_scale("rsi")
        (100, 'int16', -32768, 32767)

        >>> get_indicator_scale("ema")
        (100000, 'int64', -9223372036854775808, 9223372036854775807)
    """
    return INDICATOR_SCALES.get(indicator, _DEFAULT_SCALE)


def quantize_to_int(
    values: pl.Series, indicator: str, custom_scale: int | None = None
) -> tuple[pl.Series, int]:
    """Convert float values to fixed-point integers for storage.

    This function multiplies floats by a scale factor, rounds, and converts to int64.
    Values are clipped to the target dtype range to prevent overflow.

    Args:
        values: Polars Series of float values
        indicator: Indicator name (used to look up default scale)
        custom_scale: Optional scale override (default: uses INDICATOR_SCALES)

    Returns:
        Tuple of (quantized_series, scale_used)
        - quantized_series: Int64 Series with quantized values
        - scale_used: Scale factor that was applied

    Examples:
        >>> import polars as pl
        >>> values = pl.Series("rsi", [50.0, 65.5, 30.25])
        >>> quantized, scale = quantize_to_int(values, "rsi")
        >>> print(quantized.to_list())
        [5000, 6550, 3025]
        >>> print(scale)
        100
    """
    scale, _, min_val, max_val = get_indicator_scale(indicator)

    if custom_scale is not None:
        scale = custom_scale
        min_val = -9223372036854775808
        max_val = 9223372036854775807

    # Float64 precision limit
    max_safe_int = 2**53 - 1
    if max_val > max_safe_int:
        safe_max = max_safe_int
        safe_min = -max_safe_int
    else:
        safe_max = max_val
        safe_min = min_val

    # Clip input to prevent overflow
    max_safe_input = safe_max / scale
    min_safe_input = safe_min / scale

    # Handle infinities
    clipped_input = values.replace(float("inf"), max_safe_input)
    clipped_input = clipped_input.replace(float("-inf"), min_safe_input)
    clipped_input = clipped_input.clip(min_safe_input, max_safe_input)

    # Quantize
    quantized = (clipped_input * scale).round(0).cast(pl.Int64, strict=False)

    return quantized, scale


def dequantize_from_int(
    values: pl.Series, indicator: str, custom_scale: int | None = None
) -> pl.Series:
    """Convert fixed-point integers back to floats.

    This function divides integers by the scale factor to recover the original
    float values (within precision limits).

    Args:
        values: Polars Series of integer values
        indicator: Indicator name (used to look up default scale)
        custom_scale: Optional scale override (default: uses INDICATOR_SCALES)

    Returns:
        Float64 Series with dequantized values

    Examples:
        >>> import polars as pl
        >>> quantized = pl.Series("rsi", [5000, 6550, 3025])
        >>> dequantized = dequantize_from_int(quantized, "rsi")
        >>> print(dequantized.to_list())
        [50.0, 65.5, 30.25]
    """
    scale, _, _, _ = get_indicator_scale(indicator)

    if custom_scale is not None:
        scale = custom_scale

    return values.cast(pl.Float64) / scale
