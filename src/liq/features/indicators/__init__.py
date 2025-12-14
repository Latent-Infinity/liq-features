"""Technical indicators for quantitative analysis.

This package provides a framework for computing and caching technical indicators
with efficient storage using fixed-point quantization.

Example:
    >>> from liq.features.indicators import get_indicator, list_indicators
    >>>
    >>> # Get an indicator class
    >>> RSI = get_indicator("rsi")
    >>> rsi = RSI(params={"period": 14})
    >>> result = rsi.compute(df)
    >>>
    >>> # List available indicators
    >>> indicators = list_indicators()
"""

# Import hardcoded indicator modules to trigger @register_indicator decorators
# This ensures all indicators are available via get_indicator() without explicit imports
from liq.features.indicators import (
    momentum,  # noqa: F401
    trend,  # noqa: F401
)
from liq.features.indicators.base import BaseIndicator
from liq.features.indicators.registry import (
    configure_defaults,
    get_indicator,
    list_indicators,
    reset_defaults,
)
from liq.features.indicators.zigzag import zigzag_pivots

__all__ = [
    "BaseIndicator",
    "get_indicator",
    "list_indicators",
    "configure_defaults",
    "reset_defaults",
    "zigzag_pivots",
]
