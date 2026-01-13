"""Feature engineering and technical indicators for the LIQ stack.

This package provides:
- Derived OHLC features (midrange, range, true_range, etc.)
- Time series aggregation (e.g., 1min -> 1h)
- Technical indicators (RSI, MACD, EMA, ATR, etc.)
- Parameter hashing for indicator storage
- Fixed-point quantization for efficient storage
- Batch indicator computation
- Global parameter configuration

Example:
    >>> import polars as pl
    >>> from liq.features import compute_derived_fields, compute_indicators
    >>> from liq.features.aggregation import aggregate_to_timeframe
    >>> from liq.features.indicators import get_indicator
    >>>
    >>> # Compute derived fields
    >>> df = compute_derived_fields(ohlc_data)
    >>>
    >>> # Aggregate timeframes
    >>> hourly = aggregate_to_timeframe(minute_data, "1min", "1h")
    >>>
    >>> # Compute single indicator
    >>> RSI = get_indicator("rsi")
    >>> rsi = RSI(params={"period": 14})
    >>> result = rsi.compute(df)
    >>>
    >>> # Batch compute multiple indicators
    >>> features = compute_indicators(
    ...     bars=df,
    ...     symbol="EUR_USD",
    ...     timeframe="1h",
    ...     indicators=[("rsi", {"period": 14}), ("macd", {})],
    ...     storage=storage,
    ... )
"""

from liq.features.aggregation import Aggregator, aggregate_to_timeframe
from liq.features.batch import cache_stats, compute_indicators
from liq.features.cache import CacheManager, IndicatorCache
from liq.features.cache_exceptions import (
    CacheCleanupError,
    CacheConfigurationError,
    CacheCorruptionError,
    CacheError,
    CacheNotFoundError,
)
from liq.features.cache_models import (
    CacheEntry,
    CacheFilter,
    CacheStats,
    CleanupCriteria,
    CleanupResult,
)
from liq.features.derived import (
    DEFAULT_FIBONACCI_WINDOWS,
    compute_derived_fields,
    compute_multi_window_volatility,
    compute_returns,
    compute_rolling_returns,
    compute_volatility,
)
from liq.features.feature_set import FeatureDefinition, FeatureSet
from liq.features.indicators import configure_defaults, reset_defaults
from liq.features.params import format_params_key, hash_params, normalize_params
from liq.features.quantization import (
    INDICATOR_SCALES,
    dequantize_from_int,
    get_indicator_scale,
    quantize_to_int,
)
from liq.features.store import FeatureStore

# Selection utilities (sklearn/scipy wrappers)
from liq.features.selection import mrmr_select, mutual_info_scores, spearman_matrix

__all__ = [
    # Cache management
    "CacheManager",
    "IndicatorCache",
    "CacheEntry",
    "CacheFilter",
    "CacheStats",
    "CleanupCriteria",
    "CleanupResult",
    # Cache exceptions
    "CacheError",
    "CacheNotFoundError",
    "CacheCorruptionError",
    "CacheCleanupError",
    "CacheConfigurationError",
    # Derived features
    "compute_derived_fields",
    "compute_returns",
    "compute_volatility",
    "compute_rolling_returns",
    "compute_multi_window_volatility",
    "DEFAULT_FIBONACCI_WINDOWS",
    # Aggregation
    "Aggregator",
    "aggregate_to_timeframe",
    # Batch computation
    "compute_indicators",
    "cache_stats",
    # Configuration
    "configure_defaults",
    "reset_defaults",
    # Feature sets
    "FeatureDefinition",
    "FeatureSet",
    "FeatureStore",
    # Params
    "normalize_params",
    "hash_params",
    "format_params_key",
    # Quantization
    "INDICATOR_SCALES",
    "get_indicator_scale",
    "quantize_to_int",
    "dequantize_from_int",
    # Selection
    "mutual_info_scores",
    "mrmr_select",
    "spearman_matrix",
]
