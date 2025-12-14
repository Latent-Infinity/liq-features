# liq-features

A component of the Latent Infinity Quant (LIQ) ecosystem, `liq-features` transforms raw market data into ML-ready features. It computes technical indicators, implements the midrange-first philosophy, and provides a feature store API that caches through `liq-store` (ParquetStore by default, or any `TimeSeriesStore`).

## Installation

```bash
pip install liq-features
```

Optional TA-Lib support (150+ additional indicators):
```bash
pip install ta-lib
```

## Quick Start

```python
from liq.features import (
    compute_derived_fields,
    compute_indicators,
    aggregate_to_timeframe,
    configure_defaults,
)
from liq.features.indicators import get_indicator, list_indicators
from liq.store.parquet import ParquetStore

# Compute derived fields (midrange, range, true_range, etc.)
df = compute_derived_fields(ohlcv_data)

# Get a single indicator
RSI = get_indicator("rsi")
rsi = RSI(params={"period": 14})
result = rsi.compute(df, symbol="EUR_USD", timeframe="1m")

# Batch compute multiple indicators
storage = ParquetStore("/data/features")
features = compute_indicators(
    bars=df,
    symbol="EUR_USD",
    timeframe="1h",
    indicators=[
        ("rsi", {"period": 14}),
        ("macd", {}),
        ("atr", {"period": 14}),
    ],
    storage=storage,
)

# Aggregate timeframes (excludes incomplete bars by default)
hourly = aggregate_to_timeframe(minute_bars, "1m", "1h")
```

## Technical Indicators

### Available Indicators

liq-features provides 160+ technical indicators:

**Hardcoded (optimized):** RSI, MACD, Stochastic, EMA, SMA, BBANDS, ADX, ATR, ROC, WMA, plus midrange variants

**Dynamic (via TA-Lib):** All 150+ TA-Lib indicators available through `get_indicator()`

```python
from liq.features.indicators import get_indicator, list_indicators

# List all available indicators
indicators = list_indicators()
for ind in indicators[:5]:
    print(f"{ind['name']}: {ind['source']}")

# Get any indicator by name
CCI = get_indicator("cci")  # TA-Lib indicator
cci = CCI(params={"timeperiod": 20})
result = cci.compute(df)
```

### Global Parameter Configuration

Override default parameters globally:

```python
from liq.features import configure_defaults, reset_defaults

# Change RSI default period from 14 to 21
configure_defaults({"rsi": {"period": 21}})

# All new RSI instances now use period=21
RSI = get_indicator("rsi")
rsi = RSI()  # period=21

# Explicit params still override configured defaults
rsi_custom = RSI(params={"period": 7})  # period=7

# Reset all indicators to original defaults
reset_defaults()
```

## Aggregation

Aggregate OHLCV data to larger timeframes:

```python
from liq.features import Aggregator, aggregate_to_timeframe

# Using the class
agg = Aggregator(source_timeframe="1m", target_timeframe="1h")
hourly = agg.aggregate(minute_bars)

# Using the convenience function
hourly = aggregate_to_timeframe(minute_bars, "1m", "1h")

# Include incomplete final bar (default is False for backtesting)
hourly_all = aggregate_to_timeframe(
    minute_bars, "1m", "1h", include_incomplete=True
)
```

Supported timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`

## Cache Statistics

Inspect cached indicator computations:

```python
from liq.features import cache_stats
from liq.store.parquet import ParquetStore

storage = ParquetStore("/data/features")
stats = cache_stats(storage)
print(stats)
# shape: (3, 5)
# ┌───────────┬───────────┬──────────────┬───────────┬─────────┐
# │ indicator ┆ timeframe ┆ params_id    ┆ row_count ┆ size_mb │
# ├───────────┼───────────┼──────────────┼───────────┼─────────┤
# │ rsi       ┆ 1m        ┆ a1b2c3d4     ┆ 100000    ┆ 1.2     │
# │ macd      ┆ 1h        ┆ e5f6g7h8     ┆ 5000      ┆ 0.3     │
# └───────────┴───────────┴──────────────┴───────────┴─────────┘
```

## CLI Usage

```bash
# Compute features from bars file and store via liq-store
python -m liq.features.cli compute path/to/bars.parquet \
    --symbol EUR_USD --timeframe 1m \
    --store-root /data/features --feature-name derived_midrange

# Load bars via liq-data provider instead of file
python -m liq.features.cli compute \
    --provider oanda --symbol EUR_USD --timeframe 1m \
    --store-root /data/features --feature-name derived_midrange

# Inspect/clear indicator cache
python -m liq.features.cli indicator-cache --cache-dir /tmp/cache
python -m liq.features.cli indicator-cache --cache-dir /tmp/cache --clear

# Validate feature set definition (JSON)
python -m liq.features.cli validate path/to/feature_set.json

# Fit/transform pipelines (stationarity + scaling)
python -m liq.features.cli fit-pipeline series.parquet --model-type nn --d 0.3 --output pipeline.json
python -m liq.features.cli transform series.parquet pipeline.json --output transformed.json
```

## FeatureStore (Dependency-aware)

Define feature sets with dependencies that are resolved topologically:

```python
import polars as pl
from liq.features.feature_set import FeatureDefinition, FeatureSet
from liq.features.store import FeatureStore
from liq.store.parquet import ParquetStore

df = pl.DataFrame({
    "timestamp": [...],
    "open": [...], "high": [...], "low": [...], "close": [...]
})

def mid(df_: pl.DataFrame, _: str) -> pl.DataFrame:
    return df_.select("timestamp", ((pl.col("high") + pl.col("low")) / 2).alias("value"))

def mid_plus_one(df_: pl.DataFrame, _: str) -> pl.DataFrame:
    return df_.select("timestamp", (pl.col("mid") + 1).alias("value"))

fs_def = FeatureSet(
    name="example",
    features=[
        FeatureDefinition("mid", mid, dependencies=[]),
        FeatureDefinition("mid_plus_one", mid_plus_one, dependencies=["mid"]),
    ],
)

store = ParquetStore("/data/features")
feature_store = FeatureStore(storage=store)
feature_store.compute_feature_set(df, fs_def, symbol="EUR_USD", timeframe="1m")
```

Features:
- Topological dependency resolution
- Automatic warmup row trimming (max lookback)
- Cache invalidation via data hash
- Storage keys: `symbol/indicators/name/params:timeframe` and `symbol/features/set:timeframe`

## Derived Fields

The midrange-first philosophy provides these derived fields:

```python
from liq.features import compute_derived_fields

df = compute_derived_fields(ohlcv_data)
# Adds columns: midrange, range, true_range, true_range_midrange, true_range_hl
```

| Field | Formula |
|-------|---------|
| `midrange` | (high + low) / 2 |
| `range` | high - low |
| `true_range` | max(high - low, abs(high - prev_close), abs(low - prev_close)) |
| `true_range_midrange` | max(midrange - prev_midrange, prev_midrange - midrange) |
| `true_range_hl` | max(high - prev_low, prev_high - low) |

## API Reference

### Top-level exports

```python
from liq.features import (
    # Derived features
    compute_derived_fields,
    compute_returns,
    compute_volatility,
    # Aggregation
    Aggregator,
    aggregate_to_timeframe,
    # Batch computation
    compute_indicators,
    cache_stats,
    # Configuration
    configure_defaults,
    reset_defaults,
    # Feature sets
    FeatureDefinition,
    FeatureSet,
    FeatureStore,
    # Params
    normalize_params,
    hash_params,
    format_params_key,
    # Quantization
    INDICATOR_SCALES,
    get_indicator_scale,
    quantize_to_int,
    dequantize_from_int,
)
```

### Indicator exports

```python
from liq.features.indicators import (
    BaseIndicator,
    get_indicator,
    list_indicators,
    configure_defaults,
    reset_defaults,
)
```
