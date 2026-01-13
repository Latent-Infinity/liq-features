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

## Input DataFrame Schema

All functions expect Polars DataFrames with OHLCV data:

| Column | Type | Required |
|--------|------|----------|
| `ts` or `timestamp` | `datetime` (timezone-aware) | Yes |
| `open` | `float` | Yes |
| `high` | `float` | Yes |
| `low` | `float` | Yes |
| `close` | `float` | Yes |
| `volume` | `float` | No (some indicators require it) |

Example:
```python
import polars as pl
from datetime import datetime, timezone

ohlcv_data = pl.DataFrame({
    "ts": [
        datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 15, 10, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, 10, 2, tzinfo=timezone.utc),
    ],
    "open": [100.0, 102.0, 101.0],
    "high": [103.0, 104.0, 103.0],
    "low": [99.0, 101.0, 100.0],
    "close": [102.0, 101.0, 102.5],
    "volume": [1000.0, 1500.0, 1200.0],
})
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
# minute_bars: your 1-minute OHLCV DataFrame
hourly = aggregate_to_timeframe(minute_bars, "1m", "1h")
```

## Technical Indicators

### Available Indicators

liq-features provides 160+ technical indicators:

**Hardcoded (optimized):** RSI, MACD, Stochastic, EMA, SMA, BBANDS, ADX, ATR, ROC, WMA, HMA, plus midrange variants

**Volume indicators:** AbnormalTurnover (z-score), NormalizedVolume (ratio to MA)

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
result = cci.compute(df)  # symbol/timeframe optional (used for caching)
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

# Inspect/clear indicator cache (uses LIQ_DATA_ROOT for storage)
python -m liq.features.cli stats
python -m liq.features.cli query --symbol EUR_USD --timeframe 1m --indicator sar*
python -m liq.features.cli clear --all

# Rebuild index (useful after large parallel runs)
python -m liq.features.cli rebuild-index

# Lockless mode (default): disable index writes during runs, rebuild after
LIQ_FEATURES_INDEX=off python -m liq.experiments.indicator_ranking run ...
python -m liq.features.cli rebuild-index

# Pit-of-success pattern (recommended)
# - Run lockless (default) to avoid contention in parallel workers
# - Index rebuild is automatic at the end of indicator_ranking runs
# - For other pipelines, rebuild manually when needed
python -m liq.experiments.indicator_ranking run ...
python -m liq.features.cli rebuild-index

# Validate feature set definition (JSON)
python -m liq.features.cli validate path/to/feature_set.json

# Fit/transform pipelines (stationarity + scaling)
python -m liq.features.cli fit-pipeline series.parquet --model-type nn --d 0.3 --output pipeline.json
python -m liq.features.cli transform series.parquet pipeline.json --output transformed.json
```

### Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-type` | Target model type: `nn`, `transformer`, `diffusion`, `trees` | Required |
| `--d` | Fractional differencing degree (0-1). Lower = more memory, higher = more stationary | `0.4` |
| `--output` | Output path for pipeline JSON | Required |

Model types determine scaling:
- `nn`, `transformer`: Standardization (zero mean, unit variance)
- `diffusion`: Min-max scaling to [0, 1]
- `trees`: No scaling (tree-based models handle raw values)

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

## Rolling Returns & Volatility

Compute rolling aggregations over Fibonacci windows (55, 210, 340, 890, 3750):

```python
from liq.features import (
    compute_rolling_returns,
    compute_multi_window_volatility,
    DEFAULT_FIBONACCI_WINDOWS,
)

# Rolling sum/mean of log returns across multiple windows
df = compute_rolling_returns(ohlcv_data, windows=[55, 210, 340])
# Adds: log_return_sum_55, log_return_mean_55, log_return_sum_210, ...

# Multi-window volatility (annualized by default)
df = compute_multi_window_volatility(ohlcv_data, windows=[55, 210])
# Adds: volatility_55, volatility_210
```

## Triple-Barrier Labels

Generate ML training labels using the triple-barrier method:

```python
from liq.features.labels import TripleBarrierConfig, triple_barrier_labels_adaptive

# Fixed thresholds (2% profit, 1% loss)
cfg = TripleBarrierConfig(
    take_profit=0.02,
    stop_loss=0.01,
    max_holding=5,
)

# Adaptive thresholds (2σ profit/loss based on rolling volatility)
cfg = TripleBarrierConfig(
    profit_std_multiple=2.0,
    loss_std_multiple=2.0,
    volatility_window=20,
    max_holding=5,
)

result = triple_barrier_labels_adaptive(df, cfg)
# Adds 'label' column: 1 (profit), -1 (loss), 0 (timeout)
```

## Return Types

All functions return Polars DataFrames:

| Function | Returns |
|----------|---------|
| `compute_derived_fields()` | Input DataFrame + derived columns |
| `compute_rolling_returns()` | Input DataFrame + rolling sum/mean columns per window |
| `compute_multi_window_volatility()` | Input DataFrame + volatility columns per window |
| `compute_indicators()` | DataFrame with `ts` + all indicator columns merged |
| `aggregate_to_timeframe()` | Aggregated OHLCV DataFrame |
| `Indicator.compute()` | DataFrame with `ts`, `value` (or named outputs for multi-output indicators) |
| `triple_barrier_labels_adaptive()` | Input DataFrame + `label` column (-1, 0, 1) |
| `FeatureStore.compute_feature_set()` | `dict[str, pl.DataFrame]` mapping feature names to DataFrames |

Multi-output indicators return named columns instead of `value`:
```python
MACD = get_indicator("macd")
macd = MACD()
result = macd.compute(df)  # Returns: ts, macd, signal, histogram
```

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

## Error Handling

The library raises standard Python exceptions:

**ValueError:**
- Missing required OHLC columns in input DataFrame
- Unknown indicator name passed to `get_indicator()`
- Invalid or incompatible timeframes for aggregation
- Circular or missing feature dependencies in FeatureSet
- Insufficient data for computations

**RuntimeError:**
- Pipeline not fitted before calling `transform()`
- Attempting to serialize an unfitted pipeline

Example:
```python
import polars as pl
from liq.features import compute_derived_fields
from liq.features.indicators import get_indicator

# Missing columns raises ValueError
try:
    df = pl.DataFrame({"open": [1.0], "close": [1.0]})  # Missing high, low
    compute_derived_fields(df)
except ValueError as e:
    print(f"Missing columns: {e}")

# Unknown indicator raises ValueError
try:
    get_indicator("not_an_indicator")
except ValueError as e:
    print(f"Unknown indicator: {e}")
```

## Performance Considerations

**Memory:**
- All operations are eager Polars transformations (no lazy evaluation)
- Data is assumed to fit in memory; no automatic chunking
- Typical for intraday/daily bar data

**Caching:**
- Results are cached to storage with keys like `symbol/indicators/name/params:timeframe`
- Cache invalidation uses data hashing via `get_data_hash()`
- Use `force_recalculate=False` (default) to leverage cache

**Optimization tips:**
- Pre-compute indicators at the lowest timeframe, then aggregate
- Batch compute with `compute_indicators()` to share timestamp joins
- Use quantization (`quantize_to_int()`) for efficient storage

## Thread Safety

**Safe for concurrent use:**
- Indicator `compute()` methods (pure functions, no shared state)
- `compute_derived_fields()` (pure Polars transformations)
- Storage reads (ParquetStore supports concurrent reads)

**Not thread-safe:**
- `configure_defaults()` / `reset_defaults()` (modifies global state)
- Concurrent writes to the same storage key (use application-level coordination)

**Parallel processing:**
- `mutual_info_scores()` uses `ProcessPoolExecutor` for parallel computation

## Dependencies

**Required:**
- `polars>=1.20` - Primary DataFrame library (no pandas fallback)
- `numpy>=2.0` - Numerical operations
- `liq-store>=0.1.0` - Storage backend (LIQ ecosystem)
- `scikit-learn>=1.4` - Feature selection

**Optional:**
- `ta-lib>=0.4.0` - 150+ additional indicators
- `xxhash>=3.0.0` - Fast hashing for cache keys
- `mrmr-selection>=0.2.8` - mRMR feature selection

**Install variants:**
```bash
pip install liq-features              # Core only
pip install liq-features[talib]       # + TA-Lib indicators
pip install liq-features[all]         # All optional dependencies
```
