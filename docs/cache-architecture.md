# Indicator Cache Architecture

This document describes the indicator caching system used to avoid redundant computations during feature engineering and indicator ranking experiments.

## Overview

The indicator cache stores computed indicator values (time series) keyed by a combination of factors that uniquely identify the computation:

- **Symbol** (e.g., `BTC_USDT`, `EUR_USD`)
- **Indicator name** (e.g., `rsi`, `macd`, `sarext`)
- **Parameters** (e.g., `period=14`, `fastperiod=12`)
- **Timeframe** (e.g., `1m`, `1h`, `1d`)
- **Data hash** (hash of input DataFrame contents)

## Cache Key Format

```
{symbol}/indicators/{indicator}/{params_hash}:{timeframe}:{data_hash}
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| `symbol` | Trading pair or instrument | `BTC_USDT` |
| `indicator` | Lowercase indicator name | `sarext` |
| `params_hash` | xxhash64 of sorted parameter dict | `002f3436cc049eaa` |
| `timeframe` | Data timeframe | `1m`, `15m`, `1h` |
| `data_hash` | xxhash64 of input DataFrame bytes | `77d4f640ecc5ac07` |

### Example Keys

```
BTC_USDT/indicators/rsi/a1b2c3d4e5f6:1h:9876543210abcdef
EUR_USD/indicators/macd/f1e2d3c4b5a6:15m:fedcba0987654321
```

## Storage Location

The cache uses `liq-store`'s `ParquetStore` backend. The data root is determined by:

1. `LIQ_DATA_ROOT` environment variable (if set)
2. Default: `./data` relative to working directory

### Directory Structure

```
{data_root}/
├── {symbol}/
│   └── indicators/
│       └── {indicator}/
│           └── {params_hash}:{timeframe}:{data_hash}/
│               └── data.parquet
```

### Example

```
data/
├── BTC_USDT/
│   └── indicators/
│       ├── rsi/
│       │   ├── a1b2c3d4:1h:abcd1234/
│       │   │   └── data.parquet
│       │   └── a1b2c3d4:1m:efgh5678/
│       │       └── data.parquet
│       └── sarext/
│           └── 002f3436:1m:77d4f640/
│               └── data.parquet
└── EUR_USD/
    └── indicators/
        └── ...
```

## Cached Data Format

Each `data.parquet` file contains:

| Column | Type | Description |
|--------|------|-------------|
| `ts` | `datetime[μs, UTC]` | Timestamp |
| `value` | `f64` | Indicator value |

For multi-output indicators (e.g., MACD with signal and histogram), separate cache entries are created for each output.

## Cache Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                      Indicator Computation                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Compute cache key                                            │
│     - Hash input DataFrame → data_hash                           │
│     - Hash parameters → params_hash                              │
│     - Build key: {symbol}/indicators/{ind}/{params}:{tf}:{data} │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Cache lookup                                                 │
│     - Check if key exists in storage                             │
│     - If hit: return cached DataFrame                            │
│     - If miss: proceed to computation                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (cache miss)
┌─────────────────────────────────────────────────────────────────┐
│  3. Compute indicator                                            │
│     - Call indicator function with parameters                    │
│     - Validate output shape and types                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Store result                                                 │
│     - Write DataFrame to parquet at cache key path               │
│     - Update index (if enabled)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Return result                                                │
│     - Return computed DataFrame to caller                        │
└─────────────────────────────────────────────────────────────────┘
```

## Cache Invalidation

The cache is invalidated automatically when:

1. **Data changes**: Different input data produces different `data_hash`
2. **Parameters change**: Different parameters produce different `params_hash`
3. **Manual deletion**: Cache entries can be deleted via CLI or API

There is no automatic TTL or expiration. Cache entries persist until explicitly deleted.

## Index

An optional index tracks all cache entries for efficient querying:

**Location**: `{data_root}/cache/indicators/index/data.parquet`

**Schema**:

| Column | Type | Description |
|--------|------|-------------|
| `key` | `Utf8` | Full cache key |
| `symbol` | `Utf8` | Symbol extracted from key |
| `timeframe` | `Utf8` | Timeframe extracted from key |
| `indicator` | `Utf8` | Indicator name |
| `params_hash` | `Utf8` | Parameter hash |
| `data_hash` | `Utf8` | Data hash |

**Environment Variable**: `LIQ_FEATURES_INDEX=on` enables indexing (default: off)

## Data Flow Diagram

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Price Data     │     │  Indicator Spec  │     │   Cache Config   │
│  (OHLCV bars)    │     │  (name, params)  │     │   (data_root)    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────┬───────────┘                        │
                      │                                    │
                      ▼                                    │
         ┌────────────────────────┐                        │
         │   compute_cache_key()  │◄───────────────────────┘
         │   - get_data_hash(df)  │
         │   - hash_params(params)│
         └───────────┬────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │   IndicatorCache       │
         │   .has(key)            │
         └───────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼ (hit)                   ▼ (miss)
┌───────────────┐       ┌─────────────────────┐
│ cache.get(key)│       │ compute_indicator() │
│ return cached │       │ cache.set(key, df)  │
└───────┬───────┘       └──────────┬──────────┘
        │                          │
        └──────────┬───────────────┘
                   │
                   ▼
         ┌────────────────────────┐
         │   Indicator DataFrame  │
         │   (ts, value)          │
         └────────────────────────┘
                   │
                   ▼
         ┌────────────────────────┐
         │   MI Computation       │
         │   (mutual information) │
         └────────────────────────┘
                   │
                   ▼
         ┌────────────────────────┐
         │   Results              │
         │   (mi_scores.parquet)  │
         └────────────────────────┘
```

## Storage Characteristics

### File Sizes

| Timeframe | Typical Rows | Compressed Size |
|-----------|--------------|-----------------|
| 1m (1 year) | ~525,000 | 5-6 MB |
| 15m (1 year) | ~35,000 | 400-500 KB |
| 1h (1 year) | ~8,760 | 100-150 KB |
| 1d (1 year) | ~365 | 10-20 KB |

### Compression

- **Current**: Snappy (fast, moderate compression)
- **Planned**: Zstd (slower, 2-3× better compression)

### Storage Explosion Factors

1. **Parameter combinations**: Some indicators have many valid parameter sets
   - `sarext`: 729 combinations (acceleration × maximum variations)
   - `stoch`: 4,374 combinations (period × matype × slowk/slowd periods)

2. **Data hash fragmentation**: Each unique date range creates new entries
   - Year-by-year analysis: 5 separate data_hashes
   - Different timeframes: separate data_hashes per timeframe

3. **No automatic cleanup**: Entries persist indefinitely

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `LIQ_DATA_ROOT` | Storage root directory | `./data` |
| `LIQ_FEATURES_INDEX` | Enable cache index | `off` |
| `LIQ_CACHE_COMPRESSION_LEVEL` | Zstd compression level (planned) | `3` |

## Related Modules

- `liq.features.cache` - `IndicatorCache` class and key computation
- `liq.features.params` - Parameter hashing (`hash_params`)
- `liq.store.parquet` - `ParquetStore` storage backend
- `liq.store.key_builder` - Key path construction

## See Also

- [Cache Management Guide](../../liq-experiments/docs/cache-management-guide.md) (planned)
- [Performance Guidelines](../../liq-docs/guides/performance-guidelines.md)
