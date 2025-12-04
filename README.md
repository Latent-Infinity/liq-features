# liq-features

A component of the Latent Infinity Quant (LIQ) ecosystem, `liq-features` transforms raw market data into ML-ready features. It computes technical indicators, implements the midrange-first philosophy, and provides a feature store API.

## Installation

```bash
pip install liq-features
```

## CLI Usage

The `liq-features` CLI integrates with `liq-data` for symbol-based data access. All commands use provider/symbol/timeframe to identify data - no file paths needed.

### Storage Pattern

- Raw OHLCV data: `{data_root}/{provider}/{symbol}/{timeframe}.parquet`
- Computed features: `{data_root}/{provider}/{symbol}/{timeframe}_features.parquet`

All computed features are stored in a single unified features file per timeframe, allowing you to accumulate features over time.

### Commands

#### Compute Derived Fields

Compute derived OHLC fields (midrange, range, true_range, etc.):

```bash
# Preview results (default)
liq-features derived EUR_USD --provider oanda --timeframe 1m

# Save to features file
liq-features derived EUR_USD --provider oanda --timeframe 1m --save
```

#### Compute Returns

```bash
# Simple returns
liq-features returns EUR_USD --provider oanda --save

# Log returns with custom periods
liq-features returns EUR_USD --provider oanda --periods 5 --log --save
```

#### Compute Volatility

```bash
liq-features volatility EUR_USD --provider oanda --window 20 --save
```

#### Technical Indicators

```bash
# Compute RSI
liq-features indicator RSI EUR_USD --provider oanda --period 14 --save

# List available indicators
liq-features list-indicators
```

#### Aggregate to Higher Timeframe

```bash
# Aggregate 1m bars to 1h
liq-features aggregate EUR_USD --provider oanda --source 1m --target 1h --save
```

#### List Stored Features

```bash
liq-features list-stored oanda EUR_USD --timeframe 1m
```

Output:
```
Stored features for oanda/EUR_USD/1m

                   Stored Feature Columns
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Column              ┃ Type    ┃  Non-null ┃ Sample Value ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ midrange            │ Float64 │ 7,690,495 │      1.16280 │
│ range               │ Float64 │ 7,690,495 │      0.00004 │
│ true_range          │ Float64 │ 7,690,495 │      0.00004 │
│ return              │ Float64 │ 7,690,494 │      0.00003 │
└─────────────────────┴─────────┴───────────┴──────────────┘
```

#### Show Symbol Info

```bash
liq-features info EUR_USD --provider oanda --timeframe 1m
```

#### Pipeline Fit/Transform (stationarity + scaling)

Train-only fit persisted to JSON, then reused without refit:

```bash
# Fit pipeline
python -m liq.features.cli fit-pipeline series.parquet --model-type nn --d 0.3 --output pipeline.json

# Apply pipeline
python -m liq.features.cli transform series.parquet pipeline.json --output transformed.json
```

## Python API

```python
from liq.features.derived import compute_derived_fields, compute_returns
from liq.data.settings import load_symbol_data

# Load data via liq-data
df = load_symbol_data("oanda", "EUR_USD", "1m")

# Compute derived fields
result = compute_derived_fields(df)
print(result.columns)
# ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'midrange', 'range', 'true_range']

# Compute returns
result = compute_returns(df, column="close", periods=1)
```

## Design Principles

- **Separation of Concerns**: liq-features uses liq-data for data access, no direct file path handling
- **Symbol-based Access**: All commands use provider/symbol/timeframe, abstracting storage details
- **Unified Feature Storage**: All computed features for a symbol/timeframe are stored in a single parquet file
- **Incremental Updates**: New features are merged with existing ones, allowing gradual feature accumulation
- **Preview by Default**: Commands preview results without saving; use `--save` to persist
