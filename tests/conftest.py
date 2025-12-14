"""Pytest configuration and shared fixtures for liq-features tests."""

from datetime import UTC, datetime

import polars as pl
import pytest


@pytest.fixture
def sample_ohlc_df() -> pl.DataFrame:
    """Create a sample OHLC DataFrame for testing."""
    return pl.DataFrame({
        "ts": [
            datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
        ],
        "open": [100.0, 102.0, 101.0, 103.0, 102.0],
        "high": [103.0, 104.0, 103.0, 105.0, 104.0],
        "low": [99.0, 101.0, 100.0, 102.0, 101.0],
        "close": [102.0, 101.0, 103.0, 102.0, 103.0],
        "volume": [1000.0, 1500.0, 1200.0, 1800.0, 1400.0],
    })


@pytest.fixture
def sample_ohlc_df_large() -> pl.DataFrame:
    """Create a larger OHLC DataFrame for indicator testing (needs warmup period)."""
    import math

    n_rows = 100
    base_price = 100.0

    timestamps = [
        datetime(2024, 1, 1, i // 24, i % 24, 0, tzinfo=UTC) for i in range(n_rows)
    ]

    # Generate price series with some trend and noise
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = base_price
    for i in range(n_rows):
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

    return pl.DataFrame({
        "ts": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


@pytest.fixture
def sample_minute_df() -> pl.DataFrame:
    """Create sample minute-level data for aggregation testing."""
    timestamps = [
        datetime(2024, 1, 15, 10, i, 0, tzinfo=UTC) for i in range(60)
    ]

    base_price = 100.0
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = base_price
    for i in range(60):
        open_price = price
        change = (i % 5 - 2) * 0.1
        close_price = price + change
        high_price = max(open_price, close_price) + 0.05
        low_price = min(open_price, close_price) - 0.05

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(100 + i)

        price = close_price

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
