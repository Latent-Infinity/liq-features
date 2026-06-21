"""Tests for batch indicator orchestration helpers."""

from datetime import date
from typing import Any, cast

import polars as pl

from liq.features import batch


class _FakeIndicator:
    def __init__(
        self, *, storage: object | None = None, params: dict[str, Any] | None = None
    ) -> None:
        self.storage = storage
        self.params = params or {}

    def compute(self, bars: pl.DataFrame, *, symbol: str, timeframe: str) -> pl.DataFrame:
        del symbol, timeframe
        multiplier = self.params.get("multiplier", 1)
        return bars.select("ts", (pl.col("close") * multiplier).alias("value"))


class _MultiOutputIndicator:
    def __init__(
        self, *, storage: object | None = None, params: dict[str, Any] | None = None
    ) -> None:
        self.storage = storage
        self.params = params or {}

    def compute(self, bars: pl.DataFrame, *, symbol: str, timeframe: str) -> pl.DataFrame:
        del symbol, timeframe
        return bars.select(
            "timestamp",
            pl.col("close").alias("fast"),
            (pl.col("close") * 2).alias("slow"),
        )


def test_compute_indicators_merges_value_and_multi_output_columns(monkeypatch) -> None:
    """Generic value outputs are prefixed while named outputs are preserved."""
    indicators = {"fake": _FakeIndicator, "multi": _MultiOutputIndicator}
    monkeypatch.setattr("liq.features.indicators.get_indicator", lambda name: indicators[name])
    bars = pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "timestamp": [1, 2, 3],
            "close": [10.0, 20.0, 30.0],
        }
    )

    result = batch.compute_indicators(
        bars,
        symbol="BTC_USDT",
        timeframe="1h",
        indicators=[("fake", {"multiplier": 3}), ("multi", {})],
        storage=None,
    )

    assert result.columns == ["ts", "fake", "fast", "slow"]
    assert result["fake"].to_list() == [30.0, 60.0, 90.0]
    assert result["slow"].to_list() == [20.0, 40.0, 60.0]


class _StatsStorage:
    def __init__(self) -> None:
        self.frames = {
            "BTC_USDT/indicators/rsi/hash:1h": pl.DataFrame({"value": [1.0, 2.0]}),
            "BTC_USDT/indicators/ema/hash": pl.DataFrame({"value": [3.0]}),
            "BTC_USDT/indicators/broken/hash:1h": RuntimeError("boom"),
            "BTC_USDT/raw/bars": pl.DataFrame({"value": [99.0]}),
        }

    def list_keys(self, prefix: str = "") -> list[str]:
        return [key for key in self.frames if key.startswith(prefix)]

    def read(self, key: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        del start, end
        frame = self.frames[key]
        if isinstance(frame, Exception):
            raise frame
        return frame

    def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
        del mode
        self.frames[key] = data

    def read_latest(self, key: str, n: int = 1) -> pl.DataFrame:
        return self.read(key).tail(n)

    def exists(self, key: str) -> bool:
        return key in self.frames

    def delete(self, key: str) -> bool:
        return self.frames.pop(key, None) is not None

    def get_date_range(self, key: str) -> tuple[date, date] | None:
        del key
        return None


def test_cache_stats_summarizes_indicator_entries_and_read_failures() -> None:
    stats = batch.cache_stats(cast(Any, _StatsStorage()))

    rows = {row["indicator"]: row for row in stats.to_dicts()}
    assert rows["rsi"]["timeframe"] == "1h"
    assert rows["rsi"]["row_count"] == 2
    assert rows["ema"]["timeframe"] == "unknown"
    assert rows["broken"]["row_count"] == 0
    assert rows["broken"]["size_mb"] == 0.0


def test_cache_stats_empty_storage_has_stable_schema() -> None:
    class EmptyStorage:
        def list_keys(self, prefix: str = "") -> list[str]:
            del prefix
            return []

        def read(
            self, key: str, start: date | None = None, end: date | None = None
        ) -> pl.DataFrame:
            del key, start, end
            return pl.DataFrame()

        def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
            del key, data, mode

        def read_latest(self, key: str, n: int = 1) -> pl.DataFrame:
            del key, n
            return pl.DataFrame()

        def exists(self, key: str) -> bool:
            del key
            return False

        def delete(self, key: str) -> bool:
            del key
            return False

        def get_date_range(self, key: str) -> tuple[date, date] | None:
            del key
            return None

    stats = batch.cache_stats(cast(Any, EmptyStorage()))

    assert stats.is_empty()
    assert stats.schema == {
        "indicator": pl.Utf8,
        "timeframe": pl.Utf8,
        "params_id": pl.Utf8,
        "row_count": pl.Int64,
        "size_mb": pl.Float64,
    }
