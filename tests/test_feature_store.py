"""FeatureStore caching and keying tests."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from liq.features.feature_set import FeatureDefinition, FeatureSet
from liq.features.store import FeatureStore
from liq.store.parquet import ParquetStore


def test_feature_store_caches_results(tmp_path: Path) -> None:
    store = ParquetStore(str(tmp_path))
    fs = FeatureStore(storage=store)

    def simple_feature(df: pl.DataFrame, input_col: str) -> pl.DataFrame:
        return df.select("timestamp", pl.col(input_col).alias("value"))

    feature_def = FeatureDefinition(name="test_feature", func=simple_feature, lookback=0, input_column="close")
    feature_set = FeatureSet(name="basic", features=[feature_def])

    ts = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame({
        "timestamp": [ts],
        "open": [1.0],
        "high": [1.1],
        "low": [0.9],
        "close": [1.0],
    })

    out1 = fs.compute_feature_set(df, feature_set, symbol="EUR_USD", timeframe="1m")
    assert store.list_keys(prefix="EUR_USD/indicators/")  # key created

    out2 = fs.compute_feature_set(df, feature_set, symbol="EUR_USD", timeframe="1m")
    assert out1["test_feature"].equals(out2["test_feature"])


def test_feature_store_stores_feature_set_with_timeframe(tmp_path: Path) -> None:
    store = ParquetStore(str(tmp_path))
    fs = FeatureStore(storage=store)

    def with_lookback(df: pl.DataFrame, input_col: str) -> pl.DataFrame:
        return df.select("timestamp", pl.col(input_col).alias("value"))

    feature_def = FeatureDefinition(name="lb", func=with_lookback, lookback=1, input_column="close")
    feature_set = FeatureSet(name="setA", features=[feature_def])

    ts = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame({
        "timestamp": [ts, ts],
        "open": [1.0, 1.0],
        "high": [1.1, 1.1],
        "low": [0.9, 0.9],
        "close": [1.0, 1.1],
    })

    fs.compute_feature_set(df, feature_set, symbol="EUR_USD", timeframe="1m", force_recalculate=True)

    keys = store.list_keys(prefix="EUR_USD/features/")
    assert any(key.endswith("setA:1m") for key in keys)


def test_feature_store_resolves_dependencies(tmp_path: Path) -> None:
    store = ParquetStore(str(tmp_path))
    fs = FeatureStore(storage=store)

    def base(df: pl.DataFrame, input_col: str) -> pl.DataFrame:
        return df.select("timestamp", pl.col("close").alias("value"))

    def dependent(df: pl.DataFrame, input_col: str) -> pl.DataFrame:
        return df.select("timestamp", (pl.col("base") + 1).alias("value"))

    base_def = FeatureDefinition(name="base", func=base, lookback=0, input_column="close")
    dep_def = FeatureDefinition(name="dep", func=dependent, lookback=0, input_column="close", dependencies=["base"])
    fs_def = FeatureSet(name="deps", features=[base_def, dep_def])

    ts = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame({
        "timestamp": [ts],
        "open": [1.0],
        "high": [1.1],
        "low": [0.9],
        "close": [1.0],
    })

    outputs = fs.compute_feature_set(df, fs_def, symbol="EUR_USD", timeframe="1m", force_recalculate=True)
    assert outputs["dep"]["value"].to_list() == [2.0]
    dep_keys = [k for k in store.list_keys(prefix="EUR_USD/indicators/dep/")]
    assert dep_keys
