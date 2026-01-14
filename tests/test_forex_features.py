from datetime import UTC, datetime, timedelta

import polars as pl

from liq.features.forex import build_features


def _sample_df(rows: int, with_volume: bool = True) -> pl.DataFrame:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    base = [1.1000 + i * 0.0001 for i in range(rows)]
    df = pl.DataFrame({
        "timestamp": [ts0 + timedelta(hours=i) for i in range(rows)],
        "open": base,
        "high": [v + 0.0002 for v in base],
        "low": [v - 0.0002 for v in base],
        "close": [v + 0.00005 for v in base],
    })
    if with_volume:
        df = df.with_columns(pl.Series("volume", [1000 + i for i in range(rows)]))
    return df


def test_build_features_outputs_schema() -> None:
    df = _sample_df(200, with_volume=True)
    out = build_features(df, atr_window=14, zscore_window=20, warmup_cut=50)
    for col in ["midrange_ret", "range_ret", "volume_ret", "atr_zscore"]:
        assert col in out.columns
    any_null = out.select(pl.any_horizontal(pl.all().is_null())).to_series().any()
    assert not any_null


def test_build_features_missing_volume() -> None:
    df = _sample_df(200, with_volume=False)
    out = build_features(df, atr_window=14, zscore_window=20, warmup_cut=50)
    assert "volume_ret" in out.columns
