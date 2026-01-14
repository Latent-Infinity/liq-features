from datetime import UTC, datetime, timedelta

import polars as pl

from liq.features.forex_labels import make_forex_labels


def _sample_df(rows: int) -> pl.DataFrame:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    base = [1.0000 + i * 0.001 for i in range(rows)]
    return pl.DataFrame({
        "timestamp": [ts0 + timedelta(hours=i) for i in range(rows)],
        "high": [v + 0.001 for v in base],
        "low": [v - 0.001 for v in base],
        "close": base,
    })


def test_make_forex_labels_direction() -> None:
    df = _sample_df(10).with_columns(pl.lit(1.0).alias("spread_bps"))
    out = make_forex_labels(df, horizon_bars=2, threshold_k=0.0, spread_bps_col="spread_bps")
    assert out["label"].min() >= 1
    assert out["label"].max() <= 2
    assert out.height == 8


def test_make_forex_labels_spread_threshold() -> None:
    df = _sample_df(10).with_columns(pl.lit(2.0).alias("spread_bps"))
    out = make_forex_labels(df, horizon_bars=2, spread_bps_col="spread_bps")
    assert "future_return" in out.columns
    assert "label" in out.columns
