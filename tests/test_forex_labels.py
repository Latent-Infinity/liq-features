from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from liq.features.forex_labels import make_forex_labels


def _sample_df(rows: int) -> pl.DataFrame:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    base = [1.0000 + i * 0.001 for i in range(rows)]
    return pl.DataFrame(
        {
            "timestamp": [ts0 + timedelta(hours=i) for i in range(rows)],
            "high": [v + 0.001 for v in base],
            "low": [v - 0.001 for v in base],
            "close": base,
        }
    )


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


def test_make_forex_labels_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_bars must be positive"):
        make_forex_labels(_sample_df(10), horizon_bars=0)


def test_make_forex_labels_rejects_missing_price_column() -> None:
    with pytest.raises(ValueError, match="price_col 'mid' not in DataFrame"):
        make_forex_labels(_sample_df(10), price_col="mid")


def test_make_forex_labels_rejects_missing_atr_columns_without_spread() -> None:
    df = pl.DataFrame({"close": [1.0, 1.1, 1.2]})

    with pytest.raises(ValueError, match="ATR threshold requires high/low/close columns"):
        make_forex_labels(df, horizon_bars=1)


def test_make_forex_labels_uses_atr_threshold_when_spread_column_absent() -> None:
    out = make_forex_labels(_sample_df(20), horizon_bars=1, atr_window=3, threshold_k=0.0)

    assert out.height == 17
    assert out["threshold"].min() >= 0
    assert set(out["label"].unique().to_list()) <= {0, 1, 2}
