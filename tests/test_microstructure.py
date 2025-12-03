import polars as pl

from liq.features.microstructure import corwin_schultz_spread


def test_corwin_schultz_spread_positive() -> None:
    df = pl.DataFrame(
        {
            "high": pl.Series("high", [2.0, 2.2, 2.4], dtype=pl.Float64),
            "low": pl.Series("low", [1.8, 1.9, 2.0], dtype=pl.Float64),
            "open": pl.Series("open", [2.0, 2.0, 2.0], dtype=pl.Float64),
            "close": pl.Series("close", [2.0, 2.0, 2.0], dtype=pl.Float64),
        }
    )
    spread = corwin_schultz_spread(df)
    assert spread >= 0
