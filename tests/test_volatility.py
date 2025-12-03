import polars as pl

from liq.features.volatility import garman_klass, yang_zhang


def test_yang_zhang_non_empty() -> None:
    df = pl.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
        }
    )
    vol = yang_zhang(df)
    assert vol > 0


def test_garman_klass_non_empty() -> None:
    df = pl.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
        }
    )
    vol = garman_klass(df)
    assert vol > 0
