"""Tests for NumPy conversion fallbacks."""

import polars as pl

from liq.features.numpy_utils import to_numpy_float64


def test_to_numpy_float64_series_falls_back_when_zero_copy_is_not_possible() -> None:
    values = pl.Series("x", [1, 2, 3])

    result = to_numpy_float64(values, allow_copy=False)

    assert result.dtype == "float64"
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_to_numpy_float64_dataframe_falls_back_when_zero_copy_is_not_possible() -> None:
    values = pl.DataFrame({"x": [1, 2], "y": [3, 4]})

    result = to_numpy_float64(values, allow_copy=False)

    assert result.dtype == "float64"
    assert result.tolist() == [[1.0, 3.0], [2.0, 4.0]]
