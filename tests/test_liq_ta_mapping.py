"""Regression tests for liq-ta dynamic input mapping."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import polars as pl

from liq.features.indicators.liq_ta import map_inputs


def test_map_inputs_price0_price1_fallback_to_aliased_targets() -> None:
    df = pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "price0": [10.0, 11.0, 12.0],
            "price1": [20.0, 21.0, 22.0],
        }
    )
    input_names = OrderedDict({"price0": "close", "price1": "high"})

    inputs = map_inputs(df, input_names, indicator_name="CORREL")

    assert "close" in inputs
    assert "high" in inputs
    np.testing.assert_array_equal(inputs["close"], df["price0"].to_numpy())
    np.testing.assert_array_equal(inputs["high"], df["price1"].to_numpy())
