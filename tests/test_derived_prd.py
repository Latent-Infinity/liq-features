"""PRD-aligned tests for derived fields."""

from decimal import Decimal

import polars as pl

from liq.features.derived import compute_derived_fields


def test_compute_derived_fields_prd_formulas() -> None:
    df = pl.DataFrame({
        "open": [1.0, 2.0],
        "high": [2.0, 3.0],
        "low": [0.5, 1.5],
        "close": [1.5, 2.5],
    })

    result = compute_derived_fields(df)
    assert "midrange" in result.columns
    assert "range" in result.columns
    assert "true_range_midrange" in result.columns
    assert "true_range_hl" in result.columns

    # First row: prev values missing -> true_range_midrange equals range
    assert result["true_range_midrange"][0] == Decimal("1.5")

    # Second row: midrange diff is max(1.5, |2.25-1.25|=1.0) = 1.5
    assert result["true_range_midrange"][1] == Decimal("1.5")
    # true_range_hl: max(range=1.5, |3-2|=1, |1.5-0.5|=1) = 1.5
    assert result["true_range_hl"][1] == Decimal("1.5")
