"""Tests for regime typed result objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from liq.features.regime import RegimeFrame


def _probabilities() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "regime_0": [0.7, 0.1, 0.2],
            "regime_1": [0.2, 0.8, 0.3],
            "regime_2": [0.1, 0.1, 0.5],
        }
    )


def test_regime_frame_is_frozen_and_stores_cluster_ids_and_probabilities() -> None:
    frame = RegimeFrame(
        cluster_ids=pl.Series("cluster_id", [0, 1, 2]),
        probabilities=_probabilities(),
    )

    assert frame.cluster_ids.to_list() == [0, 1, 2]
    assert frame.probabilities.columns == ["regime_0", "regime_1", "regime_2"]
    with pytest.raises(FrozenInstanceError):
        frame.cluster_ids = pl.Series("cluster_id", [1, 1, 1])  # type: ignore[misc]


def test_regime_frame_rejects_mismatched_row_count() -> None:
    with pytest.raises(ValueError, match="row count"):
        RegimeFrame(
            cluster_ids=pl.Series("cluster_id", [0, 1]),
            probabilities=_probabilities(),
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        pl.DataFrame({"regime_0": [0.5], "regime_2": [0.5]}),
        pl.DataFrame({"regime_0": [1.0]}),
        pl.DataFrame({"foo_0": [0.5], "foo_1": [0.5]}),
    ],
)
def test_regime_frame_rejects_invalid_probability_columns(probabilities: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="probability columns"):
        RegimeFrame(
            cluster_ids=pl.Series("cluster_id", [0] * probabilities.height),
            probabilities=probabilities,
        )


def test_regime_frame_rejects_probability_rows_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum"):
        RegimeFrame(
            cluster_ids=pl.Series("cluster_id", [0, 1]),
            probabilities=pl.DataFrame(
                {
                    "regime_0": [0.4, 0.2],
                    "regime_1": [0.4, 0.2],
                }
            ),
        )


def test_regime_frame_rejects_out_of_range_cluster_ids() -> None:
    with pytest.raises(ValueError, match="cluster_id"):
        RegimeFrame(
            cluster_ids=pl.Series("cluster_id", [0, 3, 1]),
            probabilities=_probabilities(),
        )
