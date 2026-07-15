"""Causal feature-contract tests for the registered FX JEPA selector."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from liq.features.fractal.fx_jepa import (
    BLOCK_A_COLUMNS,
    BLOCK_D_COLUMNS,
    BLOCK_E_COLUMNS,
    FxJepaFractalSpec,
    build_fx_jepa_fractal_features,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "liq-experiments"
    / "tests"
    / "fixtures"
    / "approved"
    / "fce"
    / "oanda_fx"
    / "minimal.parquet"
)


def _real_return_frame(*, rows: int = 150) -> pl.DataFrame:
    return (
        pl.read_parquet(_FIXTURE)
        .head(rows)
        .with_columns(
            (
                ((pl.col("high").cast(pl.Float64) + pl.col("low").cast(pl.Float64)) / 2.0)
                .log()
                .diff()
            ).alias("midrange_ret")
        )
        .select("timestamp", "midrange_ret")
        .drop_nulls()
    )


def test_registered_schema_covers_exact_frozen_blocks() -> None:
    assert len(BLOCK_A_COLUMNS) == 12
    assert len(BLOCK_D_COLUMNS) == 11
    assert len(BLOCK_E_COLUMNS) == len(BLOCK_D_COLUMNS)
    assert BLOCK_A_COLUMNS == (
        "a_dfa_30_gt_055_value",
        "a_dfa_30_gt_055_state",
        "a_dfa_30_lt_045_value",
        "a_dfa_30_lt_045_state",
        "a_dfa_60_gt_055_value",
        "a_dfa_60_gt_055_state",
        "a_dfa_60_lt_045_value",
        "a_dfa_60_lt_045_state",
        "a_dfa_120_gt_055_value",
        "a_dfa_120_gt_055_state",
        "a_dfa_120_lt_045_value",
        "a_dfa_120_lt_045_state",
    )
    assert BLOCK_D_COLUMNS == (
        "d_scale_dispersion",
        "d_boundary_pressure",
        "d_offset_mean",
        "d_offset_dispersion",
        "d_residual_rms",
        "d_residual_tail_index",
        "d_codebook_entropy",
        "d_domain_concentration",
        "d_codebook_turnover",
        "d_topk_match_margin",
        "d_residual_hurst",
    )
    assert tuple(f"e_delta_{name[2:]}" for name in BLOCK_D_COLUMNS) == BLOCK_E_COLUMNS


def test_real_fixture_build_is_causal_deterministic_and_provenanced() -> None:
    source = _real_return_frame()

    first = build_fx_jepa_fractal_features(source)
    second = build_fx_jepa_fractal_features(source)

    assert first.frame.equals(second.frame)
    assert first.schema_hash == second.schema_hash
    assert len(first.schema_hash) == 64
    assert len(first.config_hash) == 64
    assert first.source_row_count == source.height
    assert first.frame.columns == [
        "timestamp",
        "feature_available_at",
        *BLOCK_A_COLUMNS,
        *BLOCK_D_COLUMNS,
        *BLOCK_E_COLUMNS,
    ]
    assert first.frame.height > 0
    assert first.exclusions["warmup"] == 120
    assert first.exclusions["non_finite_feature"] == 0
    assert first.frame["feature_available_at"].equals(first.frame["timestamp"])
    finite = first.frame.select(
        pl.all().exclude("timestamp", "feature_available_at").is_finite().all()
    )
    assert finite.row(0) == tuple(
        True for _ in (*BLOCK_A_COLUMNS, *BLOCK_D_COLUMNS, *BLOCK_E_COLUMNS)
    )


def test_future_return_mutation_cannot_change_features_at_or_before_cutoff() -> None:
    source = _real_return_frame(rows=160)
    cutoff = source["timestamp"][139]
    mutated = source.with_columns(
        pl.when(pl.col("timestamp") > cutoff)
        .then(pl.col("midrange_ret") * -17.0)
        .otherwise(pl.col("midrange_ret"))
        .alias("midrange_ret")
    )

    original_features = build_fx_jepa_fractal_features(source).frame.filter(
        pl.col("timestamp") <= cutoff
    )
    mutated_features = build_fx_jepa_fractal_features(mutated).frame.filter(
        pl.col("timestamp") <= cutoff
    )

    assert original_features.equals(mutated_features)


def test_evaluation_timestamps_limit_codec_work_and_use_prior_evaluation_state() -> None:
    source = _real_return_frame(rows=180)
    evaluation_timestamps = tuple(source["timestamp"][index] for index in (130, 145, 160))

    batch = build_fx_jepa_fractal_features(source, evaluation_timestamps=evaluation_timestamps)

    assert batch.frame["timestamp"].to_list() == list(evaluation_timestamps[2:])
    assert batch.exclusions["missing_prior_state"] == 1
    assert batch.exclusions["non_finite_feature"] == 1


def test_evaluation_timestamps_must_be_sorted_unique_source_members() -> None:
    source = _real_return_frame(rows=180)
    first = source["timestamp"][130]
    second = source["timestamp"][145]

    with pytest.raises(ValueError, match="sorted and unique"):
        build_fx_jepa_fractal_features(source, evaluation_timestamps=(second, first))
    with pytest.raises(ValueError, match="sorted and unique"):
        build_fx_jepa_fractal_features(source, evaluation_timestamps=(first, first))
    with pytest.raises(ValueError, match="exist in the source"):
        build_fx_jepa_fractal_features(
            source,
            evaluation_timestamps=(source["timestamp"][0].replace(year=2019),),
        )


def test_builder_rejects_raw_price_non_finite_and_non_utc_inputs() -> None:
    source = _real_return_frame()
    with pytest.raises(ValueError, match="raw.price"):
        build_fx_jepa_fractal_features(
            source.rename({"midrange_ret": "close"}), value_column="close"
        )
    with pytest.raises(ValueError, match="finite"):
        build_fx_jepa_fractal_features(
            source.with_row_index().with_columns(
                pl.when(pl.col("index") == 50)
                .then(float("nan"))
                .otherwise(pl.col("midrange_ret"))
                .alias("midrange_ret")
            )
        )
    with pytest.raises(ValueError, match="UTC"):
        build_fx_jepa_fractal_features(
            source.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        )


def test_builder_rejects_missing_null_unsorted_and_duplicate_inputs() -> None:
    source = _real_return_frame()
    with pytest.raises(ValueError, match="missing columns"):
        build_fx_jepa_fractal_features(source.drop("midrange_ret"))
    with pytest.raises(ValueError, match="finite and non-null"):
        build_fx_jepa_fractal_features(
            source.with_row_index().with_columns(
                pl.when(pl.col("index") == 50)
                .then(None)
                .otherwise(pl.col("midrange_ret"))
                .alias("midrange_ret")
            )
        )
    with pytest.raises(ValueError, match="sorted"):
        build_fx_jepa_fractal_features(source.reverse())
    with pytest.raises(ValueError, match="unique"):
        build_fx_jepa_fractal_features(pl.concat([source, source.tail(1)]))


def test_short_real_fixture_emits_typed_empty_feature_frame() -> None:
    source = _real_return_frame(rows=60)

    batch = build_fx_jepa_fractal_features(source)

    assert batch.frame.is_empty()
    assert batch.frame.schema["timestamp"] == source.schema["timestamp"]
    assert all(
        dtype == pl.Float64
        for name, dtype in batch.frame.schema.items()
        if name not in {"timestamp", "feature_available_at"}
    )


def test_spec_rejects_frozen_parameter_drift() -> None:
    with pytest.raises(ValueError, match="frozen"):
        FxJepaFractalSpec(dfa_windows=(30, 60))
    with pytest.raises(ValueError, match="frozen"):
        FxJepaFractalSpec(codec_window=32)
