"""Feature-contract tests backed by the approved OANDA fixture."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from liq.features.fractal_descriptors import (
    DescriptorInterpretation,
    build_cheap_descriptor_features,
)
from liq.fractal import FractalDescriptorSpec, InputDomain

WORKSPACE = Path(__file__).resolve().parents[2]
APPROVED_OANDA = WORKSPACE / "liq-experiments/tests/fixtures/approved/fce/oanda_fx/minimal.parquet"


def _approved_returns() -> tuple[float, ...]:
    frame = pl.read_parquet(APPROVED_OANDA).sort("timestamp")
    closes = tuple(float(value) for value in frame["close"].head(121))
    return tuple(
        (current - previous) / previous
        for previous, current in zip(closes[:-1], closes[1:], strict=True)
    )


def test_cheap_descriptor_features_label_short_windows_and_emit_metadata() -> None:
    specs = tuple(
        FractalDescriptorSpec(
            estimator=estimator,
            input_domain=InputDomain.LOG_RETURNS,
            window=120,
            sampling="1m",
            detrending="linear" if estimator == "dfa" else None,
        )
        for estimator in ("hurst_rs", "dfa", "ghe")
    )

    features = build_cheap_descriptor_features(
        _approved_returns(),
        specs=specs,
        feature_timestamp_ns=1_578_268_740_000_000_000,
        feature_available_at_ns=1_578_268_800_000_000_000,
    )

    assert [feature.metadata()["estimator"] for feature in features] == [
        "hurst_rs",
        "dfa",
        "ghe",
    ]
    for feature in features:
        metadata = feature.metadata()
        assert metadata["window"] == 120
        assert metadata["sampling"] == "1m"
        assert metadata["feature_available_at_ns"] == 1_578_268_800_000_000_000
        assert metadata["interpretation"] == "roughness_path_efficiency"
        assert feature.interpretation is DescriptorInterpretation.ROUGHNESS_PATH_EFFICIENCY


def test_diagnostics_can_upgrade_short_window_interpretation_without_threshold_rule() -> None:
    spec = FractalDescriptorSpec("dfa", InputDomain.VOLATILITY_PROXY, 60, "1m", "linear")

    (feature,) = build_cheap_descriptor_features(
        _approved_returns(),
        specs=(spec,),
        feature_timestamp_ns=10,
        feature_available_at_ns=11,
        diagnostics_passed=True,
    )

    assert feature.interpretation is DescriptorInterpretation.HURST_EXPONENT
    assert "threshold" not in feature.metadata()


def test_descriptor_builder_rejects_raw_price_inputs() -> None:
    spec = FractalDescriptorSpec("dfa", InputDomain.RAW_PRICE, 60, "1m", "linear")

    with pytest.raises(ValueError, match="raw-price"):
        build_cheap_descriptor_features(
            _approved_returns(),
            specs=(spec,),
            feature_timestamp_ns=10,
            feature_available_at_ns=11,
        )


def test_descriptor_builder_rejects_unknown_estimator_and_insufficient_window() -> None:
    unknown = FractalDescriptorSpec("unknown", InputDomain.RETURNS, 30, "1m")
    with pytest.raises(ValueError, match="unsupported estimator"):
        build_cheap_descriptor_features(
            _approved_returns(),
            specs=(unknown,),
            feature_timestamp_ns=10,
            feature_available_at_ns=11,
        )

    dfa = FractalDescriptorSpec("dfa", InputDomain.RETURNS, 120, "1m", "linear")
    with pytest.raises(ValueError, match="requires 120"):
        build_cheap_descriptor_features(
            _approved_returns()[:60],
            specs=(dfa,),
            feature_timestamp_ns=10,
            feature_available_at_ns=11,
        )
