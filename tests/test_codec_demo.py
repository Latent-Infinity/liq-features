"""Diagnostic-only tests for the fractal codec demo CLI.

These tests assert the demo makes no economic claim and stays within the
DEV_SMOKE_ONLY ceiling; they cite no performance and rely only on a committed
real-data fixture.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from liq.features.fractal.codec_demo import run_codec_demo
from liq.fractal import InputDomain
from liq.fractal.codec import CodecParams

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "liq-experiments"
    / "tests"
    / "fixtures"
    / "approved"
    / "fce"
    / "oanda_fx"
    / "minimal.parquet"
)


def test_fixture_exists() -> None:
    assert FIXTURE.exists()


def test_status_and_claims() -> None:
    report = run_codec_demo(FIXTURE)
    assert report["status"] == "DEV_SMOKE_ONLY"
    assert report["claims"] == []
    assert report["diagnostic_only"] is True
    assert report["object_type"] == "fce_codec_demo"
    assert report["selection_authority"] == "human_only"


def test_fixture_metadata() -> None:
    report = run_codec_demo(FIXTURE)
    fixture = report["fixture"]
    assert fixture["path"] == str(FIXTURE)
    assert fixture["row_count"] > 0
    assert fixture["return_count"] == fixture["row_count"] - 1


def test_codec_params_metadata() -> None:
    report = run_codec_demo(FIXTURE)
    params = report["codec_params"]
    assert params["window"] == 64
    assert params["range_length"] == 8
    assert params["stride_domain"] == 4
    assert params["input_domain"] == "returns"


def test_determinism() -> None:
    assert run_codec_demo(FIXTURE) == run_codec_demo(FIXTURE)


def test_no_economic_claim_substrings() -> None:
    report = run_codec_demo(FIXTURE)
    blob = json.dumps(report).lower()
    assert "pnl" not in blob
    assert "sharpe" not in blob
    raw = json.dumps(report)
    assert "PASS" not in raw
    assert "FAIL" not in raw


def test_well_posedness_keys() -> None:
    report = run_codec_demo(FIXTURE)
    well = report["well_posedness"]
    assert set(well) == {
        "boundary_pressure",
        "codebook_entropy",
        "hill_stable",
        "residual_tail_fallback",
    }
    assert isinstance(well["boundary_pressure"], float)
    assert isinstance(well["codebook_entropy"], float)
    assert isinstance(well["hill_stable"], bool)
    assert isinstance(well["residual_tail_fallback"], dict)


def test_block_d_feature_keys() -> None:
    report = run_codec_demo(FIXTURE)
    features = report["features"]
    for key in ("scale_dispersion", "boundary_pressure", "codebook_entropy", "residual_rms"):
        assert key in features


def test_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    mod = importlib.import_module("liq.features.fractal.codec_demo")
    assert mod.main(["--fixture", str(FIXTURE), "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "DEV_SMOKE_ONLY"


def test_cli_without_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    mod = importlib.import_module("liq.features.fractal.codec_demo")
    assert mod.main(["--fixture", str(FIXTURE)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["diagnostic_only"] is True


def test_raw_price_input_domain_rejected() -> None:
    with pytest.raises(ValueError):
        run_codec_demo(
            FIXTURE,
            params=CodecParams(window=64, range_length=8, input_domain=InputDomain.RAW_PRICE),
        )
