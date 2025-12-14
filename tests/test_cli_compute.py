"""CLI compute command tests."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from liq.features.cli import app


def test_cli_compute_stores_feature(tmp_path: Path) -> None:
    runner = CliRunner()
    data_path = tmp_path / "bars.parquet"
    store_root = tmp_path / "store"

    df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
        "open": [1.0],
        "high": [2.0],
        "low": [0.5],
        "close": [1.5],
        "volume": [10.0],
    })
    df.write_parquet(data_path)

    result = runner.invoke(
        app,
        [
            "compute",
            str(data_path),
            "--symbol",
            "EUR_USD",
            "--timeframe",
            "1m",
            "--store-root",
            str(store_root),
            "--feature-name",
            "derived_midrange",
        ],
    )
    assert result.exit_code == 0
    # Feature stored under KeyBuilder.features symbol/name
    stored = list(store_root.rglob("*.parquet"))
    assert stored
