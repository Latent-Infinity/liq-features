import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from liq.features.cli import app

runner = CliRunner()


def test_fit_pipeline_and_transform(tmp_path: Path) -> None:
    series = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
    series_path = tmp_path / "series.parquet"
    series.write_parquet(series_path)
    pipeline_path = tmp_path / "pipe.json"

    res_fit = runner.invoke(app, ["fit-pipeline", str(series_path), "--model-type", "nn", "--d", "0.3", "--output", str(pipeline_path)])
    assert res_fit.exit_code == 0
    assert pipeline_path.exists()

    out_path = tmp_path / "out.json"
    res_transform = runner.invoke(app, ["transform", str(series_path), str(pipeline_path), "--output", str(out_path)])
    assert res_transform.exit_code == 0
    transformed = json.loads(out_path.read_text())
    assert len(transformed) == 3
