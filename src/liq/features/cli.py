"""Typer CLI for feature pipeline operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console

from liq.features.pipeline import FeaturePipeline

app = typer.Typer(help="liq-features CLI")
console = Console()


@app.command("fit-pipeline")
def fit_pipeline(
    series_path: Path = typer.Argument(..., help="Path to CSV/Parquet/JSON series with a 'value' column"),
    model_type: str = typer.Option("nn", help="Model type: tree|nn|transformer|diffusion"),
    d: float = typer.Option(0.4, help="Fractional differencing d"),
    output: Path = typer.Option(..., help="Where to write pipeline JSON"),
) -> None:
    """Fit stationarity + scaling pipeline and persist state."""
    series = _load_series(series_path)
    pipeline = FeaturePipeline(model_type=model_type, d=d)
    pipeline.fit(series)
    output.write_text(json.dumps(pipeline.to_dict()))
    console.print(f"[green]Saved pipeline to {output}[/green]")


@app.command("transform")
def transform(
    series_path: Path = typer.Argument(..., help="Path to CSV/Parquet/JSON series with a 'value' column"),
    pipeline_path: Path = typer.Argument(..., help="Path to pipeline JSON"),
    output: Optional[Path] = typer.Option(None, help="Optional output path for transformed values"),
) -> None:
    """Apply persisted pipeline to a series."""
    series = _load_series(series_path)
    state = json.loads(pipeline_path.read_text())
    pipeline = FeaturePipeline.from_dict(state)
    transformed = pipeline.transform(series)
    if output:
        output.write_text(json.dumps(transformed))
        console.print(f"[green]Wrote transformed series to {output}[/green]")
    else:
        console.print(transformed)


def _load_series(path: Path) -> list[float]:
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pl.read_csv(path)
    else:
        with path.open() as f:
            data = json.load(f)
        df = pl.DataFrame(data)
    if "value" not in df.columns:
        raise ValueError("Expected column 'value'")
    return df["value"].to_list()


if __name__ == "__main__":
    app()
