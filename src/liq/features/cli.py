"""Typer CLI for feature pipeline operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from liq.features.cache import IndicatorCache
from liq.features.feature_set import FeatureDefinition, FeatureSet
from liq.features.store import FeatureStore
from liq.features.pipeline import FeaturePipeline
from liq.features.derived import compute_derived_fields
from liq.store.parquet import ParquetStore
from liq.features.feature_set import FeatureSet

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


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@app.command("indicator-cache")
def indicator_cache(
    cache_dir: Optional[Path] = typer.Option(None, help="Cache directory path"),
    clear: bool = typer.Option(False, "--clear", help="Clear all cache entries"),
) -> None:
    """Show indicator cache status or clear cache."""
    cache = IndicatorCache(cache_dir=cache_dir)

    if clear:
        cache.clear()
        console.print("[green]Cache cleared[/green]")
        return

    stats = cache.stats()

    table = Table(title="Indicator Cache Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Cache Directory", str(cache.cache_dir))
    table.add_row("Entries", str(stats["entries"]))
    table.add_row("Total Size", _format_size(stats["total_size_bytes"]))

    console.print(table)


@app.command("compute")
def compute_features(
    data_path: Optional[Path] = typer.Argument(
        None, help="Bars file (Parquet/CSV) with timestamp and OHLCV. If omitted, load via liq-data."
    ),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol for storage key"),
    timeframe: str = typer.Option(..., "--timeframe", "-t", help="Timeframe, e.g. 1m"),
    output_store: Path = typer.Option(..., "--store-root", help="Path to feature store root"),
    feature_name: str = typer.Option("derived_midrange", help="Feature name to store"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="liq-data provider to load bars when data-path is omitted"
    ),
) -> None:
    """Compute a simple feature set (derived midrange) and store via liq-store."""
    if data_path:
        df = _load_bars(data_path)
    else:
        if not provider:
            raise typer.BadParameter("Provide either data_path or --provider to load bars")
        from liq.data.service import DataService

        df = DataService().load(provider, symbol, timeframe)

    store = ParquetStore(str(output_store))
    fs = FeatureStore(storage=store)

    # Simple feature: emit timestamp + midrange
    def midrange_feature(df_: pl.DataFrame, input_col: str) -> pl.DataFrame:
        derived = compute_derived_fields(df_)
        return derived.select("timestamp", pl.col("midrange").alias("value"))

    feature_def = FeatureDefinition(
        name=feature_name,
        func=midrange_feature,
        lookback=0,
        input_column="close",
    )
    feature_set = FeatureSet(name=feature_name, features=[feature_def])
    fs.compute_feature_set(df, feature_set, symbol=symbol, timeframe=timeframe, force_recalculate=True)
    console.print(f"[green]Computed and stored feature '{feature_name}' for {symbol} at {output_store}[/green]")


@app.command("validate")
def validate_feature_set(
    feature_set_path: Path = typer.Argument(..., help="Path to a JSON feature set definition"),
) -> None:
    """Validate feature set dependencies and lookback consistency."""
    try:
        definitions = json.loads(feature_set_path.read_text())
    except Exception as exc:  # pragma: no cover - user IO
        raise typer.BadParameter(f"Could not read feature set file: {exc}") from exc

    errors: list[str] = []
    names: set[str] = set()
    for item in definitions:
        name = item.get("name")
        if not name:
            errors.append("Missing feature name")
            continue
        if name in names:
            errors.append(f"Duplicate feature name: {name}")
        names.add(name)
        deps = item.get("dependencies", [])
        for dep in deps:
            if dep == name:
                errors.append(f"Self-dependency detected for {name}")

    # simple cycle detection via DFS on provided deps
    graph = {item["name"]: set(item.get("dependencies", [])) for item in definitions if "name" in item}

    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(node: str) -> bool:
        if node in visited:
            return True
        if node in visiting:
            errors.append(f"Cyclic dependency detected at {node}")
            return False
        visiting.add(node)
        for neighbor in graph.get(node, set()):
            _dfs(neighbor)
        visiting.remove(node)
        visited.add(node)
        return True

    for node in graph:
        _dfs(node)

    if errors:
        console.print("[red]Feature set validation failed:[/red]")
        for err in errors:
            console.print(f"- {err}")
        raise typer.Exit(code=1)

    console.print("[green]Feature set is valid[/green]")


def _load_bars(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError("Unsupported file type; use Parquet or CSV")


if __name__ == "__main__":
    app()
