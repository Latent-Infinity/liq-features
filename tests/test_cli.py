import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from liq.features.cache import IndicatorCache
from liq.features.cli import app

runner = CliRunner()


class TestIndicatorCacheCLI:
    """Tests for indicator-cache CLI command."""

    def test_cache_status_empty(self, tmp_path: Path) -> None:
        """Test cache status with empty cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = runner.invoke(app, ["indicator-cache", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "0" in result.output or "entries" in result.output.lower()

    def test_cache_status_with_entries(self, tmp_path: Path) -> None:
        """Test cache status shows entry count."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(cache_dir=cache_dir)
        df = pl.DataFrame({"ts": [datetime(2024, 1, 15, tzinfo=UTC)], "value": [1.0]})
        cache.set("test_key_1", df)
        cache.set("test_key_2", df)

        result = runner.invoke(app, ["indicator-cache", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "2" in result.output

    def test_cache_clear(self, tmp_path: Path) -> None:
        """Test cache clear command."""
        cache_dir = tmp_path / "cache"
        cache = IndicatorCache(cache_dir=cache_dir)
        df = pl.DataFrame({"ts": [datetime(2024, 1, 15, tzinfo=UTC)], "value": [1.0]})
        cache.set("test_key", df)
        assert cache.has("test_key")

        result = runner.invoke(app, ["indicator-cache", "--cache-dir", str(cache_dir), "--clear"])
        assert result.exit_code == 0
        assert not cache.has("test_key")

    def test_cache_default_dir(self) -> None:
        """Test cache command uses default directory."""
        result = runner.invoke(app, ["indicator-cache"])
        assert result.exit_code == 0


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


def test_validate_command_detects_cycles(tmp_path: Path) -> None:
    feature_set_path = tmp_path / "fs.json"
    feature_set_path.write_text(
        json.dumps(
            [
                {"name": "a", "dependencies": ["b"]},
                {"name": "b", "dependencies": ["a"]},
            ]
        )
    )
    result = runner.invoke(app, ["validate", str(feature_set_path)])
    assert result.exit_code != 0
    assert "Cyclic" in result.output or "cycle" in result.output.lower()
