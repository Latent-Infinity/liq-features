"""CLI for liq-features cache management.

Provides commands for:
- Cache statistics
- Listing cached indicators
- Cleaning cache entries with filters
- Index management
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="liq-features",
    help="liq-features CLI for cache management and feature operations",
    no_args_is_help=True,
)

cache_app = typer.Typer(
    name="cache",
    help="Cache management commands",
    no_args_is_help=True,
)
app.add_typer(cache_app, name="cache")

console = Console()


def _get_cache(cache_dir: Path | None = None):
    """Get IndicatorCache instance."""
    from liq.features.cache import IndicatorCache
    from liq.store.parquet import ParquetStore

    if cache_dir is not None:
        storage = ParquetStore(str(cache_dir))
        return IndicatorCache(storage=storage)
    return IndicatorCache()


def _format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


@cache_app.command("stats")
def cache_stats(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", "-d", help="Cache directory (default: from env)"
    ),
) -> None:
    """Show cache statistics."""
    cache = _get_cache(cache_dir)
    stats = cache.stats()

    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    root = cache.storage_root
    root_label = str(root) if root else "unknown"
    table.add_row("Storage Root", root_label)
    table.add_row("Total Entries", f"{stats['entries']:,}")
    table.add_row("Total Size", _format_bytes(stats["total_size_bytes"]))

    console.print(table)


@cache_app.command("list")
def cache_list(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", "-d", help="Cache directory"
    ),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    indicator: Optional[str] = typer.Option(
        None, "--indicator", "-i", help="Filter by indicator (supports wildcards)"
    ),
    timeframe: Optional[str] = typer.Option(
        None, "--timeframe", "-t", help="Filter by timeframe"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit results"),
) -> None:
    """List cached indicator entries."""
    from liq.features.cache_models import CacheFilter

    cache = _get_cache(cache_dir)
    filters = CacheFilter(
        symbol=symbol,
        indicator=indicator,
        timeframe=timeframe,
        limit=limit,
    )

    entries = cache.list_entries(filters if filters.symbol or filters.indicator or filters.timeframe or filters.limit else None)

    if not entries:
        console.print("[yellow]No cache entries found.[/yellow]")
        return

    table = Table(title=f"Cache Entries ({len(entries)} found)")
    table.add_column("Symbol", style="cyan")
    table.add_column("Indicator", style="green")
    table.add_column("Timeframe", style="blue")
    table.add_column("Params Hash", style="dim")
    table.add_column("Size", style="magenta", justify="right")

    for entry in entries:
        table.add_row(
            entry.symbol,
            entry.indicator,
            entry.timeframe,
            entry.params_hash[:8] + "...",
            _format_bytes(entry.size_bytes),
        )

    console.print(table)


@cache_app.command("clean")
def cache_clean(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", "-d", help="Cache directory"
    ),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    indicator: Optional[str] = typer.Option(
        None, "--indicator", "-i", help="Filter by indicator (supports wildcards)"
    ),
    timeframe: Optional[str] = typer.Option(
        None, "--timeframe", "-t", help="Filter by timeframe"
    ),
    data_hash: Optional[str] = typer.Option(
        None, "--data-hash", help="Filter by data hash"
    ),
    older_than: Optional[str] = typer.Option(
        None, "--older-than", help="Filter by age (e.g., '7d', '24h')"
    ),
    all_entries: bool = typer.Option(
        False, "--all", help="Clean all entries (required if no filters)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview without deleting"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt"
    ),
) -> None:
    """Clean cache entries matching criteria."""
    from liq.features.cache_models import CleanupCriteria

    cache = _get_cache(cache_dir)

    # Check if any filter is specified
    has_filters = any([symbol, indicator, timeframe, data_hash, older_than])

    if not has_filters and not all_entries:
        console.print(
            "[red]Error:[/red] Specify at least one filter or use --all to clean everything."
        )
        raise typer.Exit(1)

    criteria = CleanupCriteria(
        symbol=symbol,
        indicator=indicator,
        timeframe=timeframe,
        data_hash=data_hash,
        older_than=older_than,
    )

    # Preview what will be deleted
    if dry_run:
        result = cache.clean(criteria, dry_run=True)
        console.print(f"[yellow]Dry run:[/yellow] Would delete {result.deleted_count} entries ({_format_bytes(result.freed_bytes)})")
        return

    # Get preview for confirmation
    result = cache.clean(criteria, dry_run=True)

    if result.deleted_count == 0:
        console.print("[yellow]No entries match the criteria.[/yellow]")
        return

    # Confirm unless --force
    if not force:
        confirmed = typer.confirm(
            f"Delete {result.deleted_count} entries ({_format_bytes(result.freed_bytes)})?"
        )
        if not confirmed:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    # Actually delete
    result = cache.clean(criteria, dry_run=False)

    if result.success:
        console.print(
            f"[green]✓[/green] Deleted {result.deleted_count} entries, freed {_format_bytes(result.freed_bytes)}"
        )
    else:
        console.print(
            f"[yellow]Deleted {result.deleted_count} entries with {len(result.errors)} errors[/yellow]"
        )
        for error in result.errors[:5]:
            console.print(f"  [red]•[/red] {error}")


@cache_app.command("rebuild-index")
def cache_rebuild_index(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", "-d", help="Cache directory"
    ),
) -> None:
    """Rebuild cache index from storage keys."""
    cache = _get_cache(cache_dir)
    df = cache.rebuild_index()
    console.print(f"[green]✓[/green] Rebuilt cache index: {len(df)} entries")


# Legacy command for backward compatibility
@app.command("indicator-cache")
def indicator_cache_legacy(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", help="Cache directory"
    ),
    clear: bool = typer.Option(False, "--clear", help="Clear all cache entries"),
) -> None:
    """Show cache status (legacy command, use 'cache stats' instead)."""
    cache = _get_cache(cache_dir)

    if clear:
        stats = cache.stats()
        cache.clear()
        console.print(f"Cleared {stats['entries']:,} cache entries")
        return

    stats = cache.stats()
    root = cache.storage_root
    root_label = str(root) if root else "unknown"
    console.print(f"Cache storage root: {root_label}")
    console.print(f"Entries: {stats['entries']:,}")
    console.print(f"Total size: {_format_bytes(stats['total_size_bytes'])}")


def main() -> int:
    """Main entry point."""
    try:
        app()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
