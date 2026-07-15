"""Diagnostic-only demo CLI for the reference fractal compression codec.

Loads a committed parquet fixture, builds simple returns from its ``close``
column (raw prices are never fed to the codec), encodes the most recent window,
and emits the Block-D codebook feature panel together with a small
well-posedness summary.

This is FCE research infrastructure. It makes NO economic claim, cites no
performance, and touches no market-data provider — it only reads a parquet file
path handed to it. Selection and any interpretation are human-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from liq.fractal import InputDomain
from liq.fractal.codec import CodecParams, codebook_features, encode_window

_STATUS = "DEV_SMOKE_ONLY"


def _default_params() -> CodecParams:
    """Default demo configuration; simple returns as the codec input domain."""

    return CodecParams(
        window=64,
        range_length=8,
        domain_factor=2,
        stride_domain=4,
        top_k=3,
        input_domain=InputDomain.RETURNS,
    )


def _simple_returns(closes: list[float]) -> list[float]:
    """Pairwise simple returns ``(c1 - c0) / c0`` from a close-price series."""

    return [
        (nxt - prev) / prev
        for prev, nxt in zip(closes[:-1], closes[1:], strict=False)
        if prev != 0.0
    ]


def _params_metadata(params: CodecParams) -> dict[str, object]:
    """Reproducibility-bearing codec configuration as plain JSON scalars."""

    return {
        "window": params.window,
        "range_length": params.range_length,
        "domain_factor": params.domain_factor,
        "stride_domain": params.stride_domain,
        "top_k": params.top_k,
        "boundary_tau": params.boundary_tau,
        "turnover_lag": params.turnover_lag,
        "hill_fraction": params.hill_fraction,
        "input_domain": params.input_domain.value,
        "domain_length": params.domain_length,
    }


def run_codec_demo(fixture_path: Path, *, params: CodecParams | None = None) -> dict[str, object]:
    """Encode the most recent window of fixture returns and summarise it.

    The returned mapping is diagnostic-only: it carries no economic claim,
    no performance figure, and defers all selection authority to a human.
    """

    resolved = params if params is not None else _default_params()

    frame = pl.read_parquet(fixture_path).sort("timestamp")
    closes = frame.get_column("close").cast(pl.Float64).to_list()
    returns = _simple_returns([float(value) for value in closes])

    window_returns = returns[-resolved.window :]
    codebook = encode_window(window_returns, resolved)
    features = codebook_features(codebook)

    hill_stable = features["residual_tail_index"] is not None

    return {
        "object_type": "fce_codec_demo",
        "status": _STATUS,
        "diagnostic_only": True,
        "claims": [],
        "fixture": {
            "path": str(fixture_path),
            "row_count": frame.height,
            "return_count": len(returns),
        },
        "codec_params": _params_metadata(resolved),
        "well_posedness": {
            "boundary_pressure": features["boundary_pressure"],
            "codebook_entropy": features["codebook_entropy"],
            "hill_stable": hill_stable,
            "residual_tail_fallback": features["residual_tail_fallback"],
        },
        "features": features,
        "selection_authority": "human_only",
    }


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the diagnostic-only codec demo CLI."""

    parser = argparse.ArgumentParser(
        prog="liq-features-codec-demo",
        description="Diagnostic-only fractal codec smoke demo (no economic claim).",
    )
    parser.add_argument("--fixture", type=Path, required=True, help="Parquet fixture path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Reserved diagnostic flag; the demo already writes no side effects.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the demo and print its diagnostic-only report as JSON."""

    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_codec_demo(args.fixture)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
