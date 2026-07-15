"""Causal fractal feature assembly for the FX JEPA selector head."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Final

import polars as pl

from liq.fractal import CodecParams, codebook_features, encode_window, estimate_dfa

_DFA_WINDOWS: Final = (30, 60, 120)
_D_FEATURE_NAMES: Final = (
    "scale_dispersion",
    "boundary_pressure",
    "offset_mean",
    "offset_dispersion",
    "residual_rms",
    "residual_tail_index",
    "codebook_entropy",
    "domain_concentration",
    "codebook_turnover",
    "topk_match_margin",
    "residual_hurst",
)

BLOCK_A_COLUMNS: Final = tuple(
    column
    for window in _DFA_WINDOWS
    for suffix in ("gt_055", "lt_045")
    for column in (f"a_dfa_{window}_{suffix}_value", f"a_dfa_{window}_{suffix}_state")
)
BLOCK_D_COLUMNS: Final = tuple(f"d_{name}" for name in _D_FEATURE_NAMES)
BLOCK_E_COLUMNS: Final = tuple(f"e_delta_{name}" for name in _D_FEATURE_NAMES)
FRACTAL_SELECTOR_COLUMNS: Final = BLOCK_A_COLUMNS + BLOCK_D_COLUMNS + BLOCK_E_COLUMNS


@dataclass(frozen=True, slots=True)
class FxJepaFractalSpec:
    """Frozen descriptor and codec parameters for the selector feature cache."""

    dfa_windows: tuple[int, ...] = _DFA_WINDOWS
    dfa_high_threshold: float = 0.55
    dfa_low_threshold: float = 0.45
    codec_window: int = 64
    codec_range_length: int = 8
    codec_domain_factor: int = 2
    codec_stride_domain: int = 4
    codec_top_k: int = 3
    sampling: str = "1m"

    def __post_init__(self) -> None:
        frozen = (
            _DFA_WINDOWS,
            0.55,
            0.45,
            64,
            8,
            2,
            4,
            3,
            "1m",
        )
        current = (
            self.dfa_windows,
            self.dfa_high_threshold,
            self.dfa_low_threshold,
            self.codec_window,
            self.codec_range_length,
            self.codec_domain_factor,
            self.codec_stride_domain,
            self.codec_top_k,
            self.sampling,
        )
        if current != frozen:
            raise ValueError("FX JEPA fractal parameters are frozen and cannot drift")

    def codec_params(self) -> CodecParams:
        """Build the alpha-safe canonical codec parameters."""

        return CodecParams(
            window=self.codec_window,
            range_length=self.codec_range_length,
            domain_factor=self.codec_domain_factor,
            stride_domain=self.codec_stride_domain,
            top_k=self.codec_top_k,
        )


@dataclass(frozen=True, slots=True)
class FractalFeatureBatch:
    """Feature rows and reproducibility metadata for one source series."""

    frame: pl.DataFrame
    schema_hash: str
    config_hash: str
    source_row_count: int
    exclusions: dict[str, int]


def build_fx_jepa_fractal_features(
    frame: pl.DataFrame,
    *,
    value_column: str = "midrange_ret",
    spec: FxJepaFractalSpec | None = None,
    evaluation_timestamps: Sequence[datetime] | None = None,
) -> FractalFeatureBatch:
    """Build Blocks A/D/E using only observations available at each timestamp."""

    resolved = spec or FxJepaFractalSpec()
    _validate_source(frame, value_column=value_column)
    values = [float(value) for value in frame[value_column].to_list()]
    timestamps = frame["timestamp"].to_list()
    codec_params = resolved.codec_params()
    evaluation_indices = _evaluation_indices(
        timestamps, evaluation_timestamps=evaluation_timestamps
    )
    previous_codebook = None
    previous_d: dict[str, float] | None = None
    rows: list[dict[str, object]] = []
    excluded_non_finite = 0
    missing_prior_state = 0

    for index in evaluation_indices:
        if index < codec_params.window - 1:
            continue
        codebook = encode_window(values[index - codec_params.window + 1 : index + 1], codec_params)
        raw_d = codebook_features(codebook, previous=previous_codebook)
        previous_codebook = codebook
        current_d = _numeric_codebook_features(raw_d)
        if index < max(resolved.dfa_windows):
            previous_d = current_d
            continue
        if current_d is None or previous_d is None:
            if current_d is None:
                excluded_non_finite += 1
            else:
                missing_prior_state += 1
            previous_d = current_d
            continue
        dfa = {
            window: estimate_dfa(values[index - window + 1 : index + 1])
            for window in resolved.dfa_windows
        }
        feature_values = _block_a(dfa, resolved=resolved)
        feature_values.update({f"d_{name}": current_d[name] for name in _D_FEATURE_NAMES})
        feature_values.update(
            {f"e_delta_{name}": current_d[name] - previous_d[name] for name in _D_FEATURE_NAMES}
        )
        if not all(math.isfinite(float(value)) for value in feature_values.values()):
            excluded_non_finite += 1
            previous_d = current_d
            continue
        timestamp = timestamps[index]
        rows.append(
            {
                "timestamp": timestamp,
                "feature_available_at": timestamp,
                **feature_values,
            }
        )
        previous_d = current_d

    output = pl.DataFrame(rows) if rows else _empty_feature_frame(frame.schema["timestamp"])
    return FractalFeatureBatch(
        frame=output.select("timestamp", "feature_available_at", *FRACTAL_SELECTOR_COLUMNS),
        schema_hash=_stable_hash({"columns": FRACTAL_SELECTOR_COLUMNS}),
        config_hash=_stable_hash(asdict(resolved)),
        source_row_count=frame.height,
        exclusions={
            "warmup": min(frame.height, max(resolved.dfa_windows)),
            "non_finite_feature": excluded_non_finite,
            "missing_prior_state": missing_prior_state,
        },
    )


def _evaluation_indices(
    timestamps: list[object], *, evaluation_timestamps: Sequence[datetime] | None
) -> tuple[int, ...]:
    if evaluation_timestamps is None:
        return tuple(range(len(timestamps)))
    requested = tuple(evaluation_timestamps)
    if requested != tuple(sorted(set(requested))):
        raise ValueError("evaluation_timestamps must be sorted and unique")
    by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps)}
    missing = [timestamp for timestamp in requested if timestamp not in by_timestamp]
    if missing:
        raise ValueError("evaluation_timestamps must exist in the source frame")
    return tuple(by_timestamp[timestamp] for timestamp in requested)


def _block_a(dfa: dict[int, float], *, resolved: FxJepaFractalSpec) -> dict[str, float]:
    features: dict[str, float] = {}
    for window, value in dfa.items():
        features[f"a_dfa_{window}_gt_055_value"] = value
        features[f"a_dfa_{window}_gt_055_state"] = float(value > resolved.dfa_high_threshold)
        features[f"a_dfa_{window}_lt_045_value"] = value
        features[f"a_dfa_{window}_lt_045_state"] = float(value < resolved.dfa_low_threshold)
    return features


def _numeric_codebook_features(raw: dict[str, object]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for name in _D_FEATURE_NAMES:
        value = raw.get(name)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return None
        values[name] = float(value)
    return values


def _validate_source(frame: pl.DataFrame, *, value_column: str) -> None:
    if value_column.lower() in {"open", "high", "low", "close", "price", "midprice"}:
        raise ValueError("raw-price inputs are forbidden for alpha-facing fractal features")
    missing = {"timestamp", value_column} - set(frame.columns)
    if missing:
        raise ValueError(f"feature source missing columns: {', '.join(sorted(missing))}")
    timestamp_dtype = frame.schema["timestamp"]
    if not isinstance(timestamp_dtype, pl.Datetime) or timestamp_dtype.time_zone != "UTC":
        raise ValueError("feature timestamps must be timezone-aware UTC")
    if frame["timestamp"].null_count() or frame[value_column].null_count():
        raise ValueError("feature source values must be finite and non-null")
    if not frame["timestamp"].is_sorted():
        raise ValueError("feature timestamps must be sorted")
    if frame["timestamp"].n_unique() != frame.height:
        raise ValueError("feature timestamps must be unique")
    if not all(math.isfinite(float(value)) for value in frame[value_column].to_list()):
        raise ValueError("feature source values must be finite")


def _empty_feature_frame(timestamp_dtype: pl.DataType) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "timestamp": timestamp_dtype,
        "feature_available_at": timestamp_dtype,
    }
    for column in FRACTAL_SELECTOR_COLUMNS:
        schema[column] = pl.Float64()
    return pl.DataFrame(schema=schema)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BLOCK_A_COLUMNS",
    "BLOCK_D_COLUMNS",
    "BLOCK_E_COLUMNS",
    "FRACTAL_SELECTOR_COLUMNS",
    "FractalFeatureBatch",
    "FxJepaFractalSpec",
    "build_fx_jepa_fractal_features",
]
