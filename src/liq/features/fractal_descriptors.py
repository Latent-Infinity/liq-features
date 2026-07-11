"""Alpha-safe feature builders for cheap fractal descriptors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from liq.fractal import (
    FractalDescriptorSpec,
    FractalEstimate,
    estimate_dfa,
    estimate_ghe,
    estimate_hurst_rs,
)
from liq.fractal.contracts import MetadataValue

_SHORT_WINDOW_MAX = 120


class DescriptorInterpretation(StrEnum):
    """Interpretation attached to an alpha-facing descriptor artifact."""

    ROUGHNESS_PATH_EFFICIENCY = "roughness_path_efficiency"
    HURST_EXPONENT = "hurst_exponent"


@dataclass(frozen=True, slots=True)
class DescriptorFeature:
    """One descriptor estimate plus its diagnostic interpretation."""

    estimate: FractalEstimate
    interpretation: DescriptorInterpretation
    diagnostics_passed: bool

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return immutable artifact metadata including availability semantics."""

        metadata = dict(self.estimate.metadata())
        metadata["interpretation"] = self.interpretation.value
        metadata["diagnostics_passed"] = self.diagnostics_passed
        return MappingProxyType(metadata)


def build_cheap_descriptor_features(
    values: Sequence[float],
    *,
    specs: Sequence[FractalDescriptorSpec],
    feature_timestamp_ns: int,
    feature_available_at_ns: int,
    diagnostics_passed: bool = False,
) -> tuple[DescriptorFeature, ...]:
    """Build H/R-S, DFA, and GHE artifacts from return or volatility inputs."""

    features: list[DescriptorFeature] = []
    for spec in specs:
        if not spec.is_alpha_safe:
            raise ValueError("alpha-facing descriptor builders reject raw-price inputs")
        if len(values) < spec.window:
            raise ValueError(
                f"descriptor {spec.estimator!r} requires {spec.window} values, got {len(values)}"
            )
        estimator = _ESTIMATORS.get(spec.estimator)
        if estimator is None:
            raise ValueError(f"unsupported estimator: {spec.estimator!r}")
        estimate = FractalEstimate(
            spec=spec,
            value=estimator(values[-spec.window :]),
            feature_timestamp_ns=feature_timestamp_ns,
            feature_available_at_ns=feature_available_at_ns,
        )
        interpretation = (
            DescriptorInterpretation.ROUGHNESS_PATH_EFFICIENCY
            if spec.window <= _SHORT_WINDOW_MAX and not diagnostics_passed
            else DescriptorInterpretation.HURST_EXPONENT
        )
        features.append(DescriptorFeature(estimate, interpretation, diagnostics_passed))
    return tuple(features)


_ESTIMATORS = {
    "hurst_rs": estimate_hurst_rs,
    "dfa": estimate_dfa,
    "ghe": estimate_ghe,
}


__all__ = [
    "DescriptorFeature",
    "DescriptorInterpretation",
    "build_cheap_descriptor_features",
]
