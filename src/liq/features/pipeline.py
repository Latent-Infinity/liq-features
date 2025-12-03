"""Pipeline bundling for stationarity and scaling with persistence hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from liq.features.scaling import ModelAwareScaler, ModelType, ScalingParams
from liq.features.stationarity import StationarityTransformer


@dataclass
class PipelineState:
    model_type: ModelType
    d: float
    scaling_params: ScalingParams | None
    fitted: bool = False


class FeaturePipeline:
    """Train-only fit pipeline that applies FFD then scaling."""

    def __init__(self, model_type: ModelType, d: float = 0.4) -> None:
        self.stationarity = StationarityTransformer(d=d)
        self.scaler = ModelAwareScaler(model_type=model_type)
        self.state = PipelineState(model_type=model_type, d=d, scaling_params=None, fitted=False)

    def fit(self, series: Iterable[float]) -> "FeaturePipeline":
        series_list = list(series)
        self.stationarity.fit(series_list)
        self.scaler.fit(series_list)
        self.state.scaling_params = self.scaler.params
        self.state.fitted = True
        return self

    def transform(self, series: Iterable[float]) -> list[float]:
        if not self.state.fitted:
            raise RuntimeError("Pipeline must be fit before transform")
        stationarized = self.stationarity.transform(series)
        return self.scaler.transform(stationarized)

    def fit_transform(self, series: Iterable[float]) -> list[float]:
        return self.fit(series).transform(series)

    def to_dict(self) -> dict:
        if not self.state.fitted:
            raise RuntimeError("Cannot serialize unfitted pipeline")
        return {
            "model_type": self.state.model_type,
            "d": self.state.d,
            "scaling_params": asdict(self.state.scaling_params) if self.state.scaling_params else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeaturePipeline":
        pipe = cls(model_type=data["model_type"], d=data["d"])
        if params := data.get("scaling_params"):
            pipe.scaler.params = ScalingParams(**params)
            pipe.state.scaling_params = pipe.scaler.params
            pipe.state.fitted = True
            pipe.stationarity.fitted_ = True
        return pipe
