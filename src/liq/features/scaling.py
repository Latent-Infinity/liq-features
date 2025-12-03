"""Model-aware scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional


ModelType = Literal["tree", "nn", "transformer", "diffusion"]


@dataclass
class ScalingParams:
    mean: float
    std: float
    min_val: float
    max_val: float


class ModelAwareScaler:
    """Train-only scaling strategy aware of model family."""

    def __init__(self, model_type: ModelType) -> None:
        self.model_type = model_type
        self.params: Optional[ScalingParams] = None

    def fit(self, series: Iterable[float]) -> "ModelAwareScaler":
        values = list(series)
        if not values:
            raise ValueError("Cannot fit scaler on empty data")
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        min_val = min(values)
        max_val = max(values)
        self.params = ScalingParams(mean=mean, std=std or 1.0, min_val=min_val, max_val=max_val)
        return self

    def transform(self, series: Iterable[float]) -> list[float]:
        if self.params is None:
            raise RuntimeError("Scaler must be fit before transform")
        values = list(series)
        if self.model_type == "tree":
            return values
        if self.model_type in ("nn", "transformer"):
            return [(v - self.params.mean) / self.params.std for v in values]
        if self.model_type == "diffusion":
            span = self.params.max_val - self.params.min_val or 1.0
            return [((v - self.params.min_val) / span) * 2 - 1 for v in values]
        return values

    def fit_transform(self, series: Iterable[float]) -> list[float]:
        return self.fit(series).transform(series)
