"""Feature definitions and sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

FeatureCallable = Callable[..., object]


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    func: FeatureCallable
    tier: str = "primary"
    lookback: int = 0
    input_column: str = "close"
    dependencies: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: list[FeatureDefinition] = field(default_factory=list)

    @property
    def max_lookback(self) -> int:
        return max((f.lookback for f in self.features), default=0)
