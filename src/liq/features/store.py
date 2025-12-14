"""FeatureStore: cache-first feature orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import polars as pl
from liq.store import key_builder
from liq.store.protocols import TimeSeriesStore

from liq.features.feature_set import FeatureDefinition, FeatureSet
from liq.features.derived import compute_derived_fields
from liq.features.params import hash_params
from liq.features.cache import get_data_hash

if TYPE_CHECKING:
    from liq.data.service import DataService


@dataclass
class FeatureStore:
    storage: TimeSeriesStore | None = None
    data_service: "DataService | None" = None

    def __post_init__(self) -> None:
        if self.storage is None and self.data_service is not None:
            self.storage = self.data_service.store
        if self.storage is None:
            raise ValueError("FeatureStore requires a storage backend or data_service.store")

    def _indicator_key(
        self,
        symbol: str,
        indicator: str,
        params: dict[str, Any],
        input_column: str,
        timeframe: str,
        dependencies: list[str] | None = None,
        lookback: int = 0,
        data_hash: str | None = None,
    ) -> str:
        params_hash = hash_params(
            {
                **params,
                "input": input_column,
                "dependencies": sorted(dependencies or []),
                "lookback": lookback,
                "data": data_hash or "na",
            }
        )
        return key_builder.indicators(symbol, indicator, f"{params_hash}:{timeframe}")

    def compute_feature(
        self,
        df: pl.DataFrame,
        definition: FeatureDefinition,
        symbol: str,
        timeframe: str,
        force_recalculate: bool = False,
    ) -> pl.DataFrame:
        params_hash = hash_params({"input": definition.input_column})
        data_hash = get_data_hash(df)
        key = self._indicator_key(
            symbol,
            definition.name,
            {"p": params_hash},
            definition.input_column,
            timeframe,
            dependencies=definition.dependencies,
            lookback=definition.lookback,
            data_hash=data_hash,
        )

        if not force_recalculate and self.storage and self.storage.exists(key):
            cached = self.storage.read(key)
            if not cached.is_empty():
                return cached

        result = definition.func(df, definition.input_column)
        if definition.lookback and result.height > definition.lookback:
            result = result.slice(definition.lookback)
        if self.storage:
            self.storage.write(key, result, mode="overwrite")
        return result

    def compute_feature_set(
        self,
        df: pl.DataFrame,
        feature_set: FeatureSet,
        symbol: str,
        timeframe: str,
        force_recalculate: bool = False,
    ) -> dict[str, pl.DataFrame]:
        df_with_derived = compute_derived_fields(df)
        outputs: dict[str, pl.DataFrame] = {}

        remaining = {f.name: f for f in feature_set.features}
        while remaining:
            progressed = False
            for name, feature in list(remaining.items()):
                if any(dep not in outputs for dep in feature.dependencies):
                    continue

                df_input = df_with_derived
                if feature.dependencies:
                    for dep in feature.dependencies:
                        dep_df = outputs[dep]
                        dep_col = dep_df.columns[-1]
                        df_input = df_input.join(
                            dep_df.rename({dep_col: dep}),
                            on="timestamp",
                            how="left",
                        )

                outputs[name] = self.compute_feature(
                    df_input, feature, symbol, timeframe, force_recalculate=force_recalculate
                )
                progressed = True
                remaining.pop(name)

            if not progressed:
                unresolved = ", ".join(remaining)
                raise ValueError(f"Circular or missing dependencies in FeatureSet: {unresolved}")

        # Optionally store combined feature set
        combined = None
        for name, feat_df in outputs.items():
            if combined is None:
                last_col = feat_df.columns[-1]
                combined = feat_df.rename({"value": name}) if "value" in feat_df.columns else feat_df.rename({last_col: name})
            else:
                combined = combined.join(
                    feat_df.rename({feat_df.columns[-1]: name}),
                    on="timestamp",
                    how="inner",
                )

        if combined is not None:
            max_lookback = feature_set.max_lookback
            if max_lookback > 0 and combined.height > max_lookback:
                combined = combined.slice(max_lookback)

            set_key = key_builder.features(symbol, f"{feature_set.name}:{timeframe}")
            if self.storage:
                self.storage.write(set_key, combined, mode="overwrite")

        return outputs
