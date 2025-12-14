"""Base indicator abstract class.

This module provides the abstract base class for all technical indicators,
implementing the Template Method pattern for consistent indicator behavior.

Design Principles:
    - SRP: Base class handles common behavior, subclasses implement specifics
    - OCP: New indicators extend without modifying base
    - DIP: Depends on abstractions (storage protocol)
    - Template Method: Defines compute skeleton, subclasses implement _compute

Example:
    >>> class MyIndicator(BaseIndicator):
    ...     name = "my_indicator"
    ...     default_params = {"period": 14}
    ...
    ...     def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
    ...         # Implement indicator logic
    ...         return result_df
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, TYPE_CHECKING

import polars as pl
from liq.store import key_builder
from liq.store.protocols import TimeSeriesStore

from liq.features.params import hash_params

if TYPE_CHECKING:
    from liq.data.service import DataService


class BaseIndicator(ABC):
    """Abstract base class for technical indicators.

    Provides the Template Method pattern for indicator computation with
    optional caching through FactorStorage.

    Class Attributes:
        name: Indicator name (lowercase, e.g., "rsi", "ema")
        default_params: Default parameters dictionary

    Instance Attributes:
        params: Merged parameters (defaults + custom)

    Subclasses must implement:
        - name: Class attribute with indicator name
        - default_params: Class attribute with default parameters
        - _compute: Method that performs the actual calculation
    """

    name: ClassVar[str]
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        storage: TimeSeriesStore | None = None,
        input_column: str = "close",
        data_service: "DataService | None" = None,
    ) -> None:
        """Initialize indicator with parameters and optional storage.

        Args:
            params: Custom parameters to override defaults
            storage: Optional FactorStorage for caching results
            data_service: Optional liq-data service (its store is used when storage is omitted)
        """
        self._params = {**self.default_params, **(params or {})}
        self._storage = storage or (data_service.store if data_service else None)
        self._input_column = input_column

    @property
    def params(self) -> dict[str, Any]:
        """Get the merged parameters."""
        return self._params

    @property
    def storage(self) -> Any | None:
        """Get the storage backend."""
        return self._storage

    def compute(
        self,
        df: pl.DataFrame,
        symbol: str | None = None,
        timeframe: str | None = None,
        use_cache: bool = True,
        component: str | None = None,
        force_recalculate: bool = False,
    ) -> pl.DataFrame:
        """Compute indicator values with optional caching.

        This is the Template Method that defines the computation flow:
        1. Check cache if storage is configured
        2. Compute values using _compute()
        3. Store results if storage is configured

        Args:
            df: Input DataFrame with OHLCV data
            symbol: Optional symbol for cache key
            timeframe: Optional timeframe for cache key
            use_cache: Whether to use cache (default: True)
            component: Optional component column to return (e.g. macd histogram)
            force_recalculate: Bypass cache when True

        Returns:
            DataFrame with computed indicator values
            Typically has columns: ts, value (or multiple output columns)
        """
        df_for_compute = df
        if self._input_column != "close" and self._input_column in df.columns:
            df_for_compute = df.with_columns(pl.col(self._input_column).alias("close"))

        cache_key = None
        if (
            self._storage is not None
            and use_cache
            and symbol
            and timeframe
        ):
            params_hash = hash_params(
                {**self._params, "input": self._input_column, "component": component or "all"}
            )
            cache_key = key_builder.indicators(symbol, self.name, f"{params_hash}:{timeframe}")

            if not force_recalculate and self._storage.exists(cache_key):
                cached = self._storage.read(cache_key)
                if cached.height > 0:
                    return self._select_component(cached, component)

        result = self._compute(df_for_compute)
        if self._storage is not None and use_cache and cache_key:
            self._storage.write(cache_key, result, mode="overwrite")

        return self._select_component(result, component)

    @abstractmethod
    def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute indicator values (to be implemented by subclasses).

        Args:
            df: Input DataFrame with required columns (typically OHLCV)

        Returns:
            DataFrame with columns: ts, value
            For multi-output indicators: ts, output1, output2, ...

        Note:
            - Must filter out NaN values from warmup period
            - Should use polars expressions for efficiency
        """
        ...

    def __repr__(self) -> str:
        """Return string representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"

    @staticmethod
    def _select_component(df: pl.DataFrame, component: str | None) -> pl.DataFrame:
        """Return a specific component column when requested."""
        if component is None or component not in df.columns:
            return df
        return df.select([col for col in df.columns if col == component or col == "ts" or col == "timestamp"])
