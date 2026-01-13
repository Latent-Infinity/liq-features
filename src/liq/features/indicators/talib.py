"""Dynamic TA-Lib indicator wrapper.

This module provides a dynamic wrapper that makes ALL TA-Lib indicators available
without hardcoding each one. It introspects TA-Lib to discover available functions
and generates indicator classes on-the-fly.

Design Principles:
    - DRY: Single implementation for all TA-Lib indicators
    - Open/Closed: New TA-Lib functions automatically available
    - SRP: Input mapping, output handling, and class generation are separate concerns
    - Fail-Fast: Clear errors for unsupported indicators

Input Mapping Rules:
    TA-Lib indicators require different input types. This module maps DataFrame columns
    to TA-Lib input arrays according to these rules:

    1. 'prices' (list of columns):
       - Maps to the specific columns listed (e.g., ['high', 'low', 'close', 'volume'])
       - Examples: MFI needs ['high', 'low', 'close', 'volume']
                   BOP needs ['open', 'high', 'low', 'close']
                   AROON needs ['high', 'low']

    2. 'price' (single column):
       - Maps to configurable price column (default: 'close')
       - Can be overridden via '_price_column' param to use 'midrange', 'typical', etc.

    3. 'real', 'real0' (single value):
       - Maps to configurable price column (default: 'close')
       - Used by most single-input indicators (RSI, EMA, etc.)

    4. 'real1' (second value):
       - Maps to 'high' column if available, otherwise 'close'
       - Used by some two-input indicators

    5. 'price0', 'price1' (two price inputs):
       - The input_spec value specifies which column to use (e.g., 'high', 'low')
       - Example: {'price0': 'high', 'price1': 'low'} for BETA, CORREL
       - Falls back to configurable price column if spec column not found

    6. 'periods' (variable periods):
       - Skipped - handled by indicator parameters
       - Used by MAVP (Moving Average with Variable Period)

    7. Other keys:
       - Direct column lookup in DataFrame
       - Fails if column not found

Output Handling:
    TA-Lib indicators can return single or multiple outputs:

    - Single output: Returns DataFrame with ['ts', 'value']
    - Multi-output: Returns DataFrame with ['ts', 'output1', 'output2', ...]
      Examples: MACD returns ['macd', 'signal', 'histogram']
                AROON returns ['aroondown', 'aroonup']
                STOCH returns ['slowk', 'slowd']

Note:
    This module requires TA-Lib to be installed (optional dependency).
    Install with: pip install ta-lib

Example:
    >>> from liq.features.indicators.talib import get_dynamic_indicator
    >>> CCI = get_dynamic_indicator("CCI")
    >>> cci = CCI(params={"timeperiod": 14})
    >>> result = cci.compute(df)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any
import threading
import weakref

import numpy as np

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

try:
    import polars as pl
    import talib
    from talib import abstract

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    talib = None  # type: ignore[assignment]
    abstract = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]

from liq.features.indicators.base import BaseIndicator
from liq.features.numpy_utils import to_numpy_float64


def _check_talib() -> None:
    """Check if TA-Lib is available.

    Raises:
        ImportError: If TA-Lib is not installed.
    """
    if not HAS_TALIB:
        raise ImportError(
            "TA-Lib is required for dynamic indicators. "
            "Install with: pip install ta-lib"
        )


def get_available_indicators() -> list[str]:
    """Get list of all available TA-Lib indicators.

    Returns:
        List of indicator names (lowercase).

    Raises:
        ImportError: If TA-Lib is not installed.
    """
    _check_talib()
    return sorted([func.lower() for func in talib.get_functions()])


def get_indicator_info(name: str) -> dict[str, Any]:
    """Get metadata about a TA-Lib indicator.

    Args:
        name: Indicator name (case-insensitive).

    Returns:
        Dictionary with indicator metadata:
        - name: Indicator name (uppercase)
        - display_name: Human-readable name
        - group: Category (e.g., "Overlap Studies", "Momentum Indicators")
        - inputs: Required input array names
        - input_names: Detailed input specification (OrderedDict)
        - parameters: Dict of parameter names to default values
        - outputs: List of output names

    Raises:
        ImportError: If TA-Lib is not installed.
        ValueError: If indicator not found.
    """
    _check_talib()
    name_upper = name.upper()

    if name_upper not in talib.get_functions():
        available = ", ".join(get_available_indicators()[:10])
        raise ValueError(
            f"Unknown TA-Lib indicator: {name}. "
            f"Available indicators include: {available}... (150+ total). "
            f"Use get_available_indicators() for full list."
        )

    func = abstract.Function(name_upper)

    inputs = list(func.input_names.keys())

    parameters = {}
    if func.parameters:
        for param_name, param_info in func.parameters.items():
            if isinstance(param_info, dict):
                parameters[param_name] = param_info.get("default", None)
            else:
                parameters[param_name] = param_info

    outputs = list(func.output_names)

    return {
        "name": name_upper,
        "display_name": func.info.get("display_name", name_upper),
        "group": func.info.get("group", "Unknown"),
        "inputs": inputs,
        "input_names": func.input_names,
        "parameters": parameters,
        "outputs": outputs,
    }


def _to_float64(arr: "NDArray[Any]") -> "NDArray[np.floating[Any]]":
    """Convert array to float64 for TA-Lib compatibility.

    TA-Lib requires float64 arrays. This handles int and other numeric types.

    Args:
        arr: Input numpy array.

    Returns:
        Float64 numpy array.
    """
    if arr.dtype != np.float64:
        return arr.astype(np.float64, copy=False)
    return arr


_TALIB_INPUT_CACHE: dict[int, dict[str, "NDArray[np.floating[Any]]"]] = {}
_TALIB_FINALIZERS: dict[int, weakref.finalize] = {}
_TALIB_CACHE_LOCK = threading.Lock()


def _get_column_cache(
    df: "pl.DataFrame",
) -> dict[str, "NDArray[np.floating[Any]]"]:
    """Return a per-DataFrame cache for TA-Lib input arrays."""
    key = id(df)
    with _TALIB_CACHE_LOCK:
        cached = _TALIB_INPUT_CACHE.get(key)
        if cached is not None:
            return cached
        cache: dict[str, "NDArray[np.floating[Any]]"] = {}
        _TALIB_INPUT_CACHE[key] = cache
        _TALIB_FINALIZERS[key] = weakref.finalize(
            df,
            _purge_talib_cache,
            key,
        )
        return cache


def _purge_talib_cache(key: int) -> None:
    with _TALIB_CACHE_LOCK:
        _TALIB_INPUT_CACHE.pop(key, None)
        _TALIB_FINALIZERS.pop(key, None)


def map_inputs(
    df: "pl.DataFrame",
    input_names: "OrderedDict[str, Any]",
    price_column: str = "close",
    indicator_name: str = "unknown",
) -> dict[str, "NDArray[np.floating[Any]]"]:
    """Map DataFrame columns to TA-Lib input arrays.

    This function implements the input mapping rules for TA-Lib indicators.
    See module docstring for detailed mapping rules.

    All arrays are converted to float64 as required by TA-Lib.

    Args:
        df: DataFrame with OHLCV columns (ts, open, high, low, close, volume).
        input_names: TA-Lib input specification from abstract.Function.input_names.
        price_column: Column to use for single-price inputs (default: 'close').
            Can be 'close', 'midrange', 'typical', 'weighted_close'.
        indicator_name: Name of indicator for error messages.

    Returns:
        Dictionary mapping TA-Lib input names to numpy float64 arrays.

    Raises:
        ValueError: If required column is missing from DataFrame.

    Example:
        >>> input_names = OrderedDict({'prices': ['high', 'low', 'close', 'volume']})
        >>> inputs = map_inputs(df, input_names, indicator_name='MFI')
        >>> print(inputs.keys())
        dict_keys(['high', 'low', 'close', 'volume'])
    """
    inputs: dict[str, NDArray[np.floating[Any]]] = {}
    column_cache = _get_column_cache(df)

    def _get_col_array(col_name: str) -> "NDArray[np.floating[Any]]":
        cached = column_cache.get(col_name)
        if cached is not None:
            return cached
        arr = to_numpy_float64(df[col_name], allow_copy=False)
        with _TALIB_CACHE_LOCK:
            column_cache.setdefault(col_name, arr)
        return arr

    for input_key, input_spec in input_names.items():
        input_key_lower = input_key.lower()

        if input_key_lower == "prices":
            # 'prices' maps to a list of required columns
            # e.g., ['high', 'low', 'close', 'volume'] for MFI
            # e.g., ['open', 'high', 'low', 'close'] for BOP
            if isinstance(input_spec, list):
                for col in input_spec:
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        inputs[col_lower] = _to_float64(_get_col_array(col_lower))
                    else:
                        raise ValueError(
                            f"Indicator {indicator_name} requires '{col_lower}' column, "
                            f"but it's not in the DataFrame. Available: {df.columns}"
                        )
            else:
                # Fallback: map standard OHLC (shouldn't happen with well-formed input_names)
                for col in ["high", "low", "close"]:
                    if col in df.columns:
                        inputs[col] = _to_float64(_get_col_array(col))

        elif input_key_lower == "price":
            # Single price input - use configurable price column
            inputs["close"] = _to_float64(_get_col_array(price_column))

        elif input_key_lower in ("real", "real0"):
            # Real value input - use configurable price column
            inputs["close"] = _to_float64(_get_col_array(price_column))

        elif input_key_lower == "real1":
            # Second real input - use high if available
            if "high" in df.columns:
                inputs["high"] = _to_float64(_get_col_array("high"))
            else:
                inputs["close"] = _to_float64(_get_col_array(price_column))

        elif input_key_lower in ("price0", "price1"):
            # Two-input indicators (BETA, CORREL) - input_spec specifies the column
            # e.g., {'price0': 'high', 'price1': 'low'}
            if isinstance(input_spec, str) and input_spec.lower() in df.columns:
                col = input_spec.lower()
                inputs[col] = _to_float64(_get_col_array(col))
            elif price_column in df.columns:
                # Fallback to configurable price column
                inputs[price_column] = _to_float64(_get_col_array(price_column))
            else:
                raise ValueError(
                    f"Indicator {indicator_name} requires '{input_spec}' column, "
                    f"but it's not in the DataFrame. Available: {df.columns}"
                )

        elif input_key_lower == "periods":
            # Variable period input (for MAVP) - skip, handled by params
            continue

        else:
            # Direct column lookup
            if input_key_lower in df.columns:
                inputs[input_key_lower] = _to_float64(_get_col_array(input_key_lower))
            else:
                raise ValueError(
                    f"Indicator {indicator_name} requires '{input_key_lower}' column, "
                    f"but it's not in the DataFrame. Available: {df.columns}"
                )

    return inputs


def format_outputs(
    ts_series: "pl.Series",
    result_arrays: Any,
    output_names: list[str],
) -> "pl.DataFrame":
    """Format TA-Lib output arrays into a Polars DataFrame.

    Handles single-output and multi-output indicators, as well as
    different return types (ndarray, tuple, list).

    Args:
        ts_series: Timestamp series from input DataFrame.
        result_arrays: Output from TA-Lib function call.
        output_names: List of output names from indicator info.

    Returns:
        DataFrame with timestamp and output columns, filtered to remove NaN rows.

    Example:
        >>> # Single output
        >>> result = format_outputs(df['ts'], rsi_result, ['real'])
        >>> print(result.columns)
        ['ts', 'value']

        >>> # Multi output
        >>> result = format_outputs(df['ts'], macd_result, ['macd', 'signal', 'histogram'])
        >>> print(result.columns)
        ['ts', 'macd', 'signal', 'histogram']
    """
    if len(output_names) == 1:
        # Single output - normalize to array
        if isinstance(result_arrays, np.ndarray):
            values = result_arrays
        elif isinstance(result_arrays, (tuple, list)):
            values = result_arrays[0]
        else:
            values = result_arrays

        result = pl.DataFrame({"ts": ts_series, "value": values})
        return result.filter(pl.col("value").is_not_nan())

    else:
        # Multi-output
        result_dict: dict[str, Any] = {"ts": ts_series}

        # Handle different return types (tuple, list, or single array)
        if isinstance(result_arrays, (tuple, list)):
            result_list = list(result_arrays)
        else:
            result_list = [result_arrays]

        # Handle case where outputs and result_arrays lengths differ
        n_outputs = min(len(output_names), len(result_list))
        for i in range(n_outputs):
            output_name = output_names[i]
            values = result_list[i]
            result_dict[output_name.lower()] = values

        result = pl.DataFrame(result_dict)

        # Filter NaN rows - only filter on columns we actually added
        for i in range(n_outputs):
            output_name = output_names[i]
            result = result.filter(pl.col(output_name.lower()).is_not_nan())

        return result


def _create_dynamic_indicator_class(name: str) -> type[BaseIndicator]:
    """Create a dynamic indicator class for a TA-Lib function.

    Args:
        name: Indicator name (case-insensitive).

    Returns:
        Dynamically created indicator class.

    Raises:
        ImportError: If TA-Lib is not installed.
        ValueError: If indicator not found or unsupported.
    """
    _check_talib()
    info = get_indicator_info(name)
    name_upper = info["name"]
    name_lower = name_upper.lower()

    func = abstract.Function(name_upper)
    params_dict = info["parameters"].copy()

    class DynamicIndicator(BaseIndicator):
        """Dynamically generated TA-Lib indicator."""

        name = name_lower
        default_params = params_dict

        _indicator_info = info
        _talib_func = func

        def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
            """Compute indicator using TA-Lib.

            Args:
                df: DataFrame with OHLCV columns.

            Returns:
                DataFrame with timestamp and indicator value(s).
            """
            # Get price column for single-value inputs
            price_col = self._params.get("_price_column", "close")
            if price_col not in df.columns:
                price_col = "close"

            # Map inputs using the extracted function
            inputs = map_inputs(
                df,
                self._talib_func.input_names,
                price_column=price_col,
                indicator_name=self.name,
            )

            # Call TA-Lib function
            result_arrays = self._talib_func(inputs, **self.params)

            # Format outputs using the extracted function
            return format_outputs(
                df["ts"],
                result_arrays,
                self._indicator_info["outputs"],
            )

    DynamicIndicator.__name__ = name_upper
    DynamicIndicator.__qualname__ = name_upper
    DynamicIndicator.__doc__ = f"""{info["display_name"]} (TA-Lib {name_upper}).

    Group: {info["group"]}

    Inputs: {", ".join(info["inputs"])}
    Parameters: {", ".join(f"{k}={v}" for k, v in params_dict.items())}
    Outputs: {", ".join(info["outputs"])}

    This is a dynamically generated indicator class that wraps TA-Lib's {name_upper} function.
    """

    return DynamicIndicator


_indicator_class_cache: dict[str, type[BaseIndicator]] = {}


def get_dynamic_indicator(name: str) -> type[BaseIndicator]:
    """Get or create a dynamic indicator class for a TA-Lib function.

    This function provides lazy loading of indicator classes. Classes are
    created on first access and cached for subsequent use.

    Args:
        name: Indicator name (case-insensitive).

    Returns:
        Indicator class that can be instantiated.

    Raises:
        ImportError: If TA-Lib is not installed.
        ValueError: If indicator not found or unsupported.

    Example:
        >>> CCI = get_dynamic_indicator("CCI")
        >>> cci = CCI(params={"timeperiod": 20})
        >>> result = cci.compute(df)
    """
    name_lower = name.lower()

    if name_lower in _indicator_class_cache:
        return _indicator_class_cache[name_lower]

    indicator_class = _create_dynamic_indicator_class(name)
    _indicator_class_cache[name_lower] = indicator_class

    return indicator_class


def clear_indicator_cache() -> None:
    """Clear the indicator class cache.

    Useful for testing or when TA-Lib configuration changes.
    """
    _indicator_class_cache.clear()


def list_dynamic_indicators() -> list[dict[str, Any]]:
    """List all available TA-Lib indicators with metadata.

    Returns:
        List of dicts with indicator information.

    Raises:
        ImportError: If TA-Lib is not installed.
    """
    _check_talib()
    result = []

    for func_name in talib.get_functions():
        try:
            info = get_indicator_info(func_name)
            result.append(
                {
                    "name": func_name.lower(),
                    "display_name": info["display_name"],
                    "group": info["group"],
                    "inputs": info["inputs"],
                    "parameters": info["parameters"],
                    "outputs": info["outputs"],
                }
            )
        except Exception:
            continue

    return result
