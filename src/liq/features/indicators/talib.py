"""Dynamic TA-Lib indicator wrapper.

This module provides a dynamic wrapper that makes ALL TA-Lib indicators available
without hardcoding each one. It introspects TA-Lib to discover available functions
and generates indicator classes on-the-fly.

Design Principles:
    - DRY: Single implementation for all TA-Lib indicators
    - Open/Closed: New TA-Lib functions automatically available
    - Fail-Fast: Clear errors for unsupported indicators

Note:
    This module requires TA-Lib to be installed (optional dependency).
    Install with: pip install ta-lib

Example:
    >>> from liq.features.indicators.talib import get_dynamic_indicator
    >>> CCI = get_dynamic_indicator("CCI")
    >>> cci = CCI(params={"timeperiod": 14})
    >>> result = cci.compute(df)
"""

from typing import Any

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


def _check_talib() -> None:
    """Check if TA-Lib is available."""
    if not HAS_TALIB:
        raise ImportError(
            "TA-Lib is required for dynamic indicators. "
            "Install with: pip install ta-lib"
        )


def get_available_indicators() -> list[str]:
    """Get list of all available TA-Lib indicators.

    Returns:
        List of indicator names (lowercase)

    Raises:
        ImportError: If TA-Lib is not installed
    """
    _check_talib()
    return sorted([func.lower() for func in talib.get_functions()])


def get_indicator_info(name: str) -> dict[str, Any]:
    """Get metadata about a TA-Lib indicator.

    Args:
        name: Indicator name (case-insensitive)

    Returns:
        Dictionary with indicator metadata:
        - name: Indicator name
        - display_name: Full name
        - group: Category (e.g., "Overlap Studies", "Momentum Indicators")
        - inputs: Required input arrays
        - parameters: Dict of parameter names to default values
        - outputs: List of output names

    Raises:
        ImportError: If TA-Lib is not installed
        ValueError: If indicator not found
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
        "parameters": parameters,
        "outputs": outputs,
    }


def _create_dynamic_indicator_class(name: str) -> type[BaseIndicator]:
    """Create a dynamic indicator class for a TA-Lib function.

    Args:
        name: Indicator name (case-insensitive)

    Returns:
        Dynamically created indicator class

    Raises:
        ImportError: If TA-Lib is not installed
        ValueError: If indicator not found or unsupported
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
            """Compute indicator using TA-Lib."""
            import numpy as np

            inputs = {}
            for input_name in self._indicator_info["inputs"]:
                input_name_lower = input_name.lower()

                if input_name_lower == "prices":
                    inputs["high"] = df["high"].to_numpy()
                    inputs["low"] = df["low"].to_numpy()
                    inputs["close"] = df["close"].to_numpy()
                elif input_name_lower == "price":
                    inputs[input_name] = df["close"].to_numpy()
                else:
                    if input_name_lower not in df.columns:
                        raise ValueError(
                            f"Indicator {self.name} requires '{input_name_lower}' column, "
                            f"but it's not in the DataFrame. Available: {df.columns}"
                        )
                    inputs[input_name] = df[input_name_lower].to_numpy()

            result_arrays = self._talib_func(inputs, **self.params)

            outputs = self._indicator_info["outputs"]

            if len(outputs) == 1:
                if isinstance(result_arrays, np.ndarray):
                    values = result_arrays
                else:
                    values = (
                        result_arrays[0]
                        if isinstance(result_arrays, tuple)
                        else result_arrays
                    )

                result = pl.DataFrame({"ts": df["ts"], "value": values})
                return result.filter(pl.col("value").is_not_nan())

            else:
                result_dict = {"ts": df["ts"]}

                if not isinstance(result_arrays, tuple):
                    result_arrays = (result_arrays,)

                for output_name, values in zip(outputs, result_arrays, strict=True):
                    result_dict[output_name.lower()] = values

                result = pl.DataFrame(result_dict)

                for output_name in outputs:
                    result = result.filter(pl.col(output_name.lower()).is_not_nan())

                return result

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
        name: Indicator name (case-insensitive)

    Returns:
        Indicator class that can be instantiated

    Raises:
        ImportError: If TA-Lib is not installed
        ValueError: If indicator not found or unsupported

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


def list_dynamic_indicators() -> list[dict[str, Any]]:
    """List all available TA-Lib indicators with metadata.

    Returns:
        List of dicts with indicator information

    Raises:
        ImportError: If TA-Lib is not installed
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
