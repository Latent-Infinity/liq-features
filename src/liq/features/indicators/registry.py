"""Indicator registry for discovering and instantiating indicators.

This module provides a central registry for looking up indicator classes
by name, combining hardcoded indicators with optional dynamic liq-ta fallback.

Design Principles:
    - SRP: Only handles indicator discovery and instantiation
    - OCP: New indicators can be added without modifying existing code
    - DRY: Central lookup logic for all indicator access

Example:
    >>> from liq.features.indicators.registry import get_indicator, list_indicators
    >>>
    >>> RSI = get_indicator("rsi")
    >>> rsi = RSI(params={"period": 14})
    >>>
    >>> available = list_indicators()
"""

from collections.abc import Iterable, Mapping
import math
from typing import Any

from liq.features.indicators.base import BaseIndicator

# Registry of hardcoded indicators
_HARDCODED_INDICATORS: dict[str, type[BaseIndicator]] = {}

# Store original default_params for reset functionality
_ORIGINAL_DEFAULTS: dict[str, dict[str, Any]] = {}
_HARD_CODED_OUTPUTS: dict[str, list[str]] = {
    "bbands": ["upper", "middle", "lower"],
    "macd": ["macd", "signal", "histogram"],
    "adx": ["adx", "plus_di", "minus_di"],
    "adx_midrange": ["adx", "plus_di", "minus_di"],
    "stochastic": ["stoch_k", "stoch_d"],
    "stochastic_midrange": ["stoch_k", "stoch_d"],
}
_COLUMN_PARAM_KEYS = ("input_column", "column")
_HARD_CODED_INPUTS: dict[str, list[str]] = {
    "atr": ["high", "low", "close"],
    "adx": ["high", "low", "close"],
    "stochastic": ["high", "low", "close"],
    "atr_midrange": ["high", "low", "midrange"],
    "adx_midrange": ["high", "low", "midrange"],
    "stochastic_midrange": ["high", "low", "midrange"],
    "abnormal_turnover": ["volume"],
    "normalized_volume": ["volume"],
}


def register_indicator(cls: type[BaseIndicator]) -> type[BaseIndicator]:
    """Decorator to register an indicator class.

    Args:
        cls: Indicator class to register

    Returns:
        The same class (for use as decorator)

    Example:
        >>> @register_indicator
        ... class MyIndicator(BaseIndicator):
        ...     name = "my_indicator"
        ...     ...
    """
    _HARDCODED_INDICATORS[cls.name] = cls
    # Store original defaults for reset functionality
    _ORIGINAL_DEFAULTS[cls.name] = cls.default_params.copy()
    return cls


def get_indicator(name: str) -> type[BaseIndicator]:
    """Get an indicator class by name.

    First checks hardcoded indicators, then falls back to dynamic liq-ta
    indicators if available.

    Args:
        name: Indicator name (case-insensitive)

    Returns:
        Indicator class

    Raises:
        ValueError: If indicator not found

    Example:
        >>> RSI = get_indicator("rsi")
        >>> rsi = RSI(params={"period": 14})
    """
    name_lower = name.lower()

    # Check hardcoded indicators first
    if name_lower in _HARDCODED_INDICATORS:
        return _HARDCODED_INDICATORS[name_lower]

    # Try dynamic liq-ta fallback
    try:
        from liq.features.indicators.liq_ta import get_dynamic_indicator

        return get_dynamic_indicator(name)
    except ImportError:
        pass
    except ValueError:
        pass

    # List available indicators in error message
    available = list(_HARDCODED_INDICATORS.keys())[:10]
    raise ValueError(
        f"Unknown indicator: {name}. "
        f"Available: {available}... "
        f"Use list_indicators() for full list."
    )


def _sorted_discrete_values(values: Any) -> list[int | float] | None:
    """Normalize and sort discrete parameter values for metadata output."""
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return None

    cleaned = [v for v in values if not isinstance(v, bool)]
    if not cleaned:
        return None

    numeric_values = []
    for value in cleaned:
        if isinstance(value, (list, tuple, dict)):
            return None
        if not isinstance(value, (int, float)):
            return None
        numeric_values.append(float(value) if isinstance(value, bool) else value)

    if not all(math.isfinite(float(v)) for v in numeric_values):
        return None

    deduped = sorted(set(numeric_values))
    # Preserve int-ness when all values are integral.
    if all(float(value).is_integer() for value in deduped):
        return [int(value) for value in deduped]
    return [float(value) for value in deduped]


def _get_output_column_names(indicator_name: str) -> list[str]:
    """Get output column names for metadata generation."""
    return _HARD_CODED_OUTPUTS.get(indicator_name, ["value"])


def _infer_input_columns(cls: type[BaseIndicator]) -> list[str]:
    """Infer default inputs for a hardcoded indicator class."""
    name = cls.name.lower()
    if name in _HARD_CODED_INPUTS:
        return _HARD_CODED_INPUTS[name][:]

    defaults = cls.default_params
    for key in _COLUMN_PARAM_KEYS:
        value = defaults.get(key)
        if isinstance(value, str):
            return [value]
    return ["close"]


def _build_metadata_parameter_specs(
    indicator_name: str,
    default_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build stable parameter metadata entries."""
    try:
        from liq.features.indicators.param_grids import get_param_grid
    except Exception:
        get_param_grid = None

    allowed_grid = {}
    if get_param_grid is not None:
        try:
            allowed_grid = get_param_grid(indicator_name)
        except Exception:
            allowed_grid = {}

    parameters: list[dict[str, Any]] = []
    for param_name, default_value in default_params.items():
        allowed = _sorted_discrete_values(allowed_grid.get(param_name, None)) or []
        if default_value not in allowed and allowed and isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
            if isinstance(default_value, float):
                allowed = sorted(set([*allowed, float(default_value)]))
            else:
                allowed = sorted(set([*allowed, int(default_value)]))

        dtype_name = type(default_value).__name__
        if isinstance(default_value, bool):
            dtype_name = "bool"
        elif not isinstance(default_value, (int, float)):
            dtype_name = "str"

        parameters.append(
            {
                "name": param_name,
                "dtype": dtype_name,
                "default": default_value,
                "allowed_values": allowed if allowed else None,
            }
        )

    return parameters


def _metadata_from_hardcoded(name: str) -> dict[str, Any]:
    cls = _HARDCODED_INDICATORS[name]
    default_params = dict(cls.default_params)
    input_columns = _infer_input_columns(cls)
    return {
        "name": name,
        "display_name": cls.__name__,
        "source": "hardcoded",
        "group": "hardcoded",
        "default_params": default_params,
        "inputs": input_columns,
        "outputs": _get_output_column_names(name),
        "input_names": {name_: [name_] for name_ in input_columns},
        "parameters": _build_metadata_parameter_specs(name, default_params),
    }


def _metadata_from_dynamic(info: dict[str, Any]) -> dict[str, Any]:
    name = str(info.get("name", "")).lower()
    display_name = info.get("display_name", name)
    default_params = dict(info.get("parameters", {}))
    inputs = list(info.get("inputs", ["close"]))
    output_names = list(info.get("outputs", ["value"]))
    input_names = info.get("input_names", {})

    normalized_input_names = {}
    if isinstance(input_names, Mapping):
        for key, value in input_names.items():
            if isinstance(value, list):
                normalized_input_names[str(key)] = value
            else:
                normalized_input_names[str(key)] = [str(value)]

    if not normalized_input_names:
        normalized_input_names = {name: [str(name)]}

    return {
        "name": name,
        "display_name": display_name,
        "source": "liq_ta",
        "group": info.get("group", "liq_ta"),
        "default_params": default_params,
        "inputs": inputs,
        "outputs": output_names,
        "input_names": normalized_input_names,
        "parameters": _build_metadata_parameter_specs(name, default_params),
    }


def get_indicator_metadata(name: str) -> dict[str, Any]:
    """Get canonical metadata for an indicator.

    Returns:
        Dictionary with:
        - name
        - display_name
        - source ("hardcoded" | "liq_ta")
        - default_params
        - parameters: list[dict] with name, dtype, default, allowed_values
    """
    name_lower = name.lower()
    if name_lower in _HARDCODED_INDICATORS:
        metadata = _metadata_from_hardcoded(name_lower)
    else:
        try:
            from liq.features.indicators.liq_ta import (
                get_indicator_metadata as get_dynamic_indicator_metadata,
            )

            ta_info = get_dynamic_indicator_metadata(name_lower)
        except Exception:
            ta_info = None

        if ta_info is None:
            info_map = {entry["name"]: entry for entry in list_indicators()}
            if name_lower in info_map:
                metadata = info_map[name_lower]
            else:
                available = ", ".join(sorted(info_map)[:10]) if info_map else "<empty>"
                raise ValueError(f"Unknown indicator: {name}. Available: {available}...")
        else:
            metadata = _metadata_from_dynamic(ta_info)

    return metadata


def list_indicators() -> list[dict[str, Any]]:
    """List all available indicators with metadata.

    Returns:
        List of dicts with indicator info:
        - name: Indicator code name
        - display_name: Human-readable name
        - default_params: Default parameters
        - source: "hardcoded" or "liq_ta"

    Example:
        >>> indicators = list_indicators()
        >>> for ind in indicators[:3]:
        ...     print(f"{ind['name']}: {ind['display_name']}")
    """
    result = [_metadata_from_hardcoded(name) for name in sorted(_HARDCODED_INDICATORS)]

    # Add dynamic liq-ta indicators if available
    try:
        from liq.features.indicators.liq_ta import list_dynamic_indicators

        for info in list_dynamic_indicators():
            # Skip if already in hardcoded
            name = str(info.get("name", "")).lower()
            if name and name not in _HARDCODED_INDICATORS:
                result.append(_metadata_from_dynamic(info))
    except ImportError:
        pass

    return result


def get_registered_indicators() -> dict[str, type[BaseIndicator]]:
    """Get dictionary of all registered hardcoded indicators.

    Returns:
        Dictionary mapping indicator names to classes
    """
    return dict(_HARDCODED_INDICATORS)


def configure_defaults(overrides: dict[str, dict[str, Any]]) -> None:
    """Override default parameters globally for registered indicators.

    This function modifies the default_params class attribute on indicator
    classes, affecting all future instantiations.

    Args:
        overrides: Dictionary mapping indicator names to parameter overrides.
            e.g., {"rsi": {"period": 21}, "macd": {"fast_period": 8}}

    Example:
        >>> from liq.features.indicators import configure_defaults, get_indicator
        >>>
        >>> configure_defaults({"rsi": {"period": 21}})
        >>> RSI = get_indicator("rsi")
        >>> rsi = RSI()  # Now uses period=21 by default
    """
    for indicator_name, params in overrides.items():
        name_lower = indicator_name.lower()
        if name_lower in _HARDCODED_INDICATORS:
            cls = _HARDCODED_INDICATORS[name_lower]
            # Update default_params with new values
            new_defaults = {**cls.default_params, **params}
            cls.default_params = new_defaults


def reset_defaults() -> None:
    """Reset all indicator default parameters to their original values.

    This undoes any changes made by configure_defaults().

    Example:
        >>> from liq.features.indicators import configure_defaults, reset_defaults
        >>>
        >>> configure_defaults({"rsi": {"period": 21}})
        >>> # ... use modified defaults ...
        >>> reset_defaults()  # Restore original period=14
    """
    for name, original_params in _ORIGINAL_DEFAULTS.items():
        if name in _HARDCODED_INDICATORS:
            _HARDCODED_INDICATORS[name].default_params = original_params.copy()
