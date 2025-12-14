"""Indicator registry for discovering and instantiating indicators.

This module provides a central registry for looking up indicator classes
by name, combining hardcoded indicators with optional dynamic TA-Lib fallback.

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

from typing import Any

from liq.features.indicators.base import BaseIndicator

# Registry of hardcoded indicators
_HARDCODED_INDICATORS: dict[str, type[BaseIndicator]] = {}

# Store original default_params for reset functionality
_ORIGINAL_DEFAULTS: dict[str, dict[str, Any]] = {}


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

    First checks hardcoded indicators, then falls back to dynamic TA-Lib
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

    # Try dynamic TA-Lib fallback
    try:
        from liq.features.indicators.talib import get_dynamic_indicator

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


def list_indicators() -> list[dict[str, Any]]:
    """List all available indicators with metadata.

    Returns:
        List of dicts with indicator info:
        - name: Indicator code name
        - display_name: Human-readable name
        - default_params: Default parameters
        - source: "hardcoded" or "talib"

    Example:
        >>> indicators = list_indicators()
        >>> for ind in indicators[:3]:
        ...     print(f"{ind['name']}: {ind['display_name']}")
    """
    result = []

    # Add hardcoded indicators
    for name, cls in sorted(_HARDCODED_INDICATORS.items()):
        result.append(
            {
                "name": name,
                "display_name": cls.__name__,
                "default_params": cls.default_params,
                "source": "hardcoded",
            }
        )

    # Add dynamic TA-Lib indicators if available
    try:
        from liq.features.indicators.talib import list_dynamic_indicators

        for info in list_dynamic_indicators():
            # Skip if already in hardcoded
            if info["name"] not in _HARDCODED_INDICATORS:
                result.append(
                    {
                        "name": info["name"],
                        "display_name": info["display_name"],
                        "default_params": info["parameters"],
                        "source": "talib",
                    }
                )
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
