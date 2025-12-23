"""Default parameter grids for indicator sweeps.

Provides sensible default parameter ranges for common indicators
used in feature selection and hyperparameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

# Fibonacci sequence for period parameters - natural fit for technical analysis
# Extended sequence: 2 through ~3750 for comprehensive parameter sweeps
FIBONACCI_PERIODS_EXTENDED: list[int] = [
    2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181
]

# Standard Fibonacci periods (up to 233) - for most experiments
FIBONACCI_PERIODS: list[int] = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
FIBONACCI_PERIODS_SHORT: list[int] = [2, 3, 5, 8, 13, 21, 34]  # For fast periods
FIBONACCI_PERIODS_MEDIUM: list[int] = [8, 13, 21, 34, 55, 89]  # For slow periods
FIBONACCI_PERIODS_LONG: list[int] = [89, 144, 233, 377, 610, 987]  # For slow periods (extended)


def get_fibonacci_periods_for_bars(
    n_bars: int,
    *,
    min_samples_ratio: float = 20.0,
    max_period: int | None = None,
) -> list[int]:
    """Get appropriate Fibonacci periods based on available data.

    The maximum usable period is n_bars / min_samples_ratio to ensure
    enough samples for meaningful statistics.

    Args:
        n_bars: Number of bars in the dataset.
        min_samples_ratio: Minimum ratio of bars to period (default: 20).
            With 1000 bars and ratio 20, max period is 50.
        max_period: Optional explicit maximum period.

    Returns:
        List of Fibonacci periods that fit within the data constraints.

    Example:
        >>> get_fibonacci_periods_for_bars(52000)  # ~6 years of 1h data
        [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
        >>> get_fibonacci_periods_for_bars(2000)  # ~3 months of 1d data
        [2, 3, 5, 8, 13, 21, 34, 55, 89]
    """
    computed_max = int(n_bars / min_samples_ratio)
    if max_period is not None:
        computed_max = min(computed_max, max_period)

    return [p for p in FIBONACCI_PERIODS_EXTENDED if p <= computed_max]


# Estimated bar counts for different timeframes (BTC 2020-2025)
TIMEFRAME_BAR_COUNTS: dict[str, int] = {
    "1m": 3_123_000,   # ~3.1M bars
    "5m": 624_600,     # ~624k bars
    "15m": 208_200,    # ~208k bars
    "30m": 104_100,    # ~104k bars
    "1h": 52_050,      # ~52k bars
    "4h": 13_010,      # ~13k bars
    "1d": 2_170,       # ~2.2k bars
}


def get_periods_for_timeframe(timeframe: str) -> list[int]:
    """Get appropriate Fibonacci periods for a given timeframe.

    Uses estimated bar counts to determine maximum useful period.

    Args:
        timeframe: Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d).

    Returns:
        List of appropriate Fibonacci periods.

    Example:
        >>> get_periods_for_timeframe("1h")
        [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
        >>> get_periods_for_timeframe("1d")
        [2, 3, 5, 8, 13, 21, 34, 55, 89]
    """
    n_bars = TIMEFRAME_BAR_COUNTS.get(timeframe, 10000)
    return get_fibonacci_periods_for_bars(n_bars)

# Default parameter grids for common indicators (using Fibonacci periods)
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    # Momentum indicators
    "rsi": {"period": [2, 3, 5, 8, 13, 21, 34, 55]},
    "roc": {"period": [2, 3, 5, 8, 13, 21]},
    "stochastic": {
        "k_period": [5, 8, 13, 21],
        "d_period": [3, 5],
        "smooth_k": [3],
    },
    "stochastic_midrange": {
        "k_period": [5, 8, 13, 21],
        "d_period": [3, 5],
        "smooth_k": [3],
    },
    # Trend indicators
    "ema": {"period": [5, 8, 13, 21, 34, 55, 89, 144, 233]},
    "sma": {"period": [5, 8, 13, 21, 34, 55, 89, 144, 233]},
    "wma": {"period": [5, 8, 13, 21, 34, 55]},
    "bbands": {
        "period": [8, 13, 21, 34, 55],
        "std_dev": [1.5, 2.0, 2.5],
    },
    "atr": {"period": [5, 8, 13, 21, 34]},
    "atr_midrange": {"period": [5, 8, 13, 21, 34]},
    "adx": {"period": [5, 8, 13, 21, 34]},
    "adx_midrange": {"period": [5, 8, 13, 21, 34]},
    # MACD variations (Fibonacci-based fast/slow)
    "macd": {
        "fast_period": [5, 8, 13],
        "slow_period": [13, 21, 34],
        "signal_period": [5, 8, 13],
    },
    # TA-Lib indicators (common ones)
    "cci": {"timeperiod": [8, 13, 21, 34, 55]},
    "willr": {"timeperiod": [5, 8, 13, 21, 34]},
    "mfi": {"timeperiod": [5, 8, 13, 21, 34]},
    "obv": {},  # No parameters
    "ad": {},  # No parameters
    "adosc": {"fastperiod": [3, 5, 8], "slowperiod": [8, 13, 21]},
    "aroon": {"timeperiod": [8, 13, 21, 34]},
    "bop": {},  # No parameters
    "dx": {"timeperiod": [5, 8, 13, 21, 34]},
    "minus_di": {"timeperiod": [5, 8, 13, 21, 34]},
    "plus_di": {"timeperiod": [5, 8, 13, 21, 34]},
    "mom": {"timeperiod": [3, 5, 8, 13, 21]},
    "ppo": {"fastperiod": [5, 8, 13], "slowperiod": [13, 21, 34]},
    "trix": {"timeperiod": [8, 13, 21, 34]},
    "ultosc": {
        "timeperiod1": [5, 8],
        "timeperiod2": [13, 21],
        "timeperiod3": [21, 34],
    },
    "natr": {"timeperiod": [5, 8, 13, 21, 34]},
    "trange": {},  # No parameters
}


@dataclass(frozen=True)
class IndicatorSpec:
    """Specification for an indicator with parameters.

    Attributes:
        name: Indicator name (lowercase).
        params: Parameter dictionary.
    """

    name: str
    params: dict[str, Any]

    @property
    def key(self) -> str:
        """Generate unique key for this indicator specification."""
        if not self.params:
            return self.name
        param_str = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return f"{self.name}_{param_str}"

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.params.items()))))


def enumerate_with_params(
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
    *,
    indicators: list[str] | None = None,
    include_defaults: bool = True,
) -> list[IndicatorSpec]:
    """Enumerate all indicator + parameter combinations.

    Generates all combinations from parameter grids for use in
    feature selection experiments.

    Args:
        param_grids: Custom parameter grids. If None, uses DEFAULT_PARAM_GRIDS.
        indicators: List of indicator names to include. If None, uses all
            indicators in param_grids.
        include_defaults: If True, also include default parameters for each
            indicator (even if not in the sweep grid).

    Returns:
        List of IndicatorSpec with all combinations.

    Example:
        >>> specs = enumerate_with_params()
        >>> len(specs)
        150  # ~150 indicator/param combinations
        >>>
        >>> # Custom grids
        >>> specs = enumerate_with_params({"rsi": {"period": [5, 10, 20]}})
        >>> [s.key for s in specs]
        ['rsi_period5', 'rsi_period10', 'rsi_period20']
    """
    from liq.features.indicators.registry import get_indicator, list_indicators

    grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS.copy()

    # Filter to requested indicators
    if indicators is not None:
        grids = {k: v for k, v in grids.items() if k in indicators}

    result: list[IndicatorSpec] = []
    seen: set[tuple[str, tuple]] = set()

    # Get list of available indicators
    available = {info["name"]: info for info in list_indicators()}

    for name, grid in grids.items():
        if name not in available:
            continue

        if not grid:
            # No parameters to sweep
            spec = IndicatorSpec(name=name, params={})
            key = (name, ())
            if key not in seen:
                result.append(spec)
                seen.add(key)
            continue

        # Generate all parameter combinations
        param_names = list(grid.keys())
        param_values = [grid[p] for p in param_names]

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            key = (name, tuple(sorted(params.items())))

            if key not in seen:
                result.append(IndicatorSpec(name=name, params=params))
                seen.add(key)

        # Include default parameters if requested
        if include_defaults:
            try:
                cls = get_indicator(name)
                default_params = cls.default_params
                key = (name, tuple(sorted(default_params.items())))

                if key not in seen:
                    result.append(IndicatorSpec(name=name, params=default_params))
                    seen.add(key)
            except ValueError:
                pass

    return result


def get_param_grid(indicator_name: str) -> dict[str, list[Any]]:
    """Get parameter grid for a specific indicator.

    Args:
        indicator_name: Indicator name (case-insensitive).

    Returns:
        Parameter grid dictionary, or empty dict if no grid defined.

    Example:
        >>> grid = get_param_grid("rsi")
        >>> print(grid)
        {'period': [2, 3, 5, 8, 13, 21, 34, 55]}
    """
    return DEFAULT_PARAM_GRIDS.get(indicator_name.lower(), {})


# Categories to exclude from full sweep by default
# Pattern Recognition produces discrete outputs (100, -100, 0) which may not work well with MI
# Math Transform (ln, log10, sqrt, sin, cos, etc.) are pure price transforms with no predictive value
EXCLUDED_CATEGORIES: set[str] = {"Pattern Recognition", "Math Transform"}

# Categories that can be optionally included
OPTIONAL_CATEGORIES: set[str] = {"Pattern Recognition"}

# Indicators that generate too many invalid parameter combinations
# These have constraints like fast < slow that cause most combinations to fail
PROBLEMATIC_INDICATORS: set[str] = {
    "macdext",  # 38k+ variations, most fail due to fast >= slow
    "mavp",     # Requires periods array input which we don't support
}

# Default period variations for auto-generated grids (Fibonacci-based)
PERIOD_VARIATIONS: list[int] = [5, 8, 13, 21, 34, 55]
FAST_PERIOD_VARIATIONS: list[int] = [3, 5, 8, 13]
SLOW_PERIOD_VARIATIONS: list[int] = [13, 21, 34, 55]

# Extended variations for comprehensive sweeps
PERIOD_VARIATIONS_EXTENDED: list[int] = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
FAST_PERIOD_VARIATIONS_EXTENDED: list[int] = [2, 3, 5, 8, 13, 21, 34, 55, 89]
SLOW_PERIOD_VARIATIONS_EXTENDED: list[int] = [13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Coarse variations for initial indicator ranking
# Use these to quickly identify promising indicator families before detailed sweep
# Based on empirical analysis: 5 catches top performers (minindex), 8 catches natr,
# 21 is middle ground, 55 is better than 89 for MAs/bbands
PERIOD_VARIATIONS_COARSE: list[int] = [5, 8, 21, 55]
FAST_PERIOD_VARIATIONS_COARSE: list[int] = [5, 8]
SLOW_PERIOD_VARIATIONS_COARSE: list[int] = [21, 55]


def auto_generate_param_grid(
    indicator_info: dict[str, Any],
    *,
    period_variations: list[int] | None = None,
    use_extended: bool = False,
    use_coarse: bool = False,
    max_period: int | None = None,
) -> dict[str, list[Any]]:
    """Auto-generate parameter grid based on TA-Lib indicator metadata.

    Creates sensible parameter variations based on default values
    and parameter names.

    Args:
        indicator_info: Indicator metadata from get_indicator_info().
        period_variations: Custom period variations. Uses default if None.
        use_extended: If True, use extended Fibonacci sequences (up to 2584).
        use_coarse: If True, use only 4 periods (5, 8, 21, 55) for fast initial ranking.
        max_period: Maximum period to include (filters all period parameters).

    Returns:
        Parameter grid dictionary.

    Example:
        >>> from liq.features.indicators.talib import get_indicator_info
        >>> info = get_indicator_info("RSI")
        >>> grid = auto_generate_param_grid(info)
        >>> print(grid)
        {'timeperiod': [5, 8, 13, 21, 34, 55]}
        >>> grid = auto_generate_param_grid(info, use_coarse=True)
        >>> print(grid)
        {'timeperiod': [5, 8, 21, 55]}
    """
    if period_variations:
        periods = period_variations
        fast_periods = [p for p in period_variations if p <= 55]
        slow_periods = [p for p in period_variations if p >= 13]
    elif use_coarse:
        periods = PERIOD_VARIATIONS_COARSE
        fast_periods = FAST_PERIOD_VARIATIONS_COARSE
        slow_periods = SLOW_PERIOD_VARIATIONS_COARSE
    elif use_extended:
        periods = PERIOD_VARIATIONS_EXTENDED
        fast_periods = FAST_PERIOD_VARIATIONS_EXTENDED
        slow_periods = SLOW_PERIOD_VARIATIONS_EXTENDED
    else:
        periods = PERIOD_VARIATIONS
        fast_periods = FAST_PERIOD_VARIATIONS
        slow_periods = SLOW_PERIOD_VARIATIONS

    # Apply max_period filter
    if max_period is not None:
        periods = [p for p in periods if p <= max_period]
        fast_periods = [p for p in fast_periods if p <= max_period]
        slow_periods = [p for p in slow_periods if p <= max_period]

    params = indicator_info.get("parameters", {})

    if not params:
        return {}

    grid: dict[str, list[Any]] = {}

    for param_name, default_value in params.items():
        param_lower = param_name.lower()

        # Period-like parameters - use Fibonacci-based sequences
        if "period" in param_lower:
            if "fast" in param_lower:
                grid[param_name] = fast_periods
            elif "slow" in param_lower:
                grid[param_name] = slow_periods
            else:
                # Use full Fibonacci periods
                grid[param_name] = periods

        # Acceleration factor (for SAR)
        elif "acceleration" in param_lower:
            if default_value:
                grid[param_name] = [default_value * 0.5, default_value, default_value * 2]
            else:
                grid[param_name] = [0.01, 0.02, 0.04]

        # Maximum values
        elif "maximum" in param_lower:
            if default_value:
                grid[param_name] = [default_value * 0.5, default_value, default_value * 1.5]
            else:
                grid[param_name] = [0.1, 0.2, 0.4]

        # Standard deviation / multipliers
        elif "nbdev" in param_lower or "deviation" in param_lower:
            grid[param_name] = [1.5, 2.0, 2.5, 3.0]

        # MA types - check BEFORE signal to handle "signalmatype" correctly
        elif "matype" in param_lower:
            grid[param_name] = [0, 1, 2]  # SMA, EMA, WMA

        # Signal periods (Fibonacci-based) - only for signalperiod, not signalmatype
        elif "signal" in param_lower and "period" in param_lower:
            grid[param_name] = [3, 5, 8, 13]

        # Default: use single default value
        else:
            if default_value is not None:
                grid[param_name] = [default_value]

    return grid


def generate_talib_param_grids(
    *,
    exclude_categories: set[str] | None = None,
    exclude_indicators: set[str] | None = None,
    include_only: list[str] | None = None,
    include_midrange: bool = False,
    include_patterns: bool = False,
    use_extended: bool = False,
    use_coarse: bool = False,
    max_period: int | None = None,
    timeframe: str | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """Generate parameter grids for all TA-Lib indicators.

    Auto-generates sensible parameter variations for each indicator
    based on their default parameters and parameter names.

    Args:
        exclude_categories: Categories to exclude (default: Pattern Recognition).
        include_only: If provided, only include these indicator names.
        include_midrange: If True, also creates variants using midrange price
            for single-price indicators.
        include_patterns: If True, include Pattern Recognition (candlestick) indicators.
            These produce discrete outputs (100, -100, 0) which may not work well
            with Mutual Information but can be useful for other analyses.
        use_extended: If True, use extended Fibonacci periods (up to 2584).
        use_coarse: If True, use only 4 periods (5, 8, 21, 55) for fast initial ranking.
            Reduces variations by ~80% for quick indicator family discovery.
        max_period: Maximum period to include in grids.
        timeframe: If provided, auto-calculate max_period based on typical bar counts.
            Overrides max_period if both are provided.

    Returns:
        Dictionary mapping indicator names to parameter grids.

    Example:
        >>> grids = generate_talib_param_grids()
        >>> len(grids)
        97  # ~97 non-pattern indicators

        >>> grids = generate_talib_param_grids(use_coarse=True)
        >>> # ~240 variations (80 indicators × 3 periods)

        >>> # Extended sweep for 1h data
        >>> grids = generate_talib_param_grids(use_extended=True, timeframe="1h")
    """
    try:
        from liq.features.indicators.talib import list_dynamic_indicators
    except ImportError:
        return {}

    # Auto-calculate max_period from timeframe if provided
    if timeframe is not None:
        n_bars = TIMEFRAME_BAR_COUNTS.get(timeframe, 10000)
        max_period = int(n_bars / 20)  # Ensure at least 20 samples per period

    # Determine which categories to exclude
    if exclude_categories is not None:
        excluded = exclude_categories
    elif include_patterns:
        excluded = set()  # Include everything
    else:
        excluded = EXCLUDED_CATEGORIES

    indicators = list_dynamic_indicators()
    grids: dict[str, dict[str, list[Any]]] = {}

    # Track which indicators use single price input
    single_price_indicators: set[str] = set()

    # Determine which indicators to exclude
    excluded_inds = exclude_indicators if exclude_indicators is not None else PROBLEMATIC_INDICATORS

    for ind in indicators:
        name = ind["name"]
        group = ind.get("group", "")
        inputs = ind.get("inputs", [])

        # Skip excluded categories
        if group in excluded:
            continue

        # Skip problematic indicators
        if name in excluded_inds:
            continue

        # Filter to include_only if specified
        if include_only and name not in include_only:
            continue

        # Use existing grid if defined, otherwise auto-generate
        if name in DEFAULT_PARAM_GRIDS and not use_coarse:
            # Apply max_period filter to DEFAULT_PARAM_GRIDS too
            grid = DEFAULT_PARAM_GRIDS[name].copy()
            if max_period is not None:
                for k, v in grid.items():
                    if "period" in k.lower():
                        grid[k] = [p for p in v if p <= max_period]
            grids[name] = grid
        else:
            grids[name] = auto_generate_param_grid(
                ind,
                use_extended=use_extended,
                use_coarse=use_coarse,
                max_period=max_period,
            )

        # Check if this indicator uses single price input
        inputs_lower = [i.lower() for i in inputs]
        if "price" in inputs_lower or "real" in inputs_lower:
            single_price_indicators.add(name)

    # Add midrange variants for single-price indicators
    if include_midrange:
        for name in single_price_indicators:
            if name in grids:
                midrange_name = f"{name}_midrange"
                midrange_grid = grids[name].copy()
                midrange_grid["_price_column"] = ["midrange"]
                grids[midrange_name] = midrange_grid

    return grids


def get_single_price_indicators() -> list[str]:
    """Get list of TA-Lib indicators that use single price input.

    These indicators can be computed on different price columns
    (close, midrange, typical price, etc.).

    Returns:
        List of indicator names.
    """
    try:
        from liq.features.indicators.talib import list_dynamic_indicators
    except ImportError:
        return []

    indicators = list_dynamic_indicators()
    result = []

    for ind in indicators:
        inputs = ind.get("inputs", [])
        inputs_lower = [i.lower() for i in inputs]
        if "price" in inputs_lower or "real" in inputs_lower:
            result.append(ind["name"])

    return result


def count_combinations(
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
) -> int:
    """Count total number of indicator/parameter combinations.

    Args:
        param_grids: Parameter grids to count. Uses DEFAULT_PARAM_GRIDS if None.

    Returns:
        Total number of combinations.
    """
    grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS

    total = 0
    for grid in grids.values():
        if not grid:
            total += 1
        else:
            count = 1
            for values in grid.values():
                count *= len(values)
            total += count

    return total
