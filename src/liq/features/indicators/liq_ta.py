"""Dynamic indicator wrapper backed by liq-ta.

This module is the canonical dynamic indicator backend used by ``liq-features``.
It keeps legacy indicator names and parameter aliases for compatibility.

Notes:
- Public indicator names stay in legacy form for compatibility (for example,
  ``willr``, ``stochf``, ``bbands``, ``cdlhammer``).
- Input mapping behavior remains compatible with existing tests/configs.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

try:
    import liq_ta
    import polars as pl

    HAS_LIQ_TA = True
except ImportError:
    HAS_LIQ_TA = False
    liq_ta = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]

from liq.features.indicators.base import BaseIndicator
from liq.features.numpy_utils import to_numpy_float64

# Legacy names mapped to liq-ta function names.
_FUNCTION_ALIASES: dict[str, str] = {
    "williams_r": "willr",
    "true_range": "trange",
    "bollinger": "bbands",
    "stochastic": "stoch",
    "stochastic_fast": "stochf",
    "stochastic_slow": "stochslow",
}

# Additional synthetic legacy indicators computed from liq-ta outputs.
_SYNTHETIC_INDICATORS: set[str] = {"natr", "plus_di", "minus_di"}

_CATEGORY_TO_GROUP: dict[str, str] = {
    "moving_average": "Overlap Studies",
    "momentum": "Momentum Indicators",
    "volatility": "Volatility Indicators",
    "volume": "Volume Indicators",
    "cycle": "Cycle Indicators",
    "price_transform": "Price Transform",
    "statistic": "Statistic Functions",
}

# Legacy parameter aliases -> liq-ta parameter names.
_GENERIC_PARAM_ALIASES: dict[str, str] = {
    "timeperiod": "period",
    "fastperiod": "fast_period",
    "slowperiod": "slow_period",
    "signalperiod": "signal_period",
    "timeperiod1": "period1",
    "timeperiod2": "period2",
    "timeperiod3": "period3",
    "nbdevup": "std_dev",
    "nbdevdn": "std_dev",
    "acceleration": "af_step",
    "maximum": "af_max",
}

# Function-specific legacy aliases.
_FUNCTION_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "stochastic": {
        "fastk_period": "k_period",
        "slowk_period": "k_slowing",
        "slowd_period": "d_period",
    },
    "stochastic_fast": {
        "fastk_period": "k_period",
        "fastd_period": "d_period",
    },
    "stochastic_slow": {
        "fastk_period": "k_period",
        "slowd_period": "d_period",
    },
    "stochrsi": {
        "timeperiod": "rsi_period",
        "fastk_period": "stoch_period",
        "fastd_period": "d_period",
    },
}

_DEFAULTS_BY_PARAM: dict[str, Any] = {
    "period": 14,
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9,
    "period1": 7,
    "period2": 14,
    "period3": 28,
    "k_period": 14,
    "d_period": 3,
    "k_slowing": 3,
    "rsi_period": 14,
    "stoch_period": 14,
    "std_dev": 2.0,
    "af_start": 0.02,
    "af_step": 0.02,
    "af_max": 0.2,
    "fast_limit": 0.5,
    "slow_limit": 0.05,
}

_FUNCTION_DEFAULT_OVERRIDES: dict[str, dict[str, Any]] = {
    "adosc": {"fast_period": 3, "slow_period": 10},
    "stochrsi": {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3},
    "ultosc": {"period1": 7, "period2": 14, "period3": 28},
    "sar": {"af_start": 0.02, "af_step": 0.02, "af_max": 0.2},
}


def _check_liq_ta() -> None:
    """Check if liq-ta backend is available."""
    if not HAS_LIQ_TA:
        raise ImportError(
            "liq-ta is required for dynamic indicators. "
            "Install with: pip install liq-ta"
        )


def _to_float64(arr: NDArray[Any]) -> NDArray[np.floating[Any]]:
    """Convert array to float64."""
    if arr.dtype != np.float64:
        return arr.astype(np.float64, copy=False)
    return arr


_LIQ_TA_INPUT_CACHE: dict[int, dict[str, NDArray[np.floating[Any]]]] = {}
_LIQ_TA_FINALIZERS: dict[int, weakref.finalize] = {}
_LIQ_TA_CACHE_LOCK = threading.Lock()


def _get_column_cache(df: pl.DataFrame) -> dict[str, NDArray[np.floating[Any]]]:
    key = id(df)
    with _LIQ_TA_CACHE_LOCK:
        cached = _LIQ_TA_INPUT_CACHE.get(key)
        if cached is not None:
            return cached
        cache: dict[str, NDArray[np.floating[Any]]] = {}
        _LIQ_TA_INPUT_CACHE[key] = cache
        _LIQ_TA_FINALIZERS[key] = weakref.finalize(df, _purge_liq_ta_cache, key)
        return cache


def _purge_liq_ta_cache(key: int) -> None:
    with _LIQ_TA_CACHE_LOCK:
        _LIQ_TA_INPUT_CACHE.pop(key, None)
        _LIQ_TA_FINALIZERS.pop(key, None)


def map_inputs(
    df: pl.DataFrame,
    input_names: OrderedDict[str, Any],
    price_column: str = "close",
    indicator_name: str = "unknown",
) -> dict[str, NDArray[np.floating[Any]]]:
    """Map DataFrame columns to input arrays using legacy-compatible rules."""
    inputs: dict[str, NDArray[np.floating[Any]]] = {}
    column_cache = _get_column_cache(df)

    def _get_col_array(col_name: str) -> NDArray[np.floating[Any]]:
        cached = column_cache.get(col_name)
        if cached is not None:
            return cached
        arr = to_numpy_float64(df[col_name], allow_copy=False)
        with _LIQ_TA_CACHE_LOCK:
            column_cache.setdefault(col_name, arr)
        return arr

    for input_key, input_spec in input_names.items():
        input_key_lower = input_key.lower()

        if input_key_lower == "prices":
            if isinstance(input_spec, list):
                fallback: NDArray[np.floating[Any]] | None = None
                if price_column not in df.columns:
                    if "close" in df.columns:
                        price_column = "close"
                    elif "prices" in df.columns:
                        price_column = "prices"
                if price_column in df.columns:
                    fallback = _to_float64(_get_col_array(price_column))
                non_ts_columns = [col for col in df.columns if col != "ts"]
                for col in input_spec:
                    col_lower = str(col).lower()
                    allow_volume_proxy = (
                        col_lower == "volume"
                        and fallback is not None
                        and len(non_ts_columns) == 1
                        and non_ts_columns[0] == price_column
                    )
                    if col_lower in df.columns:
                        inputs[col_lower] = _to_float64(_get_col_array(col_lower))
                    elif fallback is not None and col_lower in {
                        "open",
                        "high",
                        "low",
                        "close",
                        "real",
                        "price",
                    } | ({"volume"} if allow_volume_proxy else set()):
                        inputs[col_lower] = fallback
                    else:
                        raise ValueError(
                            f"Indicator {indicator_name} requires '{col_lower}' column, "
                            f"but it's not in the DataFrame. Available: {df.columns}"
                        )
            else:
                for col in ["high", "low", "close"]:
                    if col in df.columns:
                        inputs[col] = _to_float64(_get_col_array(col))

        elif input_key_lower in ("price", "real", "real0"):
            if price_column not in df.columns:
                if "real" in df.columns:
                    price_column = "real"
                elif "value" in df.columns:
                    price_column = "value"
                elif "close" in df.columns:
                    price_column = "close"
                else:
                    non_ts_columns = [col for col in df.columns if col != "ts"]
                    if non_ts_columns:
                        price_column = non_ts_columns[0]
            inputs["close"] = _to_float64(_get_col_array(price_column))

        elif input_key_lower == "real1":
            if "high" in df.columns:
                inputs["high"] = _to_float64(_get_col_array("high"))
            else:
                if price_column not in df.columns:
                    if "real" in df.columns:
                        price_column = "real"
                    elif "value" in df.columns:
                        price_column = "value"
                    elif "close" in df.columns:
                        price_column = "close"
                    else:
                        non_ts_columns = [col for col in df.columns if col != "ts"]
                        if non_ts_columns:
                            price_column = non_ts_columns[0]
                inputs["close"] = _to_float64(_get_col_array(price_column))

        elif input_key_lower in ("price0", "price1"):
            target = "close" if input_key_lower == "price0" else "high"
            if isinstance(input_spec, str) and input_spec.lower() in df.columns:
                col = input_spec.lower()
                inputs[col] = _to_float64(_get_col_array(col))
            elif input_key_lower in df.columns:
                inputs[target] = _to_float64(_get_col_array(input_key_lower))
            elif price_column in df.columns:
                inputs[target] = _to_float64(_get_col_array(price_column))
            else:
                raise ValueError(
                    f"Indicator {indicator_name} requires '{input_spec}' column, "
                    f"but it's not in the DataFrame. Available: {df.columns}"
                )

        elif input_key_lower == "periods":
            continue

        else:
            if input_key_lower in df.columns:
                inputs[input_key_lower] = _to_float64(_get_col_array(input_key_lower))
            else:
                raise ValueError(
                    f"Indicator {indicator_name} requires '{input_key_lower}' column, "
                    f"but it's not in the DataFrame. Available: {df.columns}"
                )

    return inputs


def format_outputs(
    ts_series: pl.Series,
    result_arrays: Any,
    output_names: list[str],
) -> pl.DataFrame:
    """Format output arrays into a Polars DataFrame."""
    if len(output_names) == 1:
        if isinstance(result_arrays, np.ndarray):
            values = result_arrays
        elif isinstance(result_arrays, (tuple, list)):
            values = result_arrays[0]
        else:
            values = result_arrays

        result = pl.DataFrame({"ts": ts_series, "value": values})
        return result.filter(pl.col("value").is_not_nan())

    result_dict: dict[str, Any] = {"ts": ts_series}

    if isinstance(result_arrays, (tuple, list)):
        result_list = list(result_arrays)
    else:
        result_list = [result_arrays]

    n_outputs = min(len(output_names), len(result_list))
    for i in range(n_outputs):
        output_name = output_names[i]
        result_dict[output_name.lower()] = result_list[i]

    result = pl.DataFrame(result_dict)
    for i in range(n_outputs):
        result = result.filter(pl.col(output_names[i].lower()).is_not_nan())
    return result


def _public_name_for_function(function_name: str) -> str:
    if function_name.startswith("cdl_"):
        return "cdl" + function_name[4:].replace("_", "")
    return _FUNCTION_ALIASES.get(function_name, function_name)


def _group_for_indicator(function_name: str, category: str) -> str:
    if function_name.startswith("cdl_"):
        return "Pattern Recognition"
    return _CATEGORY_TO_GROUP.get(category, "Unknown")


def _build_input_names(backend_inputs: list[str]) -> OrderedDict[str, Any]:
    if backend_inputs == ["data"]:
        return OrderedDict({"real": None})

    if backend_inputs == ["data0", "data1"]:
        return OrderedDict({"price0": "close", "price1": "high"})

    inputs_wo_periods = [x for x in backend_inputs if x != "periods"]
    if inputs_wo_periods and all(
        x in {"open", "high", "low", "close", "volume"} for x in inputs_wo_periods
    ):
        out: OrderedDict[str, Any] = OrderedDict({"prices": inputs_wo_periods})
        if "periods" in backend_inputs:
            out["periods"] = None
        return out

    out = OrderedDict()
    for item in backend_inputs:
        if item == "data":
            out["real"] = None
        elif item == "data0":
            out["price0"] = "close"
        elif item == "data1":
            out["price1"] = "high"
        else:
            out[item] = None
    return out


def _public_param_name(function_name: str, backend_param: str) -> str:
    if function_name == "stochastic":
        return {
            "k_period": "fastk_period",
            "k_slowing": "slowk_period",
            "d_period": "slowd_period",
        }.get(backend_param, backend_param)

    if function_name == "stochastic_fast":
        return {
            "k_period": "fastk_period",
            "d_period": "fastd_period",
        }.get(backend_param, backend_param)

    if function_name == "stochastic_slow":
        return {
            "k_period": "fastk_period",
            "d_period": "slowd_period",
        }.get(backend_param, backend_param)

    if function_name == "stochrsi":
        return {
            "rsi_period": "timeperiod",
            "stoch_period": "fastk_period",
            "k_period": "fastd_period",
            "d_period": "slowd_period",
        }.get(backend_param, backend_param)

    if backend_param == "period":
        return "timeperiod"
    if backend_param == "fast_period":
        return "fastperiod"
    if backend_param == "slow_period":
        return "slowperiod"
    if backend_param == "signal_period":
        return "signalperiod"
    if backend_param == "period1":
        return "timeperiod1"
    if backend_param == "period2":
        return "timeperiod2"
    if backend_param == "period3":
        return "timeperiod3"
    if backend_param == "std_dev":
        return "nbdevup"

    return backend_param


def _default_for_param(function_name: str, backend_param: str) -> Any:
    override = _FUNCTION_DEFAULT_OVERRIDES.get(function_name, {})
    if backend_param in override:
        return override[backend_param]
    return _DEFAULTS_BY_PARAM.get(backend_param)


def _public_outputs(function_name: str, backend_outputs: list[str]) -> tuple[list[str], list[int] | None]:
    if function_name == "macd":
        return ["macd", "macdsignal", "macdhist"], None
    if function_name == "stochastic":
        return ["slowk", "slowd"], None
    if function_name == "stochastic_fast":
        return ["fastk", "fastd"], None
    if function_name == "stochastic_slow":
        return ["slowk", "slowd"], None
    if function_name == "bollinger":
        return ["upperband", "middleband", "lowerband"], None
    if function_name == "aroon":
        # liq-ta metadata is [aroon_up, aroon_down], legacy style is [aroondown, aroonup]
        return ["aroondown", "aroonup"], [1, 0]

    if len(backend_outputs) == 1:
        return ["real"], None

    return [o.lower() for o in backend_outputs], None


def _build_registry() -> dict[str, dict[str, Any]]:
    _check_liq_ta()
    assert liq_ta is not None

    registry: dict[str, dict[str, Any]] = {}

    for function_name, meta in liq_ta.INDICATORS.items():
        public_name = _public_name_for_function(function_name)
        backend_params = list(meta.get("params", []))

        parameters: dict[str, Any] = {}
        for p in backend_params:
            parameters[_public_param_name(function_name, p)] = _default_for_param(function_name, p)

        if function_name == "bollinger":
            parameters.setdefault("nbdevdn", parameters.get("nbdevup", 2.0))

        outputs, output_indices = _public_outputs(function_name, list(meta.get("outputs", [])))

        registry[public_name] = {
            "name": public_name.upper(),
            "display_name": meta.get("name", public_name.upper()),
            "group": _group_for_indicator(function_name, meta.get("category", "")),
            "inputs": list(_build_input_names(list(meta.get("inputs", []))).keys()),
            "input_names": _build_input_names(list(meta.get("inputs", []))),
            "parameters": parameters,
            "outputs": outputs,
            "_function_name": function_name,
            "_backend_inputs": list(meta.get("inputs", [])),
            "_backend_params": backend_params,
            "_output_indices": output_indices,
            "_mode": "direct",
        }

    # Synthetic indicators for compatibility.
    registry["natr"] = {
        "name": "NATR",
        "display_name": "Normalized Average True Range",
        "group": "Volatility Indicators",
        "inputs": ["prices"],
        "input_names": OrderedDict({"prices": ["high", "low", "close"]}),
        "parameters": {"timeperiod": 14},
        "outputs": ["real"],
        "_function_name": "atr",
        "_backend_inputs": ["high", "low", "close"],
        "_backend_params": ["period"],
        "_output_indices": None,
        "_mode": "natr",
    }

    for name, idx in (("plus_di", 1), ("minus_di", 2)):
        registry[name] = {
            "name": name.upper(),
            "display_name": name.replace("_", " ").title(),
            "group": "Momentum Indicators",
            "inputs": ["prices"],
            "input_names": OrderedDict({"prices": ["high", "low", "close"]}),
            "parameters": {"timeperiod": 14},
            "outputs": ["real"],
            "_function_name": "adx",
            "_backend_inputs": ["high", "low", "close"],
            "_backend_params": ["period"],
            "_output_indices": [idx],
            "_mode": "adx_component",
        }

    # ADX compatibility: ADX is single-output in this surface.
    if "adx" in registry:
        registry["adx"]["outputs"] = ["real"]
        registry["adx"]["_output_indices"] = [0]

    return registry


_REGISTRY_CACHE: dict[str, dict[str, Any]] | None = None


def _registry() -> dict[str, dict[str, Any]]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


def get_available_indicators() -> list[str]:
    """Get list of all available dynamic indicator names (lowercase)."""
    _check_liq_ta()
    return sorted(_registry().keys())


def get_indicator_metadata(name: str) -> dict[str, Any]:
    """Get canonical metadata for a dynamic indicator."""
    _check_liq_ta()
    key = name.lower()

    reg = _registry()
    if key not in reg:
        available = ", ".join(get_available_indicators()[:10])
        raise ValueError(
            f"Unknown liq-ta indicator: {name}. "
            f"Available indicators include: {available}..."
        )

    info = reg[key]
    return {
        "name": info["name"],
        "display_name": info["display_name"],
        "group": info["group"],
        "inputs": info["inputs"],
        "input_names": dict(info.get("input_names", {str(name): [str(name)]})),
        "parameters": info["parameters"],
        "outputs": info["outputs"],
    }


def _backend_kwargs(entry: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    backend_params = set(entry["_backend_params"])
    function_name = entry["_function_name"]

    aliases = dict(_GENERIC_PARAM_ALIASES)
    aliases.update(_FUNCTION_PARAM_ALIASES.get(function_name, {}))

    canonicalized: dict[str, tuple[Any, bool]] = {}
    for key, value in params.items():
        if key.startswith("_"):
            continue
        mapped = aliases.get(key, key)
        if mapped not in backend_params:
            continue

        # Canonical keys should win over legacy aliases when both are present.
        is_canonical = key == mapped
        current = canonicalized.get(mapped)
        if current is None or is_canonical:
            canonicalized[mapped] = (value, is_canonical)

    kwargs: dict[str, Any] = {}
    for key, (value, _) in canonicalized.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                key.endswith("_period")
                or key in {"timeperiod", "timeperiod1", "timeperiod2", "timeperiod3"}
                or key == "displacement"
                or key in {"maxima", "lookback", "period"}
            ):
                kwargs[key] = int(value)
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value

    # Guard common cross-parameter invariants to avoid hard runtime failures in
    # callers that pass invalid combinations.
    fast_period = kwargs.get("fast_period")
    slow_period = kwargs.get("slow_period")
    if isinstance(fast_period, (int, float)) and isinstance(slow_period, (int, float)):
        fast_int = int(fast_period)
        slow_int = int(slow_period)
        if fast_int >= slow_int and slow_int > 0:
            if slow_int == 1:
                slow_int = 2
            else:
                fast_int = max(1, slow_int - 1)
            kwargs["fast_period"] = fast_int
            kwargs["slow_period"] = slow_int

    min_period = kwargs.get("min_period")
    max_period = kwargs.get("max_period")
    if isinstance(min_period, (int, float)) and isinstance(max_period, (int, float)):
        min_int = max(1, int(min_period))
        max_int = max(1, int(max_period))
        if max_int < min_int:
            min_int, max_int = max_int, min_int
        kwargs["min_period"] = min_int
        kwargs["max_period"] = max_int

    # Bollinger compatibility: legacy nbdevup/nbdevdn -> single std_dev.
    if "std_dev" in backend_params and "std_dev" not in kwargs:
        up = params.get("nbdevup")
        dn = params.get("nbdevdn")
        if up is not None:
            kwargs["std_dev"] = up
        elif dn is not None:
            kwargs["std_dev"] = dn

    # Remove None values so backend defaults apply.
    return {k: v for k, v in kwargs.items() if v is not None}


def _backend_args(
    entry: dict[str, Any],
    inputs: dict[str, NDArray[np.floating[Any]]],
    default_period: int | None = None,
) -> list[NDArray[np.floating[Any]]]:
    def _first_existing(*keys: str) -> NDArray[np.floating[Any]]:
        for key in keys:
            if key in inputs:
                return inputs[key]
        raise ValueError(f"Missing required input among: {', '.join(keys)}")

    args: list[NDArray[np.floating[Any]]] = []

    for name in entry["_backend_inputs"]:
        key = name.lower()
        if key == "data":
            args.append(_first_existing("close", "data"))
        elif key == "data0":
            args.append(_first_existing("close", "data0", "high"))
        elif key == "data1":
            args.append(_first_existing("high", "data1", "close"))
        elif key == "periods":
            if default_period is None:
                default_period = 14
            if "close" in inputs:
                size = len(inputs["close"])
            else:
                first_key = next(iter(inputs))
                size = len(inputs[first_key])
            args.append(np.full(size, float(default_period), dtype=np.float64))
        else:
            if key not in inputs:
                raise ValueError(f"Missing required input '{key}'")
            args.append(inputs[key])

    return args


def _compute_dynamic(
    entry: dict[str, Any],
    params: dict[str, Any],
    df: pl.DataFrame,
) -> pl.DataFrame:
    assert liq_ta is not None

    price_col = params.get("_price_column", "close")
    if price_col not in df.columns:
        price_col = "close"

    inputs = map_inputs(df, entry["input_names"], price_column=price_col, indicator_name=entry["name"])
    kwargs = _backend_kwargs(entry, params)
    default_period = kwargs.get("period")
    args = _backend_args(entry, inputs, default_period=default_period)

    backend_fn = getattr(liq_ta, entry["_function_name"])
    result_arrays = backend_fn(*args, **kwargs)

    if df.height == 0:
        raise ValueError("Input DataFrame must not be empty")

    target_length = len(df["ts"]) if "ts" in df.columns else len(next(iter(inputs.values())))

    def _align_output(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if len(array) == target_length:
            return array
        if len(array) < target_length:
            pad = np.full(target_length - len(array), np.nan, dtype=np.float64)
            return np.concatenate([pad, array])
        return array[-target_length:]

    if isinstance(result_arrays, (tuple, list)):
        result_arrays = [_align_output(values) for values in result_arrays]
    else:
        result_arrays = _align_output(result_arrays)

    if entry["_mode"] == "natr":
        if isinstance(result_arrays, list):
            atr_values = result_arrays[0]
        else:
            atr_values = result_arrays
        close_arr = inputs["close"]
        with np.errstate(divide="ignore", invalid="ignore"):
            close_values = np.asarray(close_arr, dtype=np.float64)
            result_arrays = (atr_values / close_values) * 100.0
            result_arrays = np.asarray(result_arrays)
            # Treat near-zero close values as invalid to avoid unstable spikes.
            result_arrays[np.abs(close_values) <= np.finfo(np.float64).eps] = np.nan
            result_arrays[~np.isfinite(result_arrays)] = np.nan

    output_indices = entry.get("_output_indices")
    if output_indices is not None:
        result_list = list(result_arrays) if isinstance(result_arrays, (tuple, list)) else [result_arrays]
        selected = [result_list[i] for i in output_indices if i < len(result_list)]
        if not selected:
            raise ValueError(f"No outputs selected for indicator {entry['name']}")
        result_arrays = selected if len(selected) > 1 else selected[0]

    return format_outputs(df["ts"], result_arrays, entry["outputs"])


def _create_dynamic_indicator_class(name: str) -> type[BaseIndicator]:
    """Create a dynamic indicator class."""
    _check_liq_ta()

    key = name.lower()
    entry = _registry().get(key)
    if entry is None:
        raise ValueError(f"Unknown liq-ta indicator: {name}")

    defaults = dict(entry["parameters"])

    class DynamicIndicator(BaseIndicator):
        name = key
        default_params = defaults
        _indicator_info = entry

        def _compute(self, df: pl.DataFrame) -> pl.DataFrame:
            return _compute_dynamic(self._indicator_info, self.params, df)

    class_name = entry["name"].replace("_", "")
    DynamicIndicator.__name__ = class_name
    DynamicIndicator.__qualname__ = class_name
    DynamicIndicator.__doc__ = (
        f"{entry['display_name']} ({entry['name']}).\n\n"
        f"Group: {entry['group']}\n"
        f"Inputs: {', '.join(entry['inputs'])}\n"
        f"Parameters: {', '.join(f'{k}={v}' for k, v in defaults.items())}\n"
        f"Outputs: {', '.join(entry['outputs'])}\n"
    )

    return DynamicIndicator


_indicator_class_cache: dict[str, type[BaseIndicator]] = {}


def get_dynamic_indicator(name: str) -> type[BaseIndicator]:
    """Get or create a dynamic indicator class."""
    key = name.lower()
    if key in _indicator_class_cache:
        return _indicator_class_cache[key]

    indicator_class = _create_dynamic_indicator_class(key)
    _indicator_class_cache[key] = indicator_class
    return indicator_class


def clear_indicator_cache() -> None:
    """Clear dynamic class cache."""
    _indicator_class_cache.clear()


def list_dynamic_indicators() -> list[dict[str, Any]]:
    """List available dynamic indicators with metadata."""
    _check_liq_ta()
    result = []

    for name in get_available_indicators():
        try:
            info = get_indicator_metadata(name)
            result.append(
                {
                    "name": name,
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
