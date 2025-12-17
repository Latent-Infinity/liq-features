"""Derived OHLC features computation.

This module computes derived fields from standard OHLC price bars,
providing additional features useful for technical analysis and ML models.

Design Principles:
    - SRP: Only computes derived fields, no storage or I/O
    - DRY: Single function for all derived calculations
    - KISS: Pure polars expressions, no complex logic

Example:
    >>> import polars as pl
    >>> from liq.features.derived import compute_derived_fields
    >>>
    >>> df = pl.DataFrame({
    ...     "open": [1.0850, 1.0860],
    ...     "high": [1.0875, 1.0890],
    ...     "low": [1.0825, 1.0850],
    ...     "close": [1.0860, 1.0885],
    ... })
    >>> result = compute_derived_fields(df)
    >>> print(result.columns)
    ['open', 'high', 'low', 'close', 'midrange', 'range', 'true_range', ...]
"""

import polars as pl

# Default Fibonacci-based windows for rolling calculations
# These periods are commonly used in quantitative trading strategies
DEFAULT_FIBONACCI_WINDOWS: list[int] = [55, 210, 340, 890, 3750]


def compute_derived_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Compute derived OHLC fields.

    Adds the following columns to the DataFrame:
        - midrange: (high + low) / 2
        - range: high - low
        - true_range: max(high-low, |high-prev_close|, |low-prev_close|)
        - true_range_midrange: true_range midpoint ((tr + prev_tr) / 2)
        - true_range_hl: high/low of true_range over lookback

    Args:
        df: DataFrame with OHLC columns (open, high, low, close)

    Returns:
        DataFrame with additional derived columns

    Raises:
        ValueError: If required OHLC columns are missing

    Design:
        - SRP: Only computes derived fields
        - KISS: Uses polars expressions for efficiency
        - No side effects, pure transformation
    """
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute basic derived fields
    result = df.with_columns(
        [
            # Midrange: average of high and low
            ((pl.col("high") + pl.col("low")) / 2).alias("midrange"),
            # Range: high minus low
            (pl.col("high") - pl.col("low")).alias("range"),
        ]
    )

    # Previous values for midrange/high/low/close
    result = result.with_columns(
        [
            pl.col("midrange").shift(1).alias("_prev_midrange"),
            pl.col("high").shift(1).alias("_prev_high"),
            pl.col("low").shift(1).alias("_prev_low"),
            pl.col("close").shift(1).alias("_prev_close"),
        ]
    )

    # Classic true range using previous close
    result = result.with_columns(
        [
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("_prev_close")).abs(),
                (pl.col("low") - pl.col("_prev_close")).abs(),
            ).alias("true_range"),
        ]
    )

    # True range midrange: max(range, |mid_t - mid_{t-1}|)
    result = result.with_columns(
        [
            pl.max_horizontal(
                pl.col("range"),
                (pl.col("midrange") - pl.col("_prev_midrange")).abs(),
            ).alias("true_range_midrange"),
            pl.max_horizontal(
                pl.col("range"),
                (pl.col("high") - pl.col("_prev_high")).abs(),
                (pl.col("low") - pl.col("_prev_low")).abs(),
            ).alias("true_range_hl"),
        ]
    )

    return result.drop("_prev_midrange", "_prev_high", "_prev_low", "_prev_close")


def compute_returns(
    df: pl.DataFrame,
    column: str = "close",
    periods: int = 1,
    log_returns: bool = False,
) -> pl.DataFrame:
    """Compute period returns for a price column.

    Args:
        df: DataFrame with price data
        column: Column to compute returns from (default: "close")
        periods: Number of periods for return calculation (default: 1)
        log_returns: If True, compute log returns; otherwise simple returns

    Returns:
        DataFrame with additional return column

    Example:
        >>> df = pl.DataFrame({"close": [100.0, 102.0, 101.0]})
        >>> result = compute_returns(df, periods=1)
        >>> # Adds 'return_1' column with [null, 0.02, -0.0098...]
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    suffix = f"return_{periods}" if periods != 1 else "return"

    if log_returns:
        # Log return: ln(price_t / price_{t-n})
        return_expr = (pl.col(column) / pl.col(column).shift(periods)).log()
        suffix = f"log_{suffix}"
    else:
        # Simple return: (price_t - price_{t-n}) / price_{t-n}
        return_expr = (pl.col(column) - pl.col(column).shift(periods)) / pl.col(
            column
        ).shift(periods)

    return df.with_columns([return_expr.alias(suffix)])


def compute_volatility(
    df: pl.DataFrame,
    column: str = "close",
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Compute rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame with price data
        column: Column to compute volatility from (default: "close")
        window: Rolling window size (default: 20)
        annualize: If True, annualize the volatility (default: True)
        periods_per_year: Periods per year for annualization (default: 252)

    Returns:
        DataFrame with additional volatility column

    Example:
        >>> df = pl.DataFrame({"close": [100.0, 102.0, 101.0, 103.0, 102.5]})
        >>> result = compute_volatility(df, window=3)
        >>> # Adds 'volatility_3' column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # First compute returns if not present
    return_col = "return"
    if return_col not in df.columns:
        df = compute_returns(df, column=column, periods=1)

    # Compute rolling standard deviation
    vol_expr = pl.col(return_col).rolling_std(window_size=window)

    if annualize:
        import math

        vol_expr = vol_expr * math.sqrt(periods_per_year)

    return df.with_columns([vol_expr.alias(f"volatility_{window}")])


def compute_rolling_returns(
    df: pl.DataFrame,
    column: str = "close",
    windows: list[int] | None = None,
    log_returns: bool = True,
    aggregations: list[str] | None = None,
) -> pl.DataFrame:
    """Compute rolling aggregations of returns over multiple windows.

    This function calculates returns and then applies rolling aggregations
    (sum, mean) over configurable window sizes. Useful for capturing
    momentum and trend features at different time scales.

    Args:
        df: DataFrame with price data
        column: Column to compute returns from (default: "close")
        windows: Window sizes for rolling aggregations
                 (default: DEFAULT_FIBONACCI_WINDOWS [55, 210, 340, 890, 3750])
        log_returns: If True, use log returns (default: True)
        aggregations: Aggregation types to compute (default: ["sum", "mean"])

    Returns:
        DataFrame with additional rolling return columns:
        - {prefix}_sum_{window} for rolling sum
        - {prefix}_mean_{window} for rolling mean
        where prefix is "log_return" or "return"

    Raises:
        ValueError: If column not found in DataFrame

    Example:
        >>> df = pl.DataFrame({"close": [100.0, 102.0, 101.0, ...]})
        >>> result = compute_rolling_returns(df, windows=[55, 210])
        >>> # Adds: log_return_sum_55, log_return_mean_55, log_return_sum_210, ...
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if windows is None:
        windows = DEFAULT_FIBONACCI_WINDOWS.copy()

    if aggregations is None:
        aggregations = ["sum", "mean"]

    # Compute returns first
    prefix = "log_return" if log_returns else "return"
    return_col = prefix

    if log_returns:
        return_expr = (pl.col(column) / pl.col(column).shift(1)).log()
    else:
        return_expr = (pl.col(column) - pl.col(column).shift(1)) / pl.col(column).shift(
            1
        )

    result = df.with_columns([return_expr.alias(return_col)])

    # Add rolling aggregations for each window
    for window in windows:
        agg_exprs = []
        if "sum" in aggregations:
            agg_exprs.append(
                pl.col(return_col)
                .rolling_sum(window_size=window)
                .alias(f"{prefix}_sum_{window}")
            )
        if "mean" in aggregations:
            agg_exprs.append(
                pl.col(return_col)
                .rolling_mean(window_size=window)
                .alias(f"{prefix}_mean_{window}")
            )
        if agg_exprs:
            result = result.with_columns(agg_exprs)

    return result


def compute_multi_window_volatility(
    df: pl.DataFrame,
    column: str = "close",
    windows: list[int] | None = None,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Compute rolling volatility over multiple windows.

    Calculates the standard deviation of returns over configurable
    window sizes, with optional annualization. Useful for capturing
    volatility clustering at different time scales.

    Args:
        df: DataFrame with price data
        column: Column to compute volatility from (default: "close")
        windows: Window sizes for volatility calculation
                 (default: DEFAULT_FIBONACCI_WINDOWS [55, 210, 340, 890, 3750])
        annualize: If True, annualize the volatility (default: True)
        periods_per_year: Periods per year for annualization (default: 252)

    Returns:
        DataFrame with additional volatility columns:
        - volatility_{window} for each window size

    Raises:
        ValueError: If column not found in DataFrame

    Example:
        >>> df = pl.DataFrame({"close": [100.0, 102.0, 101.0, ...]})
        >>> result = compute_multi_window_volatility(df, windows=[55, 210])
        >>> # Adds: volatility_55, volatility_210
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if windows is None:
        windows = DEFAULT_FIBONACCI_WINDOWS.copy()

    # Compute returns first
    return_col = "_return"
    return_expr = (pl.col(column) - pl.col(column).shift(1)) / pl.col(column).shift(1)
    result = df.with_columns([return_expr.alias(return_col)])

    # Compute volatility for each window
    import math

    annualization_factor = math.sqrt(periods_per_year) if annualize else 1.0

    vol_exprs = []
    for window in windows:
        vol_expr = (
            pl.col(return_col).rolling_std(window_size=window) * annualization_factor
        )
        vol_exprs.append(vol_expr.alias(f"volatility_{window}"))

    result = result.with_columns(vol_exprs)

    # Drop temporary return column
    return result.drop(return_col)
