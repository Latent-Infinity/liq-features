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
