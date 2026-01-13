"""Tests for indicator parameter grid enumeration."""

import pytest

from liq.features.indicators import (
    DEFAULT_PARAM_GRIDS,
    IndicatorSpec,
    count_combinations,
    enumerate_with_params,
    get_param_grid,
)
from liq.features.indicators.param_grids import (
    FAST_PERIOD_VARIATIONS_COARSE,
    PERIOD_VARIATIONS_COARSE,
    SLOW_PERIOD_VARIATIONS_COARSE,
    generate_fibonacci_period_pairs,
)


class TestIndicatorSpec:
    """Tests for IndicatorSpec dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic IndicatorSpec creation."""
        spec = IndicatorSpec(name="rsi", params={"period": 14})
        assert spec.name == "rsi"
        assert spec.params == {"period": 14}

    def test_key_generation(self) -> None:
        """Test unique key generation."""
        spec = IndicatorSpec(name="rsi", params={"period": 14})
        assert spec.key == "rsi_period14"

    def test_key_empty_params(self) -> None:
        """Test key with no parameters."""
        spec = IndicatorSpec(name="obv", params={})
        assert spec.key == "obv"

    def test_key_multiple_params(self) -> None:
        """Test key with multiple parameters."""
        spec = IndicatorSpec(name="macd", params={"fast_period": 12, "slow_period": 26})
        # Should be sorted alphabetically
        assert "fast_period12" in spec.key
        assert "slow_period26" in spec.key

    def test_hashable(self) -> None:
        """Test IndicatorSpec is hashable for use in sets."""
        spec1 = IndicatorSpec(name="rsi", params={"period": 14})
        spec2 = IndicatorSpec(name="rsi", params={"period": 14})
        spec3 = IndicatorSpec(name="rsi", params={"period": 21})

        # Equal specs should hash the same
        assert hash(spec1) == hash(spec2)

        # Can be used in sets
        specs = {spec1, spec2, spec3}
        assert len(specs) == 2


class TestEnumerateWithParams:
    """Tests for enumerate_with_params function."""

    def test_basic_enumeration(self) -> None:
        """Test basic enumeration with custom grid."""
        specs = enumerate_with_params(
            {"rsi": {"period": [5, 10, 20]}},
            include_defaults=False,
        )

        assert len(specs) == 3
        periods = [s.params["period"] for s in specs]
        assert set(periods) == {5, 10, 20}

    def test_multiple_params(self) -> None:
        """Test enumeration with multiple parameters."""
        specs = enumerate_with_params(
            {"bbands": {"period": [10, 20], "std_dev": [1.5, 2.0]}},
            include_defaults=False,
        )

        # 2 * 2 = 4 combinations
        assert len(specs) == 4

    def test_filter_by_indicators(self) -> None:
        """Test filtering to specific indicators."""
        specs = enumerate_with_params(
            DEFAULT_PARAM_GRIDS,
            indicators=["rsi"],
            include_defaults=False,
        )

        # All should be RSI
        for spec in specs:
            assert spec.name == "rsi"

    def test_no_duplicates(self) -> None:
        """Test no duplicate specs are generated."""
        specs = enumerate_with_params(
            {"rsi": {"period": [14, 14, 14]}},  # Duplicates in grid
            include_defaults=False,
        )

        assert len(specs) == 1

    def test_empty_params_indicator(self) -> None:
        """Test indicator with no parameters."""
        specs = enumerate_with_params(
            {"obv": {}},
            include_defaults=False,
        )

        assert len(specs) == 1
        assert specs[0].params == {}

    def test_include_defaults(self) -> None:
        """Test include_defaults adds default parameters."""
        specs = enumerate_with_params(
            {"rsi": {"period": [5, 10]}},
            include_defaults=True,
        )

        # Should have 5, 10, and the default (14)
        periods = [s.params.get("period") for s in specs]
        assert 14 in periods  # Default RSI period


class TestGetParamGrid:
    """Tests for get_param_grid function."""

    def test_existing_indicator(self) -> None:
        """Test getting grid for known indicator."""
        grid = get_param_grid("rsi")
        assert "period" in grid
        assert isinstance(grid["period"], list)

    def test_unknown_indicator(self) -> None:
        """Test getting grid for unknown indicator."""
        grid = get_param_grid("unknown_indicator")
        assert grid == {}

    def test_case_insensitive(self) -> None:
        """Test case-insensitive lookup."""
        grid_lower = get_param_grid("rsi")
        grid_upper = get_param_grid("RSI")
        assert grid_lower == grid_upper


class TestCountCombinations:
    """Tests for count_combinations function."""

    def test_basic_count(self) -> None:
        """Test counting combinations."""
        count = count_combinations({"rsi": {"period": [5, 10, 20]}})
        assert count == 3

    def test_multiple_params(self) -> None:
        """Test counting with multiple parameters."""
        count = count_combinations({
            "bbands": {"period": [10, 20], "std_dev": [1.5, 2.0]},
        })
        assert count == 4  # 2 * 2

    def test_empty_params(self) -> None:
        """Test counting indicator with no params."""
        count = count_combinations({"obv": {}})
        assert count == 1

    def test_default_grids(self) -> None:
        """Test counting with default grids."""
        count = count_combinations()
        assert count > 50  # Should have many combinations


class TestDefaultParamGrids:
    """Tests for DEFAULT_PARAM_GRIDS constant."""

    def test_contains_common_indicators(self) -> None:
        """Test common indicators are included."""
        assert "rsi" in DEFAULT_PARAM_GRIDS
        assert "macd" in DEFAULT_PARAM_GRIDS
        assert "ema" in DEFAULT_PARAM_GRIDS
        assert "atr" in DEFAULT_PARAM_GRIDS
        assert "bbands" in DEFAULT_PARAM_GRIDS

    def test_rsi_periods(self) -> None:
        """Test RSI has sensible period values (Fibonacci-based)."""
        rsi_grid = DEFAULT_PARAM_GRIDS["rsi"]
        periods = rsi_grid["period"]

        assert 2 in periods  # Short-term
        assert 13 in periods  # Near-standard (Fibonacci closest to 14)
        assert 21 in periods or 55 in periods  # Longer-term

    def test_all_values_are_lists(self) -> None:
        """Test all parameter values are lists."""
        for name, grid in DEFAULT_PARAM_GRIDS.items():
            for param_name, values in grid.items():
                assert isinstance(values, list), f"{name}.{param_name} is not a list"


class TestGenerateFibonacciPeriodPairs:
    """Tests for generate_fibonacci_period_pairs function."""

    def test_coarse_pairs(self) -> None:
        """Test coarse mode generates expected pairs."""
        pairs = generate_fibonacci_period_pairs(use_coarse=True)

        # Should have 6 pairs from [5, 8, 21, 55]
        expected = [(5, 8), (5, 21), (5, 55), (8, 21), (8, 55), (21, 55)]
        assert pairs == expected

    def test_all_pairs_have_fast_less_than_slow(self) -> None:
        """Test all pairs satisfy fast < slow constraint."""
        pairs = generate_fibonacci_period_pairs(use_coarse=True)

        for fast, slow in pairs:
            assert fast < slow, f"Invalid pair: {fast} >= {slow}"

    def test_custom_periods(self) -> None:
        """Test with custom period list."""
        pairs = generate_fibonacci_period_pairs(periods=[3, 5, 8, 13])

        expected = [(3, 5), (3, 8), (3, 13), (5, 8), (5, 13), (8, 13)]
        assert pairs == expected

    def test_empty_periods(self) -> None:
        """Test with empty or single period list."""
        pairs = generate_fibonacci_period_pairs(periods=[5])
        assert pairs == []

        pairs = generate_fibonacci_period_pairs(periods=[])
        assert pairs == []

    def test_two_periods(self) -> None:
        """Test with exactly two periods."""
        pairs = generate_fibonacci_period_pairs(periods=[5, 21])
        assert pairs == [(5, 21)]

    def test_extended_mode(self) -> None:
        """Test extended mode produces valid pairs."""
        pairs = generate_fibonacci_period_pairs(use_extended=True)

        # Should have many pairs
        assert len(pairs) > 50

        # All should satisfy constraint
        for fast, slow in pairs:
            assert fast < slow


class TestCoarsePeriodVariations:
    """Tests for coarse period variation constants."""

    def test_coarse_periods_are_fibonacci(self) -> None:
        """Test coarse periods are from Fibonacci sequence."""
        fibonacci = {2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233}

        for period in PERIOD_VARIATIONS_COARSE:
            assert period in fibonacci, f"{period} is not a Fibonacci number"

    def test_fast_slow_cover_all_valid_pairs(self) -> None:
        """Test FAST and SLOW variations can generate all coarse pairs."""
        # Generate pairs from FAST x SLOW Cartesian product
        cartesian_pairs = [
            (fast, slow)
            for fast in FAST_PERIOD_VARIATIONS_COARSE
            for slow in SLOW_PERIOD_VARIATIONS_COARSE
            if fast < slow
        ]

        # Generate pairs using the helper function
        fibonacci_pairs = generate_fibonacci_period_pairs(use_coarse=True)

        # Cartesian should cover all Fibonacci pairs
        assert set(cartesian_pairs) == set(fibonacci_pairs)

    def test_fast_variations_coarse(self) -> None:
        """Test FAST_PERIOD_VARIATIONS_COARSE values."""
        assert FAST_PERIOD_VARIATIONS_COARSE == [5, 8, 21]

    def test_slow_variations_coarse(self) -> None:
        """Test SLOW_PERIOD_VARIATIONS_COARSE values."""
        assert SLOW_PERIOD_VARIATIONS_COARSE == [8, 21, 55]
