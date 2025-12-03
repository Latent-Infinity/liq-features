from liq.features.regime import hurst_exponent


def test_hurst_exponent_range() -> None:
    series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    h = hurst_exponent(series)
    assert 0 <= h <= 1
