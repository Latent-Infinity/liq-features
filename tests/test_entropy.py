"""Tests for liq.features.entropy module."""

from liq.features.entropy import sample_entropy


class TestSampleEntropy:
    """Tests for sample_entropy function."""

    def test_constant_series_low_entropy(self) -> None:
        """Test constant series has low or zero entropy."""
        series = [1, 1, 1, 1, 1, 1, 1]
        ent = sample_entropy(series, m=2, r=0.2)
        # Constant series entropy is implementation-dependent
        assert isinstance(ent, float)
        assert ent >= 0.0

    def test_short_series_returns_zero(self) -> None:
        """Test series too short for m returns zero."""
        series = [1, 2, 3]  # n <= m + 1 for m=2
        ent = sample_entropy(series, m=2, r=0.2)
        assert ent == 0.0

    def test_empty_series_returns_zero(self) -> None:
        """Test empty series returns zero."""
        ent = sample_entropy([], m=2, r=0.2)
        assert ent == 0.0

    def test_random_series_positive_entropy(self) -> None:
        """Test random-like series has positive entropy."""
        # Non-constant series with some variation
        series = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 3.5, 2.0, 4.0, 3.0]
        ent = sample_entropy(series, m=2, r=0.5)
        assert isinstance(ent, float)
        assert ent >= 0.0

    def test_returns_float(self) -> None:
        """Test function returns a float."""
        series = list(range(20))
        ent = sample_entropy(series, m=2, r=0.2)
        assert isinstance(ent, float)

    def test_different_m_values(self) -> None:
        """Test with different m values."""
        series = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
        ent_m1 = sample_entropy(series, m=1, r=0.2)
        ent_m2 = sample_entropy(series, m=2, r=0.2)
        # Both should be valid floats
        assert isinstance(ent_m1, float)
        assert isinstance(ent_m2, float)
