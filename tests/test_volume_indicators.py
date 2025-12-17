"""Tests for volume-based indicators."""

from datetime import UTC, datetime

import polars as pl
import pytest

from liq.features.indicators import get_indicator


class TestAbnormalTurnover:
    """Tests for AbnormalTurnover indicator."""

    def test_abnormal_turnover_default_params(self) -> None:
        """Test AbnormalTurnover default parameters."""
        AbnormalTurnover = get_indicator("abnormal_turnover")
        indicator = AbnormalTurnover()
        assert indicator._params["window"] == 55

    def test_abnormal_turnover_custom_params(self) -> None:
        """Test AbnormalTurnover with custom window."""
        AbnormalTurnover = get_indicator("abnormal_turnover")
        indicator = AbnormalTurnover(params={"window": 20})
        assert indicator._params["window"] == 20

    def test_abnormal_turnover_compute(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test AbnormalTurnover computation."""
        AbnormalTurnover = get_indicator("abnormal_turnover")
        indicator = AbnormalTurnover(params={"window": 10})
        result = indicator.compute(sample_ohlc_df_large)

        # Should have ts and value columns
        assert "ts" in result.columns
        assert "value" in result.columns

        # Should have no NaN values in result
        assert result.filter(pl.col("value").is_nan()).height == 0

        # Z-scores should be reasonable (most within -3 to 3)
        values = result["value"].to_list()
        assert all(-10 < v < 10 for v in values)

    def test_abnormal_turnover_spike_detection(self) -> None:
        """Test that AbnormalTurnover detects volume spikes."""
        # Create data with a volume spike
        n = 50
        timestamps = [
            datetime(2024, 1, 1, i // 24, i % 24, 0, tzinfo=UTC) for i in range(n)
        ]
        volumes = [1000.0] * n
        volumes[40] = 5000.0  # Spike at position 40

        df = pl.DataFrame({
            "ts": timestamps,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": volumes,
        })

        AbnormalTurnover = get_indicator("abnormal_turnover")
        indicator = AbnormalTurnover(params={"window": 10})
        result = indicator.compute(df)

        # Find the z-score at the spike position
        spike_result = result.filter(pl.col("ts") == timestamps[40])
        if spike_result.height > 0:
            spike_zscore = spike_result["value"][0]
            # Should have high positive z-score for the spike
            assert spike_zscore > 2.0

    def test_abnormal_turnover_missing_volume_error(self) -> None:
        """Test ValueError when volume column is missing."""
        df = pl.DataFrame({
            "ts": [datetime(2024, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        })

        AbnormalTurnover = get_indicator("abnormal_turnover")
        indicator = AbnormalTurnover(params={"window": 10})

        with pytest.raises(ValueError, match="volume"):
            indicator.compute(df)

    def test_abnormal_turnover_repr(self) -> None:
        """Test string representation."""
        AbnormalTurnover = get_indicator("abnormal_turnover")
        indicator = AbnormalTurnover(params={"window": 20})
        repr_str = repr(indicator)
        assert "abnormalturnover" in repr_str.lower()
        assert "window=20" in repr_str.lower()


class TestNormalizedVolume:
    """Tests for NormalizedVolume indicator."""

    def test_normalized_volume_default_params(self) -> None:
        """Test NormalizedVolume default parameters."""
        NormalizedVolume = get_indicator("normalized_volume")
        indicator = NormalizedVolume()
        assert indicator._params["window"] == 55

    def test_normalized_volume_custom_params(self) -> None:
        """Test NormalizedVolume with custom window."""
        NormalizedVolume = get_indicator("normalized_volume")
        indicator = NormalizedVolume(params={"window": 30})
        assert indicator._params["window"] == 30

    def test_normalized_volume_compute(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test NormalizedVolume computation."""
        NormalizedVolume = get_indicator("normalized_volume")
        indicator = NormalizedVolume(params={"window": 10})
        result = indicator.compute(sample_ohlc_df_large)

        # Should have ts and value columns
        assert "ts" in result.columns
        assert "value" in result.columns

        # Should have no NaN values in result
        assert result.filter(pl.col("value").is_nan()).height == 0

        # Values should be positive (volume ratio)
        values = result["value"].to_list()
        assert all(v > 0 for v in values)

    def test_normalized_volume_stable_volume(self) -> None:
        """Test that stable volume gives ratio near 1.0."""
        n = 50
        timestamps = [
            datetime(2024, 1, 1, i // 24, i % 24, 0, tzinfo=UTC) for i in range(n)
        ]

        df = pl.DataFrame({
            "ts": timestamps,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,  # Constant volume
        })

        NormalizedVolume = get_indicator("normalized_volume")
        indicator = NormalizedVolume(params={"window": 10})
        result = indicator.compute(df)

        # With constant volume, ratio should be exactly 1.0
        values = result["value"].to_list()
        assert all(abs(v - 1.0) < 0.001 for v in values)

    def test_normalized_volume_multiple_windows(
        self, sample_ohlc_df_large: pl.DataFrame
    ) -> None:
        """Test NormalizedVolume with different window sizes."""
        NormalizedVolume = get_indicator("normalized_volume")

        for window in [10, 20, 30]:
            indicator = NormalizedVolume(params={"window": window})
            result = indicator.compute(sample_ohlc_df_large)
            assert result.height > 0
            assert result.filter(pl.col("value").is_nan()).height == 0

    def test_normalized_volume_missing_volume_error(self) -> None:
        """Test ValueError when volume column is missing."""
        df = pl.DataFrame({
            "ts": [datetime(2024, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        })

        NormalizedVolume = get_indicator("normalized_volume")
        indicator = NormalizedVolume(params={"window": 10})

        with pytest.raises(ValueError, match="volume"):
            indicator.compute(df)

    def test_normalized_volume_repr(self) -> None:
        """Test string representation."""
        NormalizedVolume = get_indicator("normalized_volume")
        indicator = NormalizedVolume(params={"window": 30})
        repr_str = repr(indicator)
        assert "normalizedvolume" in repr_str.lower()
        assert "window=30" in repr_str.lower()
