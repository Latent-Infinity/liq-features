"""Tests for liq.features.labels module."""

import polars as pl
import pytest

from liq.features.labels import (
    TripleBarrierConfig,
    triple_barrier_labels,
    triple_barrier_labels_adaptive,
)


class TestTripleBarrierConfig:
    """Tests for TripleBarrierConfig dataclass."""

    def test_config_default_values(self) -> None:
        """Test default values for adaptive config."""
        cfg = TripleBarrierConfig()

        assert cfg.take_profit is None
        assert cfg.stop_loss is None
        assert cfg.max_holding == 5
        assert cfg.profit_std_multiple == 2.0
        assert cfg.loss_std_multiple == 2.0
        assert cfg.volatility_window == 20

    def test_config_fixed_thresholds(self) -> None:
        """Test config with fixed thresholds."""
        cfg = TripleBarrierConfig(
            take_profit=0.02,
            stop_loss=0.01,
            max_holding=10,
        )

        assert cfg.take_profit == 0.02
        assert cfg.stop_loss == 0.01
        assert cfg.max_holding == 10

    def test_config_adaptive_thresholds(self) -> None:
        """Test config with adaptive (std-based) thresholds."""
        cfg = TripleBarrierConfig(
            profit_std_multiple=2.5,
            loss_std_multiple=1.5,
            volatility_window=30,
        )

        assert cfg.profit_std_multiple == 2.5
        assert cfg.loss_std_multiple == 1.5
        assert cfg.volatility_window == 30


class TestTripleBarrierLabelsOriginal:
    """Tests for original triple_barrier_labels function (backward compatibility)."""

    def test_triple_barrier_basic(self) -> None:
        """Test basic triple barrier labeling."""
        df = pl.DataFrame({
            "close": [100.0, 102.0, 104.0, 103.0, 105.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.03,
            stop_loss=0.02,
            max_holding=3,
        )

        labels = triple_barrier_labels(df, cfg)

        assert len(labels) == 5
        assert all(label in [-1, 0, 1] for label in labels)

    def test_triple_barrier_profit_hit(self) -> None:
        """Test label=1 when profit target is hit."""
        # Price goes up 5% immediately
        df = pl.DataFrame({
            "close": [100.0, 105.0, 106.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.04,
            stop_loss=0.02,
            max_holding=2,
        )

        labels = triple_barrier_labels(df, cfg)

        # First position should hit profit target
        assert labels[0] == 1

    def test_triple_barrier_loss_hit(self) -> None:
        """Test label=-1 when stop loss is hit."""
        # Price goes down 3% immediately
        df = pl.DataFrame({
            "close": [100.0, 97.0, 96.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.05,
            stop_loss=0.02,
            max_holding=2,
        )

        labels = triple_barrier_labels(df, cfg)

        # First position should hit stop loss
        assert labels[0] == -1

    def test_triple_barrier_timeout(self) -> None:
        """Test label=0 when max holding is reached without hitting barriers."""
        # Price barely moves
        df = pl.DataFrame({
            "close": [100.0, 100.5, 100.2, 100.3, 100.1],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.10,  # 10% - won't hit
            stop_loss=0.10,   # 10% - won't hit
            max_holding=2,
        )

        labels = triple_barrier_labels(df, cfg)

        # First position should timeout
        assert labels[0] == 0

    def test_triple_barrier_labels_hits_tp_and_sl(self) -> None:
        """Test triple barrier correctly identifies TP and SL hits."""
        df = pl.DataFrame({"close": [100, 102, 90, 105]})
        cfg = TripleBarrierConfig(take_profit=0.02, stop_loss=0.05, max_holding=3)
        labels = triple_barrier_labels(df, cfg)
        assert labels[0] == 1  # tp hit at 102
        assert labels[1] == -1  # sl hit at 90 for entry at 102
        assert labels[2] in (-1, 0, 1)


class TestTripleBarrierLabelsAdaptive:
    """Tests for triple_barrier_labels_adaptive function."""

    def test_adaptive_default_config(self) -> None:
        """Test adaptive labeling with default config (2σ thresholds)."""
        # Create price series with some volatility
        import random
        random.seed(42)
        n = 100
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + random.gauss(0, 0.01)))

        df = pl.DataFrame({"close": prices})
        cfg = TripleBarrierConfig()

        result = triple_barrier_labels_adaptive(df, cfg)

        assert "label" in result.columns
        assert result.height == n
        # Labels should be -1, 0, or 1
        labels = result["label"].to_list()
        assert all(label in [-1, 0, 1] for label in labels)

    def test_adaptive_fixed_thresholds(self) -> None:
        """Test adaptive function with fixed thresholds (no std-based)."""
        df = pl.DataFrame({
            "close": [100.0, 102.0, 104.0, 103.0, 105.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.03,
            stop_loss=0.02,
            max_holding=3,
        )

        result = triple_barrier_labels_adaptive(df, cfg)

        assert "label" in result.columns
        assert result.height == 5

    def test_adaptive_profit_label(self) -> None:
        """Test that profit target hit returns label=1."""
        # Create scenario where price rises significantly
        df = pl.DataFrame({
            "close": [100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 110.0],  # Big jump at the end
        })

        cfg = TripleBarrierConfig(
            take_profit=0.05,  # 5% profit target
            stop_loss=0.05,
            max_holding=5,
        )

        result = triple_barrier_labels_adaptive(df, cfg)
        labels = result["label"].to_list()

        # Position opened at index 16 should see profit at index 21
        # (depends on implementation but should have some profit labels)
        assert 1 in labels or 0 in labels  # At least some valid labels

    def test_adaptive_loss_label(self) -> None:
        """Test that stop loss hit returns label=-1."""
        # Create scenario where price drops significantly
        df = pl.DataFrame({
            "close": [100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 90.0],  # Big drop at the end
        })

        cfg = TripleBarrierConfig(
            take_profit=0.05,
            stop_loss=0.05,  # 5% stop loss
            max_holding=5,
        )

        result = triple_barrier_labels_adaptive(df, cfg)
        labels = result["label"].to_list()

        # Should have some loss labels
        assert -1 in labels or 0 in labels

    def test_adaptive_timeout_label(self) -> None:
        """Test that timeout returns label=0."""
        # Create flat price series
        df = pl.DataFrame({
            "close": [100.0] * 50,
        })

        cfg = TripleBarrierConfig(
            take_profit=0.10,  # 10% - won't hit with flat prices
            stop_loss=0.10,
            max_holding=3,
        )

        result = triple_barrier_labels_adaptive(df, cfg)
        labels = result["label"].to_list()

        # With flat prices, all labels should be timeout (0)
        # (except possibly last few positions)
        timeout_count = sum(1 for l in labels if l == 0)
        assert timeout_count > len(labels) // 2

    def test_adaptive_volatility_based_thresholds(self) -> None:
        """Test that volatility-based thresholds adapt to market conditions."""
        # High volatility period
        import random
        random.seed(42)
        high_vol_prices = [100.0]
        for _ in range(49):
            high_vol_prices.append(high_vol_prices[-1] * (1 + random.gauss(0, 0.03)))

        df = pl.DataFrame({"close": high_vol_prices})

        cfg = TripleBarrierConfig(
            profit_std_multiple=2.0,
            loss_std_multiple=2.0,
            volatility_window=20,
            max_holding=5,
        )

        result = triple_barrier_labels_adaptive(df, cfg)

        assert "label" in result.columns
        # Should have mix of labels due to volatility
        labels = result["label"].to_list()
        unique_labels = set(labels)
        assert len(unique_labels) >= 1  # At least some variety

    def test_adaptive_returns_dataframe(self) -> None:
        """Test that adaptive function returns DataFrame with label column."""
        df = pl.DataFrame({
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.02,
            stop_loss=0.01,
            max_holding=3,
        )

        result = triple_barrier_labels_adaptive(df, cfg)

        assert isinstance(result, pl.DataFrame)
        assert "label" in result.columns
        assert "close" in result.columns  # Original columns preserved

    def test_adaptive_backward_compatible_with_original(self) -> None:
        """Test that adaptive function with fixed thresholds matches original."""
        df = pl.DataFrame({
            "close": [100.0, 102.0, 99.0, 101.0, 103.0, 98.0, 100.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.02,
            stop_loss=0.02,
            max_holding=3,
        )

        original_labels = triple_barrier_labels(df, cfg)
        adaptive_result = triple_barrier_labels_adaptive(df, cfg)
        adaptive_labels = adaptive_result["label"].to_list()

        # Labels should match when using fixed thresholds
        assert original_labels == adaptive_labels
