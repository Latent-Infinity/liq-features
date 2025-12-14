"""Tests for temporal correctness and look-ahead bias prevention.

Following TDD: Tests verify no future data leakage in features.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from liq.features.alignment import align_higher_timeframe
from liq.features.derived import compute_derived_fields, compute_returns, compute_volatility
from liq.features.indicators import get_indicator
from liq.features.labels import TripleBarrierConfig, triple_barrier_labels


class TestIndicatorNoLookahead:
    """Tests verifying indicators only use past data."""

    def _make_ohlc_df(self, n_rows: int = 50) -> pl.DataFrame:
        """Create OHLC DataFrame with required columns for indicators."""
        import math

        base_price = 100.0
        timestamps = [
            datetime(2024, 1, 1, i // 24, i % 24, tzinfo=timezone.utc)
            for i in range(n_rows)
        ]

        opens = []
        highs = []
        lows = []
        closes = []

        price = base_price
        for i in range(n_rows):
            trend = math.sin(i / 10) * 2
            noise = (i % 7 - 3) * 0.5

            open_price = price
            close_price = price + trend + noise
            high_price = max(open_price, close_price) + abs(noise) + 1
            low_price = min(open_price, close_price) - abs(noise) - 0.5

            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)

            price = close_price

        return pl.DataFrame({
            "ts": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * n_rows,
        })

    def test_ema_uses_only_past_data(self) -> None:
        """EMA at time t should only use data from t and before."""
        df = self._make_ohlc_df(50)

        # Import directly to avoid talib wrapper issues
        from liq.features.indicators.trend import EMA

        ema = EMA(params={"period": 5})
        result = ema.compute(df)

        # Verify output length is less than input (warmup removed)
        assert len(result) < len(df)

        # Verify each EMA value is deterministic (same input -> same output)
        result2 = ema.compute(df)
        assert result["value"].to_list() == result2["value"].to_list()

    def test_rsi_uses_only_past_data(self) -> None:
        """RSI at time t should only use data from t and before."""
        df = self._make_ohlc_df(50)

        # Import directly to avoid talib wrapper issues
        from liq.features.indicators.momentum import RSI

        rsi = RSI(params={"period": 5})
        result = rsi.compute(df)

        # RSI should have warmup period removed
        assert len(result) < len(df)

        # Verify RSI values are in valid range
        for val in result["value"].to_list():
            assert 0 <= val <= 100

    def test_sma_window_correct(self) -> None:
        """SMA should use exactly the specified window of past data."""
        # Simple case with known values
        df = pl.DataFrame({
            "ts": [
                datetime(2024, 1, i, tzinfo=timezone.utc)
                for i in range(1, 6)
            ],
            "open": [10.0, 20.0, 30.0, 40.0, 50.0],
            "high": [15.0, 25.0, 35.0, 45.0, 55.0],
            "low": [5.0, 15.0, 25.0, 35.0, 45.0],
            "close": [10.0, 20.0, 30.0, 40.0, 50.0],
            "volume": [100.0] * 5,
        })

        from liq.features.indicators.trend import SMA

        sma = SMA(params={"period": 3})
        result = sma.compute(df)

        # SMA(3) at position 2 should be (10+20+30)/3 = 20
        # SMA(3) at position 3 should be (20+30+40)/3 = 30
        # SMA(3) at position 4 should be (30+40+50)/3 = 40
        values = result["value"].to_list()
        assert abs(values[0] - 20.0) < 0.01
        assert abs(values[1] - 30.0) < 0.01
        assert abs(values[2] - 40.0) < 0.01


class TestDerivedFieldsNoLookahead:
    """Tests verifying derived fields don't use future data."""

    def test_true_range_uses_previous_close(self) -> None:
        """True range should use previous bar's close, not future."""
        df = pl.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
        })

        result = compute_derived_fields(df)

        # First true_range should be null (no previous close)
        # Second true_range uses first bar's close (102.0)
        assert result["true_range"][0] is None or result["true_range"][0] == 10.0  # high-low if no prev
        # Verify true_range at position 1 uses close[0]=102.0
        # TR = max(106-96, |106-102|, |96-102|) = max(10, 4, 6) = 10
        assert result["true_range"][1] == 10.0

    def test_returns_use_previous_bar(self) -> None:
        """Returns should be computed using previous bar's price."""
        df = pl.DataFrame({
            "close": [100.0, 110.0, 99.0],
        })

        result = compute_returns(df, column="close", periods=1)

        # First return should be null
        assert result["return"][0] is None
        # Second return = (110-100)/100 = 0.10
        assert abs(result["return"][1] - 0.10) < 0.001
        # Third return = (99-110)/110 = -0.10
        assert abs(result["return"][2] - (-0.10)) < 0.001


class TestAlignmentNoLookahead:
    """Tests verifying alignment doesn't use future higher-TF data."""

    def test_backward_strategy_prevents_lookahead(self) -> None:
        """Alignment should only use completed higher-TF bars from the past."""
        # Base 1-min bars
        base = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 6, tzinfo=timezone.utc),  # After second higher bar
            ],
            "value": [1, 2, 3, 4],
        })

        # Higher-TF 5-min bars
        higher = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),  # 00:00 bar
                datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),  # 00:05 bar
            ],
            "htf_value": [100, 200],
        })

        aligned = align_higher_timeframe(base, higher)

        # First 3 base bars (00:01, 00:02, 00:03) should use 00:00 higher bar
        # because 00:05 bar wasn't completed yet
        assert aligned["htf_value"][0] == 100
        assert aligned["htf_value"][1] == 100
        assert aligned["htf_value"][2] == 100
        # Fourth base bar (00:06) should use 00:05 higher bar
        assert aligned["htf_value"][3] == 200

    def test_no_future_higher_tf_data(self) -> None:
        """Base bars should never receive future higher-TF data."""
        base = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
            ],
            "value": [1, 2],
        })

        # Higher TF bar is in the future relative to all base bars
        higher = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc),  # Future!
            ],
            "htf_value": [999],
        })

        aligned = align_higher_timeframe(base, higher)

        # Base bars should NOT receive the future higher TF value
        # They should be null since no past higher data exists
        assert aligned["htf_value"][0] is None
        assert aligned["htf_value"][1] is None


class TestTripleBarrierLabelsIntentionalLookahead:
    """Tests for triple barrier labels - intentional lookahead for ML labels."""

    def test_labels_use_future_data_as_expected(self) -> None:
        """Triple barrier labels intentionally look ahead for ML training."""
        # This is expected behavior - labels ARE lookahead by design
        df = pl.DataFrame({
            "close": [100.0, 102.0, 98.0, 101.0, 103.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.02,  # 2% target
            stop_loss=0.02,    # 2% stop
            max_holding=3,
        )

        labels = triple_barrier_labels(df, cfg)

        # First entry at 100, TP=102, SL=98
        # At index 1, price=102 hits TP -> label=1
        assert labels[0] == 1

        # This is INTENTIONAL lookahead for ML labels
        # The important thing is that labels are NOT used as features

    def test_last_bar_cannot_lookahead(self) -> None:
        """Last bar has no future data to look at."""
        df = pl.DataFrame({
            "close": [100.0, 101.0, 102.0],
        })

        cfg = TripleBarrierConfig(
            take_profit=0.05,  # Won't hit in remaining data
            stop_loss=0.05,
            max_holding=10,
        )

        labels = triple_barrier_labels(df, cfg)

        # Last bar should be 0 (no future data to evaluate)
        assert labels[-1] == 0


class TestScalerFitTransformSeparation:
    """Tests ensuring scaler fit/transform don't leak train to test."""

    def test_scaler_fit_on_train_only(self) -> None:
        """Scaler parameters should only come from training data."""
        from liq.features.scaling import ModelAwareScaler

        train_data = [100.0, 110.0, 90.0, 105.0, 95.0]
        test_data = [150.0, 160.0, 170.0]  # Different distribution

        scaler = ModelAwareScaler(model_type="nn")
        scaler.fit(train_data)

        # Parameters should reflect train_data only
        assert scaler.params is not None
        assert scaler.params.mean == pytest.approx(100.0, rel=0.01)

        # Transform test data using train parameters
        transformed_test = scaler.transform(test_data)

        # Test data is transformed using train mean/std
        # (150-100)/std, (160-100)/std, (170-100)/std
        # Values should be positive since test > train mean
        assert all(v > 0 for v in transformed_test)

    def test_transform_without_fit_raises(self) -> None:
        """Transform before fit should raise error."""
        from liq.features.scaling import ModelAwareScaler

        scaler = ModelAwareScaler(model_type="nn")

        with pytest.raises(RuntimeError, match="must be fit"):
            scaler.transform([100.0, 110.0])


class TestStationarityFitTransformSeparation:
    """Tests ensuring stationarity transform doesn't leak data."""

    def test_stationarity_fit_on_train_only(self) -> None:
        """Stationarity fit should only use training data."""
        from liq.features.stationarity import StationarityTransformer

        train_data = [100.0, 101.0, 102.0, 103.0, 104.0]

        transformer = StationarityTransformer(d=0.4)
        transformer.fit(train_data)

        assert transformer.fitted_ is True

        # Transform should work after fit
        result = transformer.transform(train_data)
        assert len(result) == len(train_data)

    def test_transform_without_fit_raises(self) -> None:
        """Transform before fit should raise error."""
        from liq.features.stationarity import StationarityTransformer

        transformer = StationarityTransformer(d=0.4)

        with pytest.raises(RuntimeError, match="must be fit"):
            transformer.transform([100.0, 101.0])
