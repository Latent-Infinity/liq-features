"""Tests for FeatureDefinition and FeatureSet."""

from liq.features.feature_set import FeatureDefinition, FeatureSet


def test_feature_set_max_lookback() -> None:
    f1 = FeatureDefinition(name="a", func=lambda df, col: df, lookback=2)
    f2 = FeatureDefinition(name="b", func=lambda df, col: df, lookback=5)
    fs = FeatureSet(name="test", features=[f1, f2])
    assert fs.max_lookback == 5
