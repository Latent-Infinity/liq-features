"""Tests for the optional metrics namespace."""

import liq.features.metrics as metrics


def test_metrics_namespace_is_explicitly_empty_until_optional_metrics_ship() -> None:
    """The metrics package currently exposes no public helpers."""
    assert metrics.__all__ == []
