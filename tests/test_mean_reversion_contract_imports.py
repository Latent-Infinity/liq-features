from __future__ import annotations

from importlib import import_module


def test_trailing_range_vol_contract_imports() -> None:
    module_exports = vars(import_module("liq.features.mean_reversion.vol"))

    assert callable(module_exports["trailing_range_vol"])


def test_midrange_base_helper_contract_imports() -> None:
    module_exports = vars(import_module("liq.features.mean_reversion.base"))

    assert callable(module_exports["roll_extreme_midrange"])
    assert callable(module_exports["roll_mean_midrange"])


def test_regime_label_contract_imports() -> None:
    module_exports = vars(import_module("liq.features.mean_reversion.regime"))

    assert module_exports["RegimeLabel"] is not None
