from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="mean-reversion trailing volatility helper lands with feature implementation",
)
def test_trailing_range_vol_contract_imports() -> None:
    module_exports = vars(import_module("liq.features.mean_reversion.vol"))

    assert callable(module_exports["trailing_range_vol"])


@pytest.mark.xfail(
    strict=True,
    reason="mean-reversion midrange base helpers land with feature implementation",
)
def test_midrange_base_helper_contract_imports() -> None:
    module_exports = vars(import_module("liq.features.mean_reversion.base"))

    assert callable(module_exports["roll_extreme_midrange"])
    assert callable(module_exports["roll_mean_midrange"])
