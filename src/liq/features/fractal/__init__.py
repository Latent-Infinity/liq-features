"""Fractal-codec feature diagnostics for the LIQ features layer.

Thin consumer of the reference 1D compression codec in ``liq.fractal``. This
subpackage ships diagnostic-only smoke tooling; it makes no economic claim and
touches no market-data provider.

Import the ``codec_demo`` module directly (e.g. ``python -m
liq.features.fractal.codec_demo``); this package init stays import-light so the
module runs cleanly under ``-m``.
"""

from __future__ import annotations
