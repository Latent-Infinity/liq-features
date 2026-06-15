"""Variance decomposition (`cont` / `overnight_gap` / `intraday_range` /
`jump`) and the derived signals that populate ``VolEstimate.components``.

The module is importable today so callers can rely on the surface;
the windowed aggregators land incrementally per research plan
``[DESIGN_DECOMPOSITION]``.
"""

from __future__ import annotations
