"""Calendar adapter — re-exports the helpers from ``liq.data.calendar``
that the volatility decomposition consumes (``closed_hours_between`` and
``classify_gap``). No duplication of trading-calendar logic here; this
module exists to give the volatility package a single import point for
calendar concerns.
"""

from __future__ import annotations

from liq.data.calendar import GapClass, classify_gap, closed_hours_between

__all__ = ["GapClass", "classify_gap", "closed_hours_between"]
