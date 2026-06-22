"""Regime label type contracts for mean-reversion features."""

from __future__ import annotations

from typing import Literal

RegimeLabel = Literal["trend", "chop", "indeterminate"]

__all__ = ["RegimeLabel"]
