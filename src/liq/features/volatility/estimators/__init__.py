"""Per-bar variance contribution registry + windowed aggregators.

The formula registry implements research plan ``[APPENDIX_FORMULAS]``;
the fallback chain implements §4.2. Today this subpackage's public
surface is intentionally empty — the registry is added incrementally.
"""

from __future__ import annotations
