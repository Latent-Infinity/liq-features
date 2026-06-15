"""Realized variance / bipower variation / jump variation helpers.

The minute-mode estimators (``compute_rv``, ``compute_bpv``, ``compute_jv``,
``rv_noise_gate``) implement research plan §5.3 + ``[APPENDIX_FORMULAS]``
and are added when the RV-noise gate is wired up. Today the module is a
skeleton so the import surface is stable.
"""

from __future__ import annotations
