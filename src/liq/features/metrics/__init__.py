"""Additional information-theoretic metrics for feature selection.

This module provides wrappers for alternative dependency measures:
- Transfer Entropy (asymmetric, time-lagged)
- Maximal Information Coefficient (MIC)
- Distance Correlation (dCor)

These complement Mutual Information to provide robustness checks.

Example:
    >>> import numpy as np
    >>> from liq.features.metrics import (
    ...     transfer_entropy,
    ...     maximal_information_coefficient,
    ...     distance_correlation,
    ... )
    >>>
    >>> # Check if Transfer Entropy is available
    >>> te = transfer_entropy(source, target, k=1)
    >>>
    >>> # MIC for general non-linear relationships
    >>> mic = maximal_information_coefficient(x, y)
    >>>
    >>> # Distance correlation (0 iff independent)
    >>> dcor = distance_correlation(x, y)
"""

from __future__ import annotations

__all__: list[str] = []
