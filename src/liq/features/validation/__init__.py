"""Validation and robustness analysis for feature selection.

This module provides tools for validating MI-based feature selection:
- Out-of-sample validation (train/test MI stability)
- Effect size calculations (Cohen's d)
- MI estimator sensitivity analysis (k-NN parameter sensitivity)
- Temporal stability analysis (rolling windows)
- Model comparison framework (LightGBM close vs midrange)

Example:
    >>> import polars as pl
    >>> from liq.features.validation import (
    ...     validate_oos,
    ...     cohens_d,
    ...     mi_sensitivity_analysis,
    ...     rolling_mi_analysis,
    ... )
    >>>
    >>> # Out-of-sample validation
    >>> oos_result = validate_oos(X, y, features, test_ratio=0.2)
    >>>
    >>> # Effect size between close and midrange MI
    >>> effect = cohens_d(close_mi_scores, midrange_mi_scores)
"""

from __future__ import annotations

# Protocols and interfaces
from liq.features.validation.protocols import (
    MetricCalculator,
    TemporalAnalyzer,
    Validator,
)

# Result dataclasses
from liq.features.validation.results import (
    EffectSizeResult,
    ModelComparisonResult,
    OutOfSampleResult,
    SensitivityResult,
    TemporalStabilityResult,
)

# Exceptions
from liq.features.validation.exceptions import (
    ConfigurationError,
    ConvergenceError,
    InsufficientDataError,
    ValidationError,
)

# Effect size calculations
from liq.features.validation.effect_size import (
    batch_cohens_d,
    cohens_d,
    cohens_d_ci,
    pooled_std,
)

# Sensitivity analysis
from liq.features.validation.sensitivity import (
    batch_sensitivity_analysis,
    mi_sensitivity_analysis,
)

# Out-of-sample validation
from liq.features.validation.out_of_sample import validate_oos

# Temporal stability
from liq.features.validation.temporal import rolling_mi_analysis

__all__ = [
    # Protocols
    "Validator",
    "MetricCalculator",
    "TemporalAnalyzer",
    # Result types
    "OutOfSampleResult",
    "ModelComparisonResult",
    "EffectSizeResult",
    "SensitivityResult",
    "TemporalStabilityResult",
    # Exceptions
    "ValidationError",
    "InsufficientDataError",
    "ConvergenceError",
    "ConfigurationError",
    # Effect size
    "cohens_d",
    "cohens_d_ci",
    "pooled_std",
    "batch_cohens_d",
    # Sensitivity analysis
    "mi_sensitivity_analysis",
    "batch_sensitivity_analysis",
    # Out-of-sample validation
    "validate_oos",
    # Temporal stability
    "rolling_mi_analysis",
]
