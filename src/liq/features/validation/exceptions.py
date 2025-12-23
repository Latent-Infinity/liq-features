"""Custom exceptions for validation and robustness analysis.

Provides a hierarchy of exceptions for handling validation errors
with contextual information for debugging.
"""

from __future__ import annotations

from typing import Any


class ValidationError(Exception):
    """Base exception for validation errors.

    All validation-specific exceptions inherit from this class,
    allowing callers to catch all validation errors with a single except.

    Attributes:
        message: Human-readable error message.
        context: Additional context dictionary.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize validation error.

        Args:
            message: Human-readable error message.
            context: Additional context for debugging.
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        if not self.context:
            return self.message
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} ({context_str})"


class InsufficientDataError(ValidationError):
    """Raised when sample size is too small for reliable analysis.

    This occurs when:
    - Not enough samples for train/test split
    - Window size exceeds available data
    - Bootstrap/permutation iterations exceed sample count

    Attributes:
        required: Minimum required sample size.
        actual: Actual sample size.
    """

    def __init__(
        self,
        message: str,
        required: int,
        actual: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize insufficient data error.

        Args:
            message: Human-readable error message.
            required: Minimum required sample size.
            actual: Actual sample size.
            context: Additional context for debugging.
        """
        self.required = required
        self.actual = actual
        full_context = {"required": required, "actual": actual}
        if context:
            full_context.update(context)
        super().__init__(message, full_context)


class ConvergenceError(ValidationError):
    """Raised when MI estimation fails to converge.

    This can occur when:
    - KSG estimator encounters numerical issues
    - Zero variance in features or targets
    - All values are identical

    Attributes:
        feature: Feature name that failed.
        iteration: Iteration number where failure occurred.
    """

    def __init__(
        self,
        message: str,
        feature: str | None = None,
        iteration: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize convergence error.

        Args:
            message: Human-readable error message.
            feature: Feature name that failed.
            iteration: Iteration number where failure occurred.
            context: Additional context for debugging.
        """
        self.feature = feature
        self.iteration = iteration
        full_context = {}
        if feature is not None:
            full_context["feature"] = feature
        if iteration is not None:
            full_context["iteration"] = iteration
        if context:
            full_context.update(context)
        super().__init__(message, full_context)


class ConfigurationError(ValidationError):
    """Raised when configuration parameters are invalid.

    This occurs when:
    - Invalid k-NN values (k < 1)
    - Invalid split ratios (not in (0, 1))
    - Invalid window sizes (non-positive)
    - Missing required parameters

    Attributes:
        parameter: Parameter name that is invalid.
        value: Invalid value provided.
        valid_range: Description of valid values.
    """

    def __init__(
        self,
        message: str,
        parameter: str,
        value: Any,
        valid_range: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Human-readable error message.
            parameter: Parameter name that is invalid.
            value: Invalid value provided.
            valid_range: Description of valid values.
            context: Additional context for debugging.
        """
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range
        full_context = {"parameter": parameter, "value": value}
        if valid_range is not None:
            full_context["valid_range"] = valid_range
        if context:
            full_context.update(context)
        super().__init__(message, full_context)
