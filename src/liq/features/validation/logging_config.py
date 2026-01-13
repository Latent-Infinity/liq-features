"""Logging configuration for validation module.

Provides structured logging with:
- Module-specific loggers
- Consistent format across all validation functions
- No sensitive data exposure
"""

from __future__ import annotations

import logging
from typing import Any

# Module logger
logger = logging.getLogger("liq.features.validation")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific submodule.

    Args:
        name: Submodule name (e.g., "effect_size", "temporal").

    Returns:
        Logger configured for the submodule.
    """
    return logging.getLogger(f"liq.features.validation.{name}")


def log_function_entry(
    logger: logging.Logger,
    func_name: str,
    **params: Any,
) -> None:
    """Log function entry with parameters.

    Args:
        logger: Logger instance.
        func_name: Name of the function.
        **params: Key parameters to log (sanitized).
    """
    # Filter out large arrays and sensitive data
    safe_params = {}
    for key, value in params.items():
        if hasattr(value, "__len__") and not isinstance(value, (str, dict)):
            safe_params[key] = f"<{type(value).__name__} len={len(value)}>"
        elif isinstance(value, (int, float, str, bool, type(None))):
            safe_params[key] = value
        elif isinstance(value, list) and len(value) <= 10:
            safe_params[key] = value
        else:
            safe_params[key] = f"<{type(value).__name__}>"

    param_str = ", ".join(f"{k}={v}" for k, v in safe_params.items())
    logger.debug(f"Entering {func_name}({param_str})")


def log_function_exit(
    logger: logging.Logger,
    func_name: str,
    result_summary: str | None = None,
) -> None:
    """Log function exit with optional result summary.

    Args:
        logger: Logger instance.
        func_name: Name of the function.
        result_summary: Optional summary of the result.
    """
    if result_summary:
        logger.debug(f"Exiting {func_name}: {result_summary}")
    else:
        logger.debug(f"Exiting {func_name}")


def log_result(
    logger: logging.Logger,
    message: str,
    **metrics: Any,
) -> None:
    """Log a result or key metric at INFO level.

    Args:
        logger: Logger instance.
        message: Description of the result.
        **metrics: Key metrics to include.
    """
    if metrics:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        logger.info(f"{message}: {metric_str}")
    else:
        logger.info(message)


def log_warning(
    logger: logging.Logger,
    message: str,
    **context: Any,
) -> None:
    """Log a warning with context.

    Args:
        logger: Logger instance.
        message: Warning message.
        **context: Additional context.
    """
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.warning(f"{message} ({context_str})")
    else:
        logger.warning(message)


def log_error(
    logger: logging.Logger,
    message: str,
    exc: Exception | None = None,
    **context: Any,
) -> None:
    """Log an error with context and optional exception.

    Args:
        logger: Logger instance.
        message: Error message.
        exc: Optional exception that caused the error.
        **context: Additional context.
    """
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} ({context_str})"
    else:
        full_message = message

    if exc:
        logger.error(full_message, exc_info=exc)
    else:
        logger.error(full_message)
