"""Parameter hashing and normalization for indicators.

This module provides utilities for creating consistent identifiers from indicator
parameters using xxHash. This ensures the same parameters always produce
the same ID, enabling efficient factor storage and retrieval.

Design Principles:
    - SRP: Only handles parameter hashing and formatting
    - KISS: Simple JSON serialization + xxHash (non-cryptographic, fast)
    - DRY: Reusable across all indicator types
"""

import json
from typing import Any

try:
    import xxhash

    HAS_XXHASH = True
except ImportError:
    import hashlib

    HAS_XXHASH = False


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize parameter dictionary for consistent hashing.

    Recursively sorts dictionary keys to ensure consistent ordering regardless
    of how the dict was constructed. List order is preserved.

    Args:
        params: Parameter dictionary to normalize

    Returns:
        Normalized dictionary with sorted keys

    Examples:
        >>> normalize_params({"z": 1, "a": 2})
        {'a': 2, 'z': 1}

        >>> normalize_params({"outer": {"z": 1, "a": 2}})
        {'outer': {'a': 2, 'z': 1}}
    """
    if not isinstance(params, dict):
        return params

    normalized = {}
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, dict):
            normalized[key] = normalize_params(value)
        else:
            normalized[key] = value

    return normalized


def hash_params(params: dict[str, Any]) -> str:
    """Create a stable hash from indicator parameters.

    Uses xxHash64 (non-cryptographic) for very fast, collision-resistant hashing.
    Parameters are normalized (sorted keys) and serialized to JSON before
    hashing to ensure consistency. Falls back to MD5 if xxhash not available.

    Args:
        params: Parameter dictionary

    Returns:
        16-character hex string (8 bytes of hash)

    Examples:
        >>> params = {"period": 14, "signal": 9}
        >>> hash1 = hash_params(params)
        >>> hash2 = hash_params({"signal": 9, "period": 14})
        >>> hash1 == hash2  # Order doesn't matter
        True

        >>> len(hash1)
        16
    """
    normalized = normalize_params(params)
    json_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")

    if HAS_XXHASH:
        hash_val = xxhash.xxh64(json_bytes).intdigest()
        return format(hash_val, "016x")[:16]
    else:
        hash_obj = hashlib.md5(json_bytes, usedforsecurity=False)
        return hash_obj.hexdigest()[:16]


def format_params_key(params: dict[str, Any]) -> str:
    """Format parameters as human-readable key string.

    Creates a readable string representation of parameters for logging
    and debugging. Not used for storage IDs (use hash_params for that).

    Args:
        params: Parameter dictionary

    Returns:
        Formatted parameter string

    Examples:
        >>> format_params_key({"period": 14, "signal": 9})
        'period=14,signal=9'

        >>> format_params_key({})
        'default'
    """
    if not params:
        return "default"

    sorted_params = normalize_params(params)

    parts = []
    for key, value in sorted_params.items():
        if isinstance(value, dict):
            value_str = json.dumps(value, sort_keys=True, separators=(",", ":"))
            parts.append(f"{key}={value_str}")
        elif isinstance(value, list):
            value_str = ";".join(str(v) for v in value)
            parts.append(f"{key}={value_str}")
        else:
            parts.append(f"{key}={value}")

    return ",".join(parts)
