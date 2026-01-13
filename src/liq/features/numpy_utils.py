"""NumPy conversion helpers for Polars data.

These helpers aim to minimize unnecessary copies while enforcing float64
when required by downstream libraries (TA-Lib, scikit-learn, etc.).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def to_numpy_float64(
    data: pl.DataFrame | pl.Series,
    *,
    allow_copy: bool = False,
    order: str = "fortran",
) -> np.ndarray:
    """Convert Polars data to a float64 numpy array with minimal copying."""
    if isinstance(data, pl.Series):
        try:
            arr = data.to_numpy(writable=False, allow_copy=allow_copy)
        except Exception:
            arr = data.to_numpy(writable=False, allow_copy=True)
    else:
        try:
            arr = data.to_numpy(order=order, writable=False, allow_copy=allow_copy)
        except Exception:
            arr = data.to_numpy(order=order, writable=False, allow_copy=True)

    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    return arr
