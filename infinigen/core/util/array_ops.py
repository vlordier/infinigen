# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Fast helpers for common NumPy array operations.

The main entry-point is :func:`unique_rows`, which replaces
``np.unique(arr, axis=0, ...)`` with a void-view trick that
avoids the slow lexsort path and runs a fast 1-D unique instead.
"""

import numpy as np


def unique_rows(arr, return_inverse=False, return_index=False, return_counts=False):
    """Return unique rows of a 2-D array using a void-view reinterpretation.

    ``np.unique(..., axis=0)`` falls back to a lexsort-based path that
    is O(N·log N·C) where C is the number of columns.  By viewing each
    row as an opaque ``np.void`` blob we reduce the problem to a 1-D
    unique, which NumPy handles via a single O(N·log N) sort.

    Parameters
    ----------
    arr : np.ndarray
        2-D array whose rows should be de-duplicated.
    return_inverse, return_index, return_counts : bool
        Forwarded to ``np.unique`` (same semantics).

    Returns
    -------
    unique_rows : np.ndarray
        The unique rows, same dtype and number of columns as *arr*.
    *extras
        Optional inverse / index / counts arrays (same order as ``np.unique``).
    """
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"unique_rows expects a 2-D array, got ndim={arr.ndim}")

    void_dt = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    void_view = arr.view(void_dt).ravel()

    result = np.unique(
        void_view,
        return_inverse=return_inverse,
        return_index=return_index,
        return_counts=return_counts,
    )

    if isinstance(result, tuple):
        unique_void = result[0]
        extras = result[1:]
    else:
        unique_void = result
        extras = ()

    unique = unique_void.view(arr.dtype).reshape(-1, arr.shape[1])

    if extras:
        return (unique, *extras)
    return unique
