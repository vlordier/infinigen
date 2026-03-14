# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Thread-pooled batch processing utilities for CPU-bound NumPy workloads.

Provides ``parallel_map`` for embarrassingly-parallel array operations and
``chunked_concat`` for large concatenation workloads that benefit from
parallel chunk processing.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


def _default_num_workers() -> int:
    """Return the number of worker threads for parallel operations.

    Respects ``INFINIGEN_NUM_THREADS`` env var.  Defaults to
    ``min(os.cpu_count(), 8)`` to avoid over-subscription on large machines.
    """
    env = os.environ.get("INFINIGEN_NUM_THREADS")
    if env:
        return max(1, int(env))
    return min(os.cpu_count() or 4, 8)


def parallel_map(fn, items, *, num_workers: int | None = None):
    """Apply *fn* to each element of *items* using a thread pool.

    Falls back to a sequential ``map`` when *num_workers* is 1 or when
    there are fewer than 2 items.

    Parameters
    ----------
    fn : callable
        Pure function ``fn(item) -> result``.
    items : sequence
        Inputs to map over.
    num_workers : int | None
        Worker threads.  ``None`` uses :func:`_default_num_workers`.

    Returns
    -------
    list
        Results in input order.
    """
    if num_workers is None:
        num_workers = _default_num_workers()

    if num_workers <= 1 or len(items) < 2:
        return [fn(item) for item in items]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(fn, items))


def chunked_concat(arrays, *, axis: int = 0, num_workers: int | None = None):
    """Concatenate a large list of arrays using parallel chunked merging.

    Splits *arrays* into *num_workers* chunks, concatenates each chunk in a
    separate thread, then does a final concatenation of the partial results.
    This is faster than a single ``np.concatenate`` for very large lists
    because each chunk can run on a separate core.

    For small lists (< 16 arrays) this falls back to ``np.concatenate``.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Arrays to concatenate.
    axis : int
        Concatenation axis.
    num_workers : int | None
        Worker threads.  ``None`` uses :func:`_default_num_workers`.

    Returns
    -------
    np.ndarray
    """
    if not arrays:
        raise ValueError("chunked_concat requires at least one array")

    if num_workers is None:
        num_workers = _default_num_workers()

    if len(arrays) < 16 or num_workers <= 1:
        return np.concatenate(arrays, axis=axis)

    # Split into roughly equal chunks
    chunk_size = max(1, len(arrays) // num_workers)
    chunks = [
        arrays[i : i + chunk_size] for i in range(0, len(arrays), chunk_size)
    ]

    def _concat_chunk(chunk):
        return np.concatenate(chunk, axis=axis)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        partial = list(pool.map(_concat_chunk, chunks))

    return np.concatenate(partial, axis=axis)


def batched_apply(fn, data, batch_size: int, *, axis: int = 0):
    """Apply *fn* to *data* in batches along *axis* and concatenate results.

    Useful for applying GPU-bound operations on data that exceeds VRAM.

    Parameters
    ----------
    fn : callable
        ``fn(chunk) -> np.ndarray`` where chunk is a slice of *data*.
    data : np.ndarray
        Input array to batch over.
    batch_size : int
        Number of elements per batch along *axis*.
    axis : int
        Axis along which to batch.

    Returns
    -------
    np.ndarray
    """
    n = data.shape[axis]
    results = []
    for start in range(0, n, batch_size):
        slc = [slice(None)] * data.ndim
        slc[axis] = slice(start, min(start + batch_size, n))
        results.append(fn(data[tuple(slc)]))
    return np.concatenate(results, axis=axis)
