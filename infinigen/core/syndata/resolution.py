# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Resolution ladder for curriculum RL training.

Provides :func:`resolution_for_stage` which maps a curriculum stage to a
render resolution.  Resolutions are always powers-of-two (GPU-friendly) and
clamped to a configurable range.

All helpers are pure Python — no ``bpy`` dependency.
"""

from __future__ import annotations

import math

__all__ = ["resolution_for_stage"]

# Default ladder: stage 0 → 64 px, stage N-1 → 2048 px
_DEFAULT_MIN = 64
_DEFAULT_MAX = 2048


def resolution_for_stage(
    stage: int,
    total_stages: int,
    *,
    min_res: int = _DEFAULT_MIN,
    max_res: int = _DEFAULT_MAX,
    aspect_ratio: float = 1.0,
) -> tuple[int, int]:
    """Compute render resolution for a curriculum stage.

    Parameters
    ----------
    stage : int
        Current stage index (0-based).
    total_stages : int
        Total number of curriculum stages.
    min_res : int
        Shortest side at stage 0.
    max_res : int
        Shortest side at the final stage.
    aspect_ratio : float
        Width / height.  Use ``4/3`` for standard drone FPV or ``16/9``
        for wide-angle.  Values < 1 are portrait orientation.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` both rounded to the nearest power-of-two.

    Raises
    ------
    ValueError
        If inputs are out of range.
    """
    if total_stages < 1:
        msg = "total_stages must be >= 1"
        raise ValueError(msg)
    if not 0 <= stage < total_stages:
        msg = f"stage must be in [0, {total_stages})"
        raise ValueError(msg)
    if min_res < 1 or max_res < 1:
        msg = "min_res and max_res must be positive"
        raise ValueError(msg)
    if min_res > max_res:
        msg = f"min_res ({min_res}) must be <= max_res ({max_res})"
        raise ValueError(msg)
    if aspect_ratio <= 0:
        msg = "aspect_ratio must be positive"
        raise ValueError(msg)
    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        msg = f"aspect_ratio={aspect_ratio} out of range; use 0.25–4.0"
        raise ValueError(msg)

    t = stage / max(total_stages - 1, 1)
    log_min = math.log2(min_res)
    log_max = math.log2(max_res)
    short_side = int(2 ** round(log_min + t * (log_max - log_min)))
    # Clamp to configured bounds (round-to-po2 can overshoot)
    short_side = max(min_res, min(short_side, max_res))

    if aspect_ratio >= 1.0:
        width = int(2 ** round(math.log2(short_side * aspect_ratio)))
        height = short_side
    else:
        width = short_side
        height = int(2 ** round(math.log2(short_side / aspect_ratio)))

    return (width, height)
