# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Complexity-aware density scaling for asset placement.

Provides helpers that modulate scatter / placement density as a function of
the current curriculum :class:`~infinigen.core.complexity.ComplexityLevel`.

Instead of hard-coding density values for every asset, callers can use
:func:`scaled_density` to automatically adapt an asset's base density to
the active complexity level::

    from infinigen.core.placement import density_scaling

    # base density is the value used at complexity level 5 (FULL)
    effective = density_scaling.scaled_density(base=0.12, level=3)
    # effective ≈ 0.12 * 0.5  (the MODERATE scatter multiplier)
"""

from __future__ import annotations

import logging
import math

from infinigen.core.syndata.complexity import (
    ComplexityLevel,
    get_complexity_params,
)

logger = logging.getLogger(__name__)


def scaled_density(base: float, level: int | ComplexityLevel) -> float:
    """Scale *base* density by the scatter multiplier of *level*.

    Parameters
    ----------
    base : float
        The reference density at full complexity (level 5).
    level : int or ComplexityLevel
        The target complexity level (1-5).

    Returns
    -------
    float
        ``base * params.scatter_density_multiplier`` for the given level.
    """
    params = get_complexity_params(level)
    return base * params.scatter_density_multiplier


def density_ramp(
    base: float,
    level: int | ComplexityLevel,
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float:
    """Compute a clamped density value for *level*.

    Same as :func:`scaled_density` but guarantees the result is within
    ``[floor, ceiling]``.
    """
    raw = scaled_density(base, level)
    return max(floor, min(raw, ceiling))


def species_count_for_level(
    base_max: int,
    level: int | ComplexityLevel,
) -> int:
    """Determine how many species to generate given a curriculum *level*.

    Maps the full-complexity count *base_max* through a fractional ramp
    so that lower levels produce fewer species.

    Parameters
    ----------
    base_max : int
        Maximum species count at full complexity.
    level : int or ComplexityLevel
        Current complexity level.

    Returns
    -------
    int
        Species count in ``[0, base_max]``.
    """
    params = get_complexity_params(level)
    return max(0, min(base_max, math.ceil(base_max * params.scatter_density_multiplier)))


def resolution_for_level(
    base_x: int,
    base_y: int,
    level: int | ComplexityLevel,
) -> tuple[int, int]:
    """Scale render resolution by the level's ``resolution_scale``.

    Ensures dimensions are always even (required by many video codecs).
    """
    params = get_complexity_params(level)
    s = params.resolution_scale
    x = max(2, int(base_x * s) // 2 * 2)
    y = max(2, int(base_y * s) // 2 * 2)
    return (x, y)
