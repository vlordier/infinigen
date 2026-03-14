# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Curriculum learning complexity controller for synthetic data generation.

Defines discrete complexity levels (1-5) that map to scene generation parameters.
This enables RL agents to train on progressively complex scenes, following
curriculum learning best practices:
  - Level 1: Minimal / empty terrain with no assets
  - Level 2: Simple scenes with sparse, single-species vegetation
  - Level 3: Moderate scenes with mixed vegetation and basic lighting
  - Level 4: Complex scenes with many species, weather, and effects
  - Level 5: Full photorealistic scenes with all systems enabled
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class ComplexityLevel(IntEnum):
    """Discrete complexity tiers for curriculum learning."""

    MINIMAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    FULL = 5


# Allowed level values for validation
_VALID_LEVELS = frozenset(int(lv) for lv in ComplexityLevel)
MIN_LEVEL = int(ComplexityLevel.MINIMAL)
MAX_LEVEL = int(ComplexityLevel.FULL)


@dataclass(frozen=True)
class ComplexityParams:
    """Immutable snapshot of generation parameters for a given complexity level.

    Parameters
    ----------
    max_tree_species : int
        Upper bound on number of tree species to include.
    max_bush_species : int
        Upper bound on number of bush species to include.
    tree_density : float
        Vegetation density multiplier in [0, 1].
    render_num_samples : int
        Cycles render sample count (higher = slower + better quality).
    enable_terrain_erosion : bool
        Whether to run the costly erosion simulation.
    enable_clouds : bool
        Whether to generate volumetric clouds.
    enable_weather : bool
        Whether to add rain / snow particle effects.
    enable_creatures : bool
        Whether to populate creatures in the scene.
    scatter_density_multiplier : float
        Global multiplier for all scatter densities in [0, 1].
    resolution_scale : float
        Render resolution scale factor in (0, 1].
    """

    max_tree_species: int = 0
    max_bush_species: int = 0
    tree_density: float = 0.0
    render_num_samples: int = 64
    enable_terrain_erosion: bool = False
    enable_clouds: bool = False
    enable_weather: bool = False
    enable_creatures: bool = False
    scatter_density_multiplier: float = 0.0
    resolution_scale: float = 0.25


# ---- built-in presets --------------------------------------------------

_PRESETS: dict[int, ComplexityParams] = {
    ComplexityLevel.MINIMAL: ComplexityParams(
        max_tree_species=0,
        max_bush_species=0,
        tree_density=0.0,
        render_num_samples=64,
        enable_terrain_erosion=False,
        enable_clouds=False,
        enable_weather=False,
        enable_creatures=False,
        scatter_density_multiplier=0.0,
        resolution_scale=0.25,
    ),
    ComplexityLevel.SIMPLE: ComplexityParams(
        max_tree_species=1,
        max_bush_species=0,
        tree_density=0.03,
        render_num_samples=128,
        enable_terrain_erosion=False,
        enable_clouds=False,
        enable_weather=False,
        enable_creatures=False,
        scatter_density_multiplier=0.2,
        resolution_scale=0.5,
    ),
    ComplexityLevel.MODERATE: ComplexityParams(
        max_tree_species=2,
        max_bush_species=1,
        tree_density=0.06,
        render_num_samples=512,
        enable_terrain_erosion=True,
        enable_clouds=True,
        enable_weather=False,
        enable_creatures=False,
        scatter_density_multiplier=0.5,
        resolution_scale=0.75,
    ),
    ComplexityLevel.COMPLEX: ComplexityParams(
        max_tree_species=4,
        max_bush_species=2,
        tree_density=0.10,
        render_num_samples=2048,
        enable_terrain_erosion=True,
        enable_clouds=True,
        enable_weather=True,
        enable_creatures=True,
        scatter_density_multiplier=0.8,
        resolution_scale=1.0,
    ),
    ComplexityLevel.FULL: ComplexityParams(
        max_tree_species=6,
        max_bush_species=3,
        tree_density=0.15,
        render_num_samples=4096,
        enable_terrain_erosion=True,
        enable_clouds=True,
        enable_weather=True,
        enable_creatures=True,
        scatter_density_multiplier=1.0,
        resolution_scale=1.0,
    ),
}


def get_complexity_params(level: int | ComplexityLevel) -> ComplexityParams:
    """Return the ``ComplexityParams`` for the given *level*.

    Parameters
    ----------
    level : int or ComplexityLevel
        The desired complexity tier (1-5).

    Returns
    -------
    ComplexityParams

    Raises
    ------
    ValueError
        If *level* is not in [1, 5].
    """
    level = int(level)
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"Complexity level must be in {MIN_LEVEL}..{MAX_LEVEL}, got {level}"
        )
    return _PRESETS[level]


def get_gin_overrides(level: int | ComplexityLevel) -> list[str]:
    """Produce a list of gin override strings for the given complexity *level*.

    These strings can be passed directly to ``gin.parse_config`` to adapt
    all downstream gin-configurable functions.

    Parameters
    ----------
    level : int or ComplexityLevel
        The desired complexity tier (1-5).

    Returns
    -------
    list[str]
        Gin-format binding strings such as
        ``"compose_nature.max_tree_species = 2"``.
    """
    params = get_complexity_params(level)
    overrides = [
        f"compose_nature.max_tree_species = {params.max_tree_species}",
        f"compose_nature.max_bush_species = {params.max_bush_species}",
        f"compose_nature.tree_density = {params.tree_density}",
        f"configure_render_cycles.num_samples = {params.render_num_samples}",
    ]
    logger.info(
        "Generated %d gin overrides for complexity level %d", len(overrides), level
    )
    return overrides


@dataclass
class ComplexityScheduler:
    """Iterate through complexity levels during curriculum training.

    Parameters
    ----------
    start_level : int
        Initial complexity level (default ``1``).
    max_level : int
        Maximum complexity level to reach (default ``5``).
    episodes_per_level : int
        Number of training episodes before advancing one level.
    """

    start_level: int = MIN_LEVEL
    max_level: int = MAX_LEVEL
    episodes_per_level: int = 100
    _current_level: int = field(init=False, default=MIN_LEVEL)
    _episodes_completed: int = field(init=False, default=0)

    def __post_init__(self):
        if not (MIN_LEVEL <= self.start_level <= MAX_LEVEL):
            raise ValueError(
                f"start_level must be in {MIN_LEVEL}..{MAX_LEVEL}, got {self.start_level}"
            )
        if not (MIN_LEVEL <= self.max_level <= MAX_LEVEL):
            raise ValueError(
                f"max_level must be in {MIN_LEVEL}..{MAX_LEVEL}, got {self.max_level}"
            )
        if self.start_level > self.max_level:
            raise ValueError(
                f"start_level ({self.start_level}) must be <= max_level ({self.max_level})"
            )
        if self.episodes_per_level < 1:
            raise ValueError(
                f"episodes_per_level must be >= 1, got {self.episodes_per_level}"
            )
        self._current_level = self.start_level

    @property
    def current_level(self) -> int:
        return self._current_level

    @property
    def current_params(self) -> ComplexityParams:
        return get_complexity_params(self._current_level)

    def step(self) -> int:
        """Record one completed episode and maybe advance the level.

        Returns
        -------
        int
            The *new* current complexity level (may be unchanged).
        """
        self._episodes_completed += 1
        if (
            self._episodes_completed % self.episodes_per_level == 0
            and self._current_level < self.max_level
        ):
            self._current_level += 1
            logger.info(
                "Curriculum scheduler advanced to level %d after %d episodes",
                self._current_level,
                self._episodes_completed,
            )
        return self._current_level

    def reset(self) -> None:
        """Reset the scheduler to its initial state."""
        self._current_level = self.start_level
        self._episodes_completed = 0


def interpolate_params(
    level_low: int,
    level_high: int,
    t: float,
) -> dict[str, Any]:
    """Linearly interpolate between two complexity levels.

    This enables smooth transitions rather than discrete jumps, which can
    benefit RL agents that are sensitive to distribution shifts.

    Parameters
    ----------
    level_low : int
        Lower complexity level (1-5).
    level_high : int
        Upper complexity level (1-5), must be > *level_low*.
    t : float
        Interpolation factor in [0, 1].  ``t=0`` returns *level_low* params,
        ``t=1`` returns *level_high* params.

    Returns
    -------
    dict[str, Any]
        Interpolated parameter dictionary.

    Raises
    ------
    ValueError
        If *t* is outside [0, 1] or levels are invalid.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Interpolation factor t must be in [0, 1], got {t}")
    if level_low >= level_high:
        raise ValueError(
            f"level_low ({level_low}) must be < level_high ({level_high})"
        )
    low = get_complexity_params(level_low)
    high = get_complexity_params(level_high)

    def _lerp_int(a: int, b: int) -> int:
        return int(round(a + (b - a) * t))

    def _lerp_float(a: float, b: float) -> float:
        return a + (b - a) * t

    def _lerp_bool(a: bool, b: bool) -> bool:
        # Switch at midpoint
        return b if t >= 0.5 else a

    return {
        "max_tree_species": _lerp_int(low.max_tree_species, high.max_tree_species),
        "max_bush_species": _lerp_int(low.max_bush_species, high.max_bush_species),
        "tree_density": _lerp_float(low.tree_density, high.tree_density),
        "render_num_samples": _lerp_int(
            low.render_num_samples, high.render_num_samples
        ),
        "enable_terrain_erosion": _lerp_bool(
            low.enable_terrain_erosion, high.enable_terrain_erosion
        ),
        "enable_clouds": _lerp_bool(low.enable_clouds, high.enable_clouds),
        "enable_weather": _lerp_bool(low.enable_weather, high.enable_weather),
        "enable_creatures": _lerp_bool(low.enable_creatures, high.enable_creatures),
        "scatter_density_multiplier": _lerp_float(
            low.scatter_density_multiplier, high.scatter_density_multiplier
        ),
        "resolution_scale": _lerp_float(low.resolution_scale, high.resolution_scale),
    }
