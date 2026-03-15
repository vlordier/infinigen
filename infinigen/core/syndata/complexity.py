# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Curriculum complexity scheduler for progressive RL training.

Maps a *stage* index (0 … ``total_stages - 1``) to concrete scene-generation
parameters: geometry detail, texture resolution, object count, and scatter
density.  The scaling follows an exponential ease-in so that early stages are
cheap to render and later stages approach full photorealism.

All public helpers are pure Python / NumPy — no ``bpy`` dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "CurriculumConfig",
    "curriculum_overrides",
]


@dataclass(frozen=True)
class CurriculumConfig:
    """Immutable snapshot of curriculum parameters for one training stage.

    Parameters
    ----------
    stage : int
        Current difficulty stage (0-indexed).
    total_stages : int
        Total number of stages in the curriculum.
    min_subdiv : int
        Minimum subdivision level for the easiest stage.
    max_subdiv : int
        Maximum subdivision level for the hardest stage.
    min_texture_res : int
        Minimum texture resolution (px) for the easiest stage.
    max_texture_res : int
        Maximum texture resolution (px) for the hardest stage.
    min_objects : int
        Minimum object count for the easiest stage.
    max_objects : int
        Maximum object count for the hardest stage.
    exponent : float
        Controls the shape of the difficulty curve.  1.0 = linear,
        >1 = slow start / fast finish (recommended for RL).
    """

    stage: int = 0
    total_stages: int = 10
    min_subdiv: int = 0
    max_subdiv: int = 4
    min_texture_res: int = 64
    max_texture_res: int = 2048
    min_objects: int = 3
    max_objects: int = 80
    min_scatter_density: float = 0.05
    exponent: float = 2.0

    # Derived values (populated by __post_init__)
    _progress: float = field(init=False, repr=False)
    _subdiv_level: int = field(init=False, repr=False)
    _texture_resolution: int = field(init=False, repr=False)
    _object_count: int = field(init=False, repr=False)
    _scatter_density: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total_stages < 1:
            msg = "total_stages must be >= 1"
            raise ValueError(msg)
        if not 0 <= self.stage < self.total_stages:
            msg = f"stage must be in [0, {self.total_stages})"
            raise ValueError(msg)
        if self.min_subdiv > self.max_subdiv:
            msg = f"min_subdiv ({self.min_subdiv}) must be <= max_subdiv ({self.max_subdiv})"
            raise ValueError(msg)
        if self.min_texture_res > self.max_texture_res:
            msg = f"min_texture_res ({self.min_texture_res}) must be <= max_texture_res ({self.max_texture_res})"
            raise ValueError(msg)
        if self.min_objects > self.max_objects:
            msg = f"min_objects ({self.min_objects}) must be <= max_objects ({self.max_objects})"
            raise ValueError(msg)
        if self.exponent <= 0:
            msg = f"exponent must be positive, got {self.exponent}"
            raise ValueError(msg)
        if not 0.0 < self.min_scatter_density <= 1.0:
            msg = f"min_scatter_density must be in (0, 1], got {self.min_scatter_density}"
            raise ValueError(msg)
        # Exponential ease-in: slow ramp then fast increase
        linear = self.stage / max(self.total_stages - 1, 1)
        p = math.pow(linear, self.exponent)
        object.__setattr__(self, "_progress", p)

        # Pre-compute derived values to avoid repeated calculation
        object.__setattr__(
            self, "_subdiv_level",
            round(self.min_subdiv + p * (self.max_subdiv - self.min_subdiv)),
        )
        raw_tex = self.min_texture_res + p * (self.max_texture_res - self.min_texture_res)
        po2 = int(2 ** round(math.log2(max(raw_tex, 1))))
        object.__setattr__(
            self, "_texture_resolution",
            max(self.min_texture_res, min(po2, self.max_texture_res)),
        )
        object.__setattr__(
            self, "_object_count",
            round(self.min_objects + p * (self.max_objects - self.min_objects)),
        )
        object.__setattr__(
            self, "_scatter_density",
            max(self.min_scatter_density, p),
        )

    # ---- derived properties -------------------------------------------------

    @property
    def progress(self) -> float:
        """Normalised difficulty in [0, 1] after applying the exponent curve."""
        return self._progress

    @property
    def subdiv_level(self) -> int:
        """Subdivision level for this stage (integer)."""
        return self._subdiv_level

    @property
    def texture_resolution(self) -> int:
        """Texture resolution (px) for this stage, rounded to nearest power-of-2.

        The result is clamped to ``[min_texture_res, max_texture_res]`` so it
        never overshoots the configured bounds.
        """
        return self._texture_resolution

    @property
    def object_count(self) -> int:
        """Target number of scene objects for this stage."""
        return self._object_count

    @property
    def scatter_density(self) -> float:
        """Scatter-density multiplier in [min_scatter_density, 1] for this stage."""
        return self._scatter_density


def curriculum_overrides(cfg: CurriculumConfig) -> dict[str, object]:
    """Return a flat dict of scene-generation overrides for the given curriculum stage.

    The keys are *logical* parameter names that downstream tooling (e.g. a
    Gin binding layer or a JSON config driver) can map to actual function
    parameters.  The ``execute_tasks.generate_resolution`` key matches the
    real Gin-configurable parameter in ``infinigen.core.execute_tasks``.
    """
    return {
        "grid_coarsen": max(1, 4 - cfg.subdiv_level),
        "object_count": cfg.object_count,
        "scatter_density_multiplier": round(cfg.scatter_density, 4),
        "texture_resolution": cfg.texture_resolution,
        "execute_tasks.generate_resolution": (
            cfg.texture_resolution,
            cfg.texture_resolution,
        ),
    }
