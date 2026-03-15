# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Density scaling for object population by curriculum level.

:class:`DensityScaler` provides multipliers for scatter density, instance
count, and obstacle population.  The multipliers grow with a configurable
curve from a sparse "easy" scene to a dense "hard" scene.

All helpers are pure Python — no ``bpy`` dependency.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import ClassVar

__all__ = ["DensityScaler", "InterpolationCurve"]


class InterpolationCurve(enum.Enum):
    """Interpolation curve types for density scaling.

    Using an enum prevents typos and gives IDE autocompletion.
    """

    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SQRT = "sqrt"


@dataclass(frozen=True)
class DensityScaler:
    """Scale scatter / instance density by curriculum difficulty.

    Parameters
    ----------
    difficulty : float
        Normalised difficulty in [0, 1].
    min_multiplier : float
        Density multiplier at difficulty = 0.
    max_multiplier : float
        Density multiplier at difficulty = 1.
    obstacle_min : int
        Minimum obstacle count at difficulty = 0.
    obstacle_max : int
        Maximum obstacle count at difficulty = 1.
    curve : str
        Interpolation curve: ``"linear"``, ``"quadratic"``, or ``"sqrt"``.
        Can also be an :class:`InterpolationCurve` enum member.
    """

    difficulty: float = 0.5
    min_multiplier: float = 0.1
    max_multiplier: float = 1.0
    obstacle_min: int = 2
    obstacle_max: int = 50
    curve: str = "linear"

    _CURVES: ClassVar[frozenset[str]] = frozenset(c.value for c in InterpolationCurve)

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            msg = "difficulty must be in [0.0, 1.0]"
            raise ValueError(msg)
        # Accept both string and InterpolationCurve enum values
        curve_val = self.curve.value if isinstance(self.curve, InterpolationCurve) else self.curve
        if curve_val not in self._CURVES:
            msg = f"curve must be one of {sorted(self._CURVES)}"
            raise ValueError(msg)
        if curve_val != self.curve:
            object.__setattr__(self, "curve", curve_val)
        if self.min_multiplier < 0:
            msg = f"min_multiplier must be non-negative, got {self.min_multiplier}"
            raise ValueError(msg)
        if self.min_multiplier > self.max_multiplier:
            msg = f"min_multiplier ({self.min_multiplier}) must be <= max_multiplier ({self.max_multiplier})"
            raise ValueError(msg)
        if self.obstacle_min > self.obstacle_max:
            msg = f"obstacle_min ({self.obstacle_min}) must be <= obstacle_max ({self.obstacle_max})"
            raise ValueError(msg)

    def _t(self) -> float:
        """Apply the interpolation curve to the difficulty value.

        Returns a transformed difficulty in [0, 1] shaped by the
        selected curve: linear (identity), quadratic (slow start),
        or sqrt (fast start).
        """
        d = self.difficulty
        if self.curve == "quadratic":
            return d * d
        if self.curve == "sqrt":
            return math.sqrt(d)
        return d

    @property
    def scatter_multiplier(self) -> float:
        """Multiplier ∈ [min_multiplier, max_multiplier] for scatter density."""
        t = self._t()
        return self.min_multiplier + t * (self.max_multiplier - self.min_multiplier)

    @property
    def instance_multiplier(self) -> float:
        """Multiplier for instanced-object counts (same curve as scatter)."""
        return self.scatter_multiplier

    @property
    def obstacle_count(self) -> int:
        """Integer obstacle count for the current difficulty."""
        t = self._t()
        return round(self.obstacle_min + t * (self.obstacle_max - self.obstacle_min))

    def gin_overrides(self) -> dict[str, object]:
        """Return logical overrides for density parameters.

        Keys are *logical* parameter names; downstream tooling maps them to
        actual Gin bindings or scene-builder arguments.
        """
        return {
            "scatter_density_multiplier": round(self.scatter_multiplier, 4),
            "instance_density_multiplier": round(self.instance_multiplier, 4),
            "obstacle_count": self.obstacle_count,
        }
