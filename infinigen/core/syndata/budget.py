# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Scene complexity budget enforcement.

Prevents scenes from exceeding resource limits (polygon count, memory, object
count) during generation.  A :class:`SceneBudget` is initialised with hard
limits; code that adds geometry to the scene should call :meth:`try_allocate`
before committing new objects.

Typical usage::

    budget = SceneBudget(max_polygons=2_000_000, max_objects=200)
    if budget.try_allocate(polygons=50_000, objects=1):
        ...  # add asset
    else:
        ...  # skip, budget exhausted
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SceneBudget:
    """Hard resource limits for scene generation.

    Parameters
    ----------
    max_polygons : int
        Maximum total polygon count.
    max_vertices : int
        Maximum total vertex count.
    max_objects : int
        Maximum distinct object count.
    max_memory_mb : float
        Maximum estimated memory usage in MB.
    """

    max_polygons: int = 5_000_000
    max_vertices: int = 2_500_000
    max_objects: int = 500
    max_memory_mb: float = 4096.0

    # Running totals
    _used_polygons: int = field(init=False, default=0)
    _used_vertices: int = field(init=False, default=0)
    _used_objects: int = field(init=False, default=0)
    _used_memory_mb: float = field(init=False, default=0.0)
    _rejection_count: int = field(init=False, default=0)

    @property
    def used_polygons(self) -> int:
        return self._used_polygons

    @property
    def used_vertices(self) -> int:
        return self._used_vertices

    @property
    def used_objects(self) -> int:
        return self._used_objects

    @property
    def used_memory_mb(self) -> float:
        return self._used_memory_mb

    @property
    def rejection_count(self) -> int:
        return self._rejection_count

    def remaining_polygons(self) -> int:
        remaining = self.max_polygons - self._used_polygons
        if remaining < 0:
            logger.warning(
                "Budget overflow: used %d > max %d polygons",
                self._used_polygons, self.max_polygons,
            )
        return max(0, remaining)

    def remaining_objects(self) -> int:
        remaining = self.max_objects - self._used_objects
        if remaining < 0:
            logger.warning(
                "Budget overflow: used %d > max %d objects",
                self._used_objects, self.max_objects,
            )
        return max(0, remaining)

    def utilisation(self) -> dict[str, float]:
        """Return utilisation fractions for each resource in [0, 1]."""
        return {
            "polygons": self._used_polygons / max(self.max_polygons, 1),
            "vertices": self._used_vertices / max(self.max_vertices, 1),
            "objects": self._used_objects / max(self.max_objects, 1),
            "memory_mb": self._used_memory_mb / max(self.max_memory_mb, 1.0),
        }

    def can_allocate(
        self,
        *,
        polygons: int = 0,
        vertices: int = 0,
        objects: int = 0,
        memory_mb: float = 0.0,
    ) -> bool:
        """Check whether the allocation would fit without modifying state."""
        return (
            self._used_polygons + polygons <= self.max_polygons
            and self._used_vertices + vertices <= self.max_vertices
            and self._used_objects + objects <= self.max_objects
            and self._used_memory_mb + memory_mb <= self.max_memory_mb
        )

    def try_allocate(
        self,
        *,
        polygons: int = 0,
        vertices: int = 0,
        objects: int = 0,
        memory_mb: float = 0.0,
    ) -> bool:
        """Try to allocate resources.  Returns ``True`` on success.

        If the allocation would exceed any limit, the state is unchanged
        and ``False`` is returned.
        """
        if not self.can_allocate(
            polygons=polygons,
            vertices=vertices,
            objects=objects,
            memory_mb=memory_mb,
        ):
            self._rejection_count += 1
            logger.debug(
                "Budget rejected: +%d polys, +%d verts, +%d objs, +%.1f MB "
                "(used: %d/%d polys, %d/%d objs)",
                polygons,
                vertices,
                objects,
                memory_mb,
                self._used_polygons,
                self.max_polygons,
                self._used_objects,
                self.max_objects,
            )
            return False

        self._used_polygons += polygons
        self._used_vertices += vertices
        self._used_objects += objects
        self._used_memory_mb += memory_mb
        return True

    def force_allocate(
        self,
        *,
        polygons: int = 0,
        vertices: int = 0,
        objects: int = 0,
        memory_mb: float = 0.0,
    ) -> None:
        """Allocate resources unconditionally (may exceed limits)."""
        self._used_polygons += polygons
        self._used_vertices += vertices
        self._used_objects += objects
        self._used_memory_mb += memory_mb

    def release(
        self,
        *,
        polygons: int = 0,
        vertices: int = 0,
        objects: int = 0,
        memory_mb: float = 0.0,
    ) -> None:
        """Return previously allocated resources.

        Raises
        ------
        ValueError
            If any amount is negative.
        """
        if any(x < 0 for x in (polygons, vertices, objects, memory_mb)):
            raise ValueError("Cannot release negative amounts")
        for name, releasing, used in [
            ("polygons", polygons, self._used_polygons),
            ("vertices", vertices, self._used_vertices),
            ("objects", objects, self._used_objects),
            ("memory_mb", memory_mb, self._used_memory_mb),
        ]:
            if releasing > used:
                logger.warning(
                    "Releasing %s %s but only %s used", releasing, name, used,
                )
        self._used_polygons = max(0, self._used_polygons - polygons)
        self._used_vertices = max(0, self._used_vertices - vertices)
        self._used_objects = max(0, self._used_objects - objects)
        self._used_memory_mb = max(0.0, self._used_memory_mb - memory_mb)

    def reset(self) -> None:
        """Reset all counters to zero."""
        self._used_polygons = 0
        self._used_vertices = 0
        self._used_objects = 0
        self._used_memory_mb = 0.0
        self._rejection_count = 0
