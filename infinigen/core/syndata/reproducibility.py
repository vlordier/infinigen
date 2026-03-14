# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Enhanced per-component reproducibility for scene generation.

Extends the existing ``FixedSeed`` mechanism with a hierarchical seed registry
so that individual scene components (e.g. "terrain.erosion", "vegetation.oak")
can be replayed *exactly* without forcing the rest of the scene to be identical.

Typical usage::

    registry = SeedRegistry(base_seed=42)
    seed_terrain = registry.get("terrain")          # deterministic from base
    seed_oak     = registry.get("vegetation.oak")   # different, still deterministic
    registry.pin("vegetation.oak", 12345)           # override with exact seed
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _component_hash(base_seed: int, component: str, max_val: int = 2**32 - 1) -> int:
    """Derive a deterministic integer seed from *base_seed* and *component*.

    Uses MD5 (same as :func:`infinigen.core.util.math.int_hash`) so that
    the result is consistent across Python versions and platforms.
    """
    m = hashlib.md5()
    m.update(str(base_seed).encode("utf-8"))
    m.update(component.encode("utf-8"))
    return int(m.hexdigest(), 16) % max_val


class SeedRegistry:
    """Deterministic, per-component seed manager.

    Parameters
    ----------
    base_seed : int
        The scene-level base seed from which per-component seeds are derived.
    """

    def __init__(self, base_seed: int):
        if not isinstance(base_seed, int):
            raise TypeError(f"base_seed must be int, got {type(base_seed).__name__}")
        self._base_seed = base_seed
        self._pinned: dict[str, int] = {}
        self._access_log: list[str] = []

    @property
    def base_seed(self) -> int:
        return self._base_seed

    def get(self, component: str) -> int:
        """Return the seed for *component*.

        If a seed has been explicitly pinned via :meth:`pin`, that value is
        returned.  Otherwise a deterministic hash of ``(base_seed, component)``
        is computed.
        """
        if not component:
            raise ValueError("component name must be a non-empty string")
        self._access_log.append(component)
        if component in self._pinned:
            return self._pinned[component]
        return _component_hash(self._base_seed, component)

    def pin(self, component: str, seed: int) -> None:
        """Force *component* to use the given exact *seed*."""
        if not component:
            raise ValueError("component name must be a non-empty string")
        self._pinned[component] = int(seed)
        logger.debug("Pinned component '%s' to seed %d", component, seed)

    def unpin(self, component: str) -> None:
        """Remove a previously pinned seed, reverting to the default hash."""
        self._pinned.pop(component, None)

    @property
    def access_log(self) -> list[str]:
        """Components accessed since creation, in order."""
        return list(self._access_log)

    def clear_log(self) -> None:
        self._access_log.clear()


@contextmanager
def component_seed_context(
    registry: SeedRegistry, component: str
) -> Iterator[int]:
    """Context manager that yields the seed for *component*.

    This is a convenience wrapper around :meth:`SeedRegistry.get` that
    also logs entry/exit for debugging.

    Usage::

        with component_seed_context(registry, "terrain.erosion") as seed:
            np.random.seed(seed)
            ...  # deterministic work
    """
    seed = registry.get(component)
    logger.debug("Entering component '%s' with seed %d", component, seed)
    try:
        yield seed
    finally:
        logger.debug("Exiting component '%s'", component)
