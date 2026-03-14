# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Parallel pipeline stage definitions.

:class:`StageGraph` encodes which Infinigen generation stages can safely run
in parallel and which have strict ordering dependencies.  This enables
multi-scene batch generation on GPU clusters without data races.

All helpers are pure Python — no ``bpy`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    """A single pipeline stage with explicit dependencies.

    Parameters
    ----------
    name : str
        Human-readable stage name (e.g. ``"coarse"``, ``"render"``).
    depends_on : frozenset[str]
        Names of stages that *must* complete before this one starts.
    gpu : bool
        Whether this stage requires a GPU.
    """

    name: str
    depends_on: frozenset[str] = frozenset()
    gpu: bool = False


# ---------------------------------------------------------------------------
# Default stage definitions matching Infinigen's task enum
# ---------------------------------------------------------------------------

_DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage(name="coarse", depends_on=frozenset(), gpu=False),
    Stage(name="populate", depends_on=frozenset({"coarse"}), gpu=False),
    Stage(name="fine_terrain", depends_on=frozenset({"coarse"}), gpu=False),
    Stage(name="render", depends_on=frozenset({"populate", "fine_terrain"}), gpu=True),
    Stage(name="ground_truth", depends_on=frozenset({"render"}), gpu=True),
    Stage(name="mesh_save", depends_on=frozenset({"populate", "fine_terrain"}), gpu=False),
    Stage(name="export", depends_on=frozenset({"populate", "fine_terrain"}), gpu=False),
)


@dataclass
class StageGraph:
    """Directed acyclic graph of pipeline stages.

    Parameters
    ----------
    stages : tuple[Stage, ...]
        The ordered list of stages.  Defaults to the standard Infinigen
        pipeline.

    Examples
    --------
    >>> g = StageGraph()
    >>> g.parallel_groups()
    [['coarse'], ['populate', 'fine_terrain'], ['render', 'mesh_save', 'export'], ['ground_truth']]
    """

    stages: tuple[Stage, ...] = field(default_factory=lambda: _DEFAULT_STAGES)

    def _by_name(self) -> dict[str, Stage]:
        return {s.name: s for s in self.stages}

    def parallel_groups(self) -> list[list[str]]:
        """Return stages grouped into waves that can execute concurrently.

        Each wave contains stages whose dependencies are all satisfied by
        earlier waves.  Within a wave, stages are independent and can run
        in parallel.
        """
        by_name = self._by_name()
        remaining = set(by_name)
        completed: set[str] = set()
        waves: list[list[str]] = []

        while remaining:
            wave = sorted(
                name
                for name in remaining
                if by_name[name].depends_on <= completed
            )
            if not wave:
                msg = "Cycle detected in stage dependencies"
                raise RuntimeError(msg)
            waves.append(wave)
            completed.update(wave)
            remaining -= set(wave)

        return waves

    def gpu_stages(self) -> list[str]:
        """Return names of stages that require a GPU."""
        return [s.name for s in self.stages if s.gpu]

    def cpu_only_stages(self) -> list[str]:
        """Return names of stages that can run on CPU only."""
        return [s.name for s in self.stages if not s.gpu]

    def topological_order(self) -> list[str]:
        """Return a flat topological ordering (deterministic)."""
        order: list[str] = []
        for wave in self.parallel_groups():
            order.extend(wave)
        return order
