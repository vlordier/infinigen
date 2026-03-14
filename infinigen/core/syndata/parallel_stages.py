# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Parallel execution support for independent pipeline stages.

Provides :class:`ParallelStageExecutor` that wraps multiple
:func:`concurrent.futures.ThreadPoolExecutor` workers to run independent
stages concurrently while respecting dependency ordering.

.. note::
   Blender (bpy) is **not** thread-safe for scene mutation, so this executor
   is intended for *non-Blender* computation (e.g. parameter sampling, file
   I/O, mesh pre-processing).  Blender-touching stages should still run
   sequentially via :class:`RandomStageExecutor`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageSpec:
    """Specification for a single parallel stage.

    Parameters
    ----------
    name : str
        Human-readable stage name.
    fn : Callable
        The function to execute.
    args : tuple
        Positional arguments for *fn*.
    kwargs : dict
        Keyword arguments for *fn*.
    depends_on : list[str]
        Names of stages that must complete before this one starts.
    """

    name: str
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class StageOutcome:
    """Result of a completed stage."""

    name: str
    result: Any = None
    error: Exception | None = None
    elapsed_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


class ParallelStageExecutor:
    """Execute independent stages concurrently.

    Stages are submitted to a thread pool.  Dependency edges are respected:
    a stage will not start until all stages it depends on have completed.

    Parameters
    ----------
    max_workers : int
        Maximum number of threads. ``1`` means serial execution.
    """

    def __init__(self, max_workers: int = 2):
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._max_workers = max_workers
        self._outcomes: list[StageOutcome] = []

    @property
    def outcomes(self) -> list[StageOutcome]:
        """All completed stage outcomes."""
        return list(self._outcomes)

    def run(
        self, stages: list[StageSpec], *, timeout_s: float | None = None
    ) -> list[StageOutcome]:
        """Execute *stages* respecting dependencies, return outcomes.

        Parameters
        ----------
        stages : list[StageSpec]
            Stages to execute.  The order within the list does **not** matter;
            dependency edges determine actual execution order.
        timeout_s : float or None
            Maximum seconds to wait for all futures.  ``None`` means no limit.

        Returns
        -------
        list[StageOutcome]
            One outcome per stage, in completion order.
        """
        if not stages:
            return []

        # Validate dependencies
        stage_names = {s.name for s in stages}
        for s in stages:
            for dep in s.depends_on:
                if dep not in stage_names:
                    raise ValueError(
                        f"Stage '{s.name}' depends on unknown stage '{dep}'"
                    )

        completed: dict[str, StageOutcome] = {}
        pending = {s.name: s for s in stages}
        futures: dict[Future, str] = {}
        outcomes: list[StageOutcome] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending or futures:
                # Submit stages whose dependencies are met
                ready = [
                    name
                    for name, spec in pending.items()
                    if all(d in completed for d in spec.depends_on)
                ]
                for name in ready:
                    spec = pending.pop(name)
                    future = pool.submit(self._run_one, spec)
                    futures[future] = name

                if not futures:
                    # Deadlock detection
                    if pending:
                        unsatisfied = [
                            f"  - '{n}' depends on {s.depends_on}"
                            for n, s in pending.items()
                        ]
                        raise RuntimeError(
                            "Dependency deadlock. Unsatisfiable stages:\n"
                            + "\n".join(unsatisfied)
                        )
                    break

                # Wait for all currently-submitted futures to complete
                done_futures: list[Future] = []
                for future in as_completed(futures, timeout=timeout_s):
                    name = futures[future]
                    outcome = future.result()
                    completed[name] = outcome
                    outcomes.append(outcome)
                    done_futures.append(future)

                for f in done_futures:
                    del futures[f]

        self._outcomes = outcomes
        return outcomes

    @staticmethod
    def _run_one(spec: StageSpec) -> StageOutcome:
        t0 = time.monotonic()
        try:
            result = spec.fn(*spec.args, **spec.kwargs)
            return StageOutcome(
                name=spec.name, result=result, elapsed_s=time.monotonic() - t0
            )
        except Exception as exc:
            logger.warning("Stage '%s' failed: %s", spec.name, exc)
            return StageOutcome(
                name=spec.name, error=exc, elapsed_s=time.monotonic() - t0
            )
