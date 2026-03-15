# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Validation utilities for generated scenes.

:class:`SceneValidator` checks that a generated scene meets minimum quality
and complexity constraints before it is fed to an RL training loop.  This
avoids wasting GPU cycles on degenerate or trivially easy scenes.

All helpers are pure Python / NumPy — no ``bpy`` dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["SceneValidator", "ValidationResult"]

# Type alias for custom validation check callables.
# Each callable receives the metadata dict and returns (passed, message).
CheckFn = Callable[[dict[str, object]], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of a single validation check.

    Parameters
    ----------
    name : str
        Short identifier for the check.
    passed : bool
        Whether the scene passed this check.
    message : str
        Human-readable explanation.
    """

    name: str
    passed: bool
    message: str = ""


@dataclass
class SceneValidator:
    """Configurable validator for generated scenes.

    Parameters
    ----------
    min_obstacles : int
        Minimum number of obstacle bounding boxes.
    max_obstacles : int
        Maximum number of obstacle bounding boxes.
    min_depth_range_m : float
        Minimum depth range (max−min) in metres.
    min_traversability : float
        Minimum traversability ratio (0–1).
    max_traversability : float
        Maximum traversability ratio — scenes that are *too* open
        are trivially easy for RL agents.
    min_poly_count : int
        Minimum total triangle count.
    max_poly_count : int
        Maximum total triangle count (to prevent OOM).
    custom_checks : list
        List of ``(name, callable)`` pairs.  Each callable receives
        the metadata dict and returns ``(passed: bool, msg: str)``.
    """

    min_obstacles: int = 1
    max_obstacles: int = 200
    min_depth_range_m: float = 1.0
    min_traversability: float = 0.05
    max_traversability: float = 0.95
    min_poly_count: int = 1000
    max_poly_count: int = 10_000_000
    custom_checks: list[tuple[str, CheckFn]] = field(default_factory=list)

    def validate(self, metadata: dict[str, object], *, fail_fast: bool = False) -> list[ValidationResult]:
        """Run all checks against the supplied metadata dict.

        Expected keys (all optional — missing keys skip their check):

        * ``"obstacles"`` — list of obstacle dicts
        * ``"depth_stats"`` — dict with ``min_m``, ``max_m``
        * ``"traversability_ratio"`` — float in [0, 1]
        * ``"poly_count"`` — int

        Parameters
        ----------
        metadata : dict[str, object]
            Scene metadata to validate.
        fail_fast : bool
            If *True*, stop after the first failed check.  Useful in
            hot loops where only pass/fail matters and detailed
            diagnostics are not needed.

        Returns
        -------
        list[ValidationResult]
            One result per check (or fewer if *fail_fast* triggered).
        """
        results: list[ValidationResult] = []

        def _append(result: ValidationResult) -> bool:
            """Append a result; return True if fail_fast should stop."""
            results.append(result)
            if fail_fast and not result.passed:
                logger.debug("validate: fail_fast triggered on check %r", results[-1].name)
                return True
            return False

        # ---- obstacle count -------------------------------------------------
        obstacles = metadata.get("obstacles")
        if obstacles is not None:
            n = len(obstacles)
            if _append(
                ValidationResult(
                    name="obstacle_count",
                    passed=self.min_obstacles <= n <= self.max_obstacles,
                    message=f"{n} obstacles (need {self.min_obstacles}–{self.max_obstacles})",
                )
            ):
                return results

        # ---- depth range ----------------------------------------------------
        ds = metadata.get("depth_stats")
        if ds is not None:
            has_min = "min_m" in ds
            has_max = "max_m" in ds
            if has_min and has_max:
                depth_range = ds["max_m"] - ds["min_m"]
                if _append(
                    ValidationResult(
                        name="depth_range",
                        passed=depth_range >= self.min_depth_range_m,
                        message=f"depth range {depth_range:.1f} m (need >= {self.min_depth_range_m})",
                    )
                ):
                    return results
            else:
                missing = []
                if not has_min:
                    missing.append("min_m")
                if not has_max:
                    missing.append("max_m")
                if _append(
                    ValidationResult(
                        name="depth_range",
                        passed=False,
                        message=f"depth_stats missing required key(s): {', '.join(missing)}",
                    )
                ):
                    return results

        # ---- traversability -------------------------------------------------
        trav = metadata.get("traversability_ratio")
        if trav is not None:
            if _append(
                ValidationResult(
                    name="traversability",
                    passed=self.min_traversability <= trav <= self.max_traversability,
                    message=(
                        f"traversability {trav:.2f} "
                        f"(need {self.min_traversability}–{self.max_traversability})"
                    ),
                )
            ):
                return results

        # ---- poly count -----------------------------------------------------
        polys = metadata.get("poly_count")
        if polys is not None:
            if _append(
                ValidationResult(
                    name="poly_count",
                    passed=self.min_poly_count <= polys <= self.max_poly_count,
                    message=f"{polys} tris (need {self.min_poly_count}–{self.max_poly_count})",
                )
            ):
                return results

        # ---- custom checks --------------------------------------------------
        for name, fn in self.custom_checks:
            passed, msg = fn(metadata)
            if _append(ValidationResult(name=name, passed=passed, message=msg)):
                return results

        return results

    def is_valid(self, metadata: dict[str, object]) -> bool:
        """Return *True* if all checks pass."""
        return all(r.passed for r in self.validate(metadata))
