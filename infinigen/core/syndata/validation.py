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

from dataclasses import dataclass, field
from typing import Any

__all__ = ["SceneValidator", "ValidationResult"]


@dataclass
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
    custom_checks: list[tuple[str, Any]] = field(default_factory=list)

    def validate(self, metadata: dict[str, Any]) -> list[ValidationResult]:
        """Run all checks against the supplied metadata dict.

        Expected keys (all optional — missing keys skip their check):

        * ``"obstacles"`` — list of obstacle dicts
        * ``"depth_stats"`` — dict with ``min_m``, ``max_m``
        * ``"traversability_ratio"`` — float in [0, 1]
        * ``"poly_count"`` — int

        Returns
        -------
        list[ValidationResult]
            One result per check.
        """
        results: list[ValidationResult] = []

        # ---- obstacle count -------------------------------------------------
        obstacles = metadata.get("obstacles")
        if obstacles is not None:
            n = len(obstacles)
            results.append(
                ValidationResult(
                    name="obstacle_count",
                    passed=self.min_obstacles <= n <= self.max_obstacles,
                    message=f"{n} obstacles (need {self.min_obstacles}–{self.max_obstacles})",
                )
            )

        # ---- depth range ----------------------------------------------------
        ds = metadata.get("depth_stats")
        if ds is not None:
            has_min = "min_m" in ds
            has_max = "max_m" in ds
            if has_min and has_max:
                depth_range = ds["max_m"] - ds["min_m"]
                results.append(
                    ValidationResult(
                        name="depth_range",
                        passed=depth_range >= self.min_depth_range_m,
                        message=f"depth range {depth_range:.1f} m (need >= {self.min_depth_range_m})",
                    )
                )
            else:
                missing = []
                if not has_min:
                    missing.append("min_m")
                if not has_max:
                    missing.append("max_m")
                results.append(
                    ValidationResult(
                        name="depth_range",
                        passed=False,
                        message=f"depth_stats missing required key(s): {', '.join(missing)}",
                    )
                )

        # ---- traversability -------------------------------------------------
        trav = metadata.get("traversability_ratio")
        if trav is not None:
            results.append(
                ValidationResult(
                    name="traversability",
                    passed=self.min_traversability <= trav <= self.max_traversability,
                    message=(
                        f"traversability {trav:.2f} "
                        f"(need {self.min_traversability}–{self.max_traversability})"
                    ),
                )
            )

        # ---- poly count -----------------------------------------------------
        polys = metadata.get("poly_count")
        if polys is not None:
            results.append(
                ValidationResult(
                    name="poly_count",
                    passed=self.min_poly_count <= polys <= self.max_poly_count,
                    message=f"{polys} tris (need {self.min_poly_count}–{self.max_poly_count})",
                )
            )

        # ---- custom checks --------------------------------------------------
        for name, fn in self.custom_checks:
            passed, msg = fn(metadata)
            results.append(ValidationResult(name=name, passed=passed, message=msg))

        return results

    def is_valid(self, metadata: dict[str, Any]) -> bool:
        """Return *True* if all checks pass."""
        return all(r.passed for r in self.validate(metadata))
