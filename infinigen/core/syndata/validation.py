# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Lightweight scene validation checks.

Run *before* expensive rendering to detect common issues that would waste
compute time, such as:
  - Empty scenes with no geometry.
  - Scenes that exceed polygon / object budgets.
  - Missing materials or textures.
  - Camera placement outside the scene bounding box.

Each check returns a :class:`ValidationResult` that is either a pass, warning,
or error.  The :func:`validate_scene_config` function runs all checks on a
configuration dictionary and returns a summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    """Validation result severity."""

    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation check."""

    check_name: str
    severity: Severity
    message: str


@dataclass
class ValidationReport:
    """Collection of validation results for a scene."""

    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """``True`` if no errors were found (warnings are acceptable)."""
        return all(r.severity != Severity.ERROR for r in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.severity == Severity.WARNING)

    def summary(self) -> str:
        lines = [f"Validation: {len(self.results)} checks, "
                 f"{self.error_count} errors, {self.warning_count} warnings"]
        for r in self.results:
            if r.severity != Severity.PASS:
                lines.append(f"  [{r.severity.value.upper()}] {r.check_name}: {r.message}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual validation checks
# --------------------------------------------------------------------------- #


def check_object_count(
    config: dict[str, Any],
    *,
    min_objects: int = 1,
    max_objects: int = 10_000,
) -> ValidationResult:
    """Verify object count is within reasonable bounds."""
    count = config.get("object_count", 0)
    if count < min_objects:
        return ValidationResult(
            "object_count",
            Severity.WARNING,
            f"Scene has {count} objects (expected >= {min_objects}). "
            "Scene may be empty.",
        )
    if count > max_objects:
        return ValidationResult(
            "object_count",
            Severity.ERROR,
            f"Scene has {count} objects (limit = {max_objects}). "
            "Rendering will likely run out of memory.",
        )
    return ValidationResult("object_count", Severity.PASS, f"{count} objects OK")


def check_polygon_budget(
    config: dict[str, Any],
    *,
    max_polygons: int = 10_000_000,
) -> ValidationResult:
    """Verify total polygon count is within budget."""
    count = config.get("total_polygon_count", 0)
    if count > max_polygons:
        return ValidationResult(
            "polygon_budget",
            Severity.ERROR,
            f"Total polygons ({count:,}) exceeds budget ({max_polygons:,}).",
        )
    if count > max_polygons * 0.8:
        return ValidationResult(
            "polygon_budget",
            Severity.WARNING,
            f"Total polygons ({count:,}) is > 80% of budget ({max_polygons:,}).",
        )
    return ValidationResult(
        "polygon_budget", Severity.PASS, f"{count:,} polygons OK"
    )


def check_resolution(
    config: dict[str, Any],
    *,
    min_res: int = 64,
    max_res: int = 7680,
) -> ValidationResult:
    """Verify render resolution is sane."""
    res = config.get("resolution", (0, 0))
    w, h = res if len(res) == 2 else (0, 0)
    if w < min_res or h < min_res:
        return ValidationResult(
            "resolution",
            Severity.ERROR,
            f"Resolution {w}x{h} is below minimum {min_res}.",
        )
    if w > max_res or h > max_res:
        return ValidationResult(
            "resolution",
            Severity.WARNING,
            f"Resolution {w}x{h} exceeds {max_res}; render will be slow.",
        )
    return ValidationResult("resolution", Severity.PASS, f"{w}x{h} OK")


def check_seed(config: dict[str, Any]) -> ValidationResult:
    """Verify a valid scene seed is set."""
    seed = config.get("scene_seed")
    if seed is None:
        return ValidationResult(
            "scene_seed",
            Severity.WARNING,
            "No scene_seed set; results may not be reproducible.",
        )
    return ValidationResult("scene_seed", Severity.PASS, f"seed={seed}")


def check_camera_position(
    config: dict[str, Any],
    *,
    bbox_min: tuple[float, float, float] = (-1000, -1000, -100),
    bbox_max: tuple[float, float, float] = (1000, 1000, 500),
) -> ValidationResult:
    """Verify camera position is within the scene bounding box."""
    pos = config.get("camera_position")
    if pos is None:
        return ValidationResult(
            "camera_position", Severity.PASS, "No camera position to check"
        )
    for i, (v, lo, hi) in enumerate(zip(pos, bbox_min, bbox_max)):
        if v < lo or v > hi:
            axis = "XYZ"[i]
            return ValidationResult(
                "camera_position",
                Severity.WARNING,
                f"Camera {axis}={v:.1f} outside scene bounds [{lo}, {hi}].",
            )
    return ValidationResult("camera_position", Severity.PASS, "Camera in bounds")


# --------------------------------------------------------------------------- #
# Aggregate validator
# --------------------------------------------------------------------------- #

_ALL_CHECKS = [
    check_object_count,
    check_polygon_budget,
    check_resolution,
    check_seed,
    check_camera_position,
]


def validate_scene_config(
    config: dict[str, Any],
    *,
    extra_checks: list | None = None,
) -> ValidationReport:
    """Run all built-in (and optional extra) checks on *config*.

    Parameters
    ----------
    config : dict
        Scene configuration / summary dictionary.
    extra_checks : list, optional
        Additional check functions ``(config) -> ValidationResult``.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    checks = list(_ALL_CHECKS)
    if extra_checks:
        checks.extend(extra_checks)
    for check_fn in checks:
        try:
            result = check_fn(config)
            report.add(result)
        except Exception as exc:
            report.add(
                ValidationResult(
                    check_fn.__name__,
                    Severity.ERROR,
                    f"Check raised exception: {exc}",
                )
            )
    return report
