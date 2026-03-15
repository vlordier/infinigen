# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Pre-flight system health checks for Infinigen.

Before launching expensive generation or rendering jobs, it is useful to
verify that the host machine meets basic requirements.  This module
collects those checks in one place so they can be invoked from CLI
scripts, job launchers, or CI.

Usage::

    from infinigen.core.util.health_check import run_health_checks, HealthStatus

    report = run_health_checks()
    for item in report:
        print(f"[{item.status.value}] {item.name}: {item.message}")
    if any(item.status == HealthStatus.FAIL for item in report):
        sys.exit(1)
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Outcome of a single health check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str
    details: dict = field(default_factory=dict)


def check_python_version(
    required_major: int = 3,
    required_minor: int = 11,
) -> HealthCheckResult:
    """Verify the running Python version meets the minimum requirement."""
    current = sys.version_info
    if current.major == required_major and current.minor == required_minor:
        return HealthCheckResult(
            name="python_version",
            status=HealthStatus.PASS,
            message=f"Python {current.major}.{current.minor}.{current.micro}",
            details={"version": f"{current.major}.{current.minor}.{current.micro}"},
        )
    if current.major == required_major and current.minor > required_minor:
        return HealthCheckResult(
            name="python_version",
            status=HealthStatus.WARN,
            message=(
                f"Python {current.major}.{current.minor} detected; "
                f"project targets {required_major}.{required_minor}.x"
            ),
            details={"version": f"{current.major}.{current.minor}.{current.micro}"},
        )
    return HealthCheckResult(
        name="python_version",
        status=HealthStatus.FAIL,
        message=(
            f"Python {current.major}.{current.minor} detected; "
            f"requires {required_major}.{required_minor}.x"
        ),
        details={"version": f"{current.major}.{current.minor}.{current.micro}"},
    )


def check_disk_space(
    path: str | Path = ".",
    min_gb: float = 5.0,
    warn_gb: float = 20.0,
) -> HealthCheckResult:
    """Verify sufficient free disk space at *path*.

    Parameters
    ----------
    path:
        Directory to check (defaults to cwd).
    min_gb:
        Fail threshold in GiB.
    warn_gb:
        Warning threshold in GiB.
    """
    path = Path(path).resolve()
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.FAIL,
            message=f"Cannot stat {path}: {exc}",
        )

    free_gb = usage.free / (1024 ** 3)
    details = {
        "path": str(path),
        "free_gb": round(free_gb, 2),
        "total_gb": round(usage.total / (1024 ** 3), 2),
    }

    if free_gb < min_gb:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.FAIL,
            message=f"Only {free_gb:.1f} GiB free at {path} (need >= {min_gb} GiB)",
            details=details,
        )
    if free_gb < warn_gb:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.WARN,
            message=f"{free_gb:.1f} GiB free at {path} (recommend >= {warn_gb} GiB)",
            details=details,
        )
    return HealthCheckResult(
        name="disk_space",
        status=HealthStatus.PASS,
        message=f"{free_gb:.1f} GiB free at {path}",
        details=details,
    )


def check_required_packages(
    packages: list[str] | None = None,
) -> HealthCheckResult:
    """Verify that required Python packages are importable.

    Parameters
    ----------
    packages:
        Package names to check.  Defaults to core Infinigen dependencies.
    """
    if packages is None:
        packages = [
            "gin",
            "numpy",
            "scipy",
            "cv2",
            "imageio",
            "trimesh",
            "tqdm",
        ]

    missing: list[str] = []
    found: list[str] = []
    for pkg in packages:
        if importlib.util.find_spec(pkg) is not None:
            found.append(pkg)
        else:
            missing.append(pkg)

    if missing:
        return HealthCheckResult(
            name="required_packages",
            status=HealthStatus.FAIL,
            message=f"Missing packages: {', '.join(missing)}",
            details={"missing": missing, "found": found},
        )
    return HealthCheckResult(
        name="required_packages",
        status=HealthStatus.PASS,
        message=f"All {len(found)} required packages importable",
        details={"found": found},
    )


def check_gpu_availability() -> HealthCheckResult:
    """Check whether a GPU (CUDA or MPS) is available via PyTorch."""
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        return HealthCheckResult(
            name="gpu_availability",
            status=HealthStatus.SKIP,
            message="PyTorch not installed; skipping GPU check",
        )

    import torch

    details: dict = {"cuda": False, "mps": False}

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        details["cuda"] = True
        details["gpu_name"] = gpu_name
        return HealthCheckResult(
            name="gpu_availability",
            status=HealthStatus.PASS,
            message=f"CUDA available: {gpu_name}",
            details=details,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        details["mps"] = True
        return HealthCheckResult(
            name="gpu_availability",
            status=HealthStatus.PASS,
            message="MPS (Apple Silicon) available",
            details=details,
        )

    return HealthCheckResult(
        name="gpu_availability",
        status=HealthStatus.WARN,
        message="No GPU detected; rendering will use CPU (slow)",
        details=details,
    )


def check_blender_availability() -> HealthCheckResult:
    """Check whether Blender's ``bpy`` module is importable."""
    spec = importlib.util.find_spec("bpy")
    if spec is None:
        return HealthCheckResult(
            name="blender",
            status=HealthStatus.FAIL,
            message="bpy module not found; install with: pip install bpy==4.5.7",
        )
    return HealthCheckResult(
        name="blender",
        status=HealthStatus.PASS,
        message="bpy module available",
    )


def check_output_directory(
    path: str | Path | None = None,
) -> HealthCheckResult:
    """Verify that the output directory exists and is writable.

    Parameters
    ----------
    path:
        Directory to check.  If ``None``, the check is skipped.
    """
    if path is None:
        return HealthCheckResult(
            name="output_directory",
            status=HealthStatus.SKIP,
            message="No output directory specified",
        )

    path = Path(path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            return HealthCheckResult(
                name="output_directory",
                status=HealthStatus.PASS,
                message=f"Created output directory: {path}",
                details={"path": str(path), "created": True},
            )
        except OSError as exc:
            return HealthCheckResult(
                name="output_directory",
                status=HealthStatus.FAIL,
                message=f"Cannot create {path}: {exc}",
                details={"path": str(path)},
            )

    if not os.access(path, os.W_OK):
        return HealthCheckResult(
            name="output_directory",
            status=HealthStatus.FAIL,
            message=f"Output directory not writable: {path}",
            details={"path": str(path)},
        )

    return HealthCheckResult(
        name="output_directory",
        status=HealthStatus.PASS,
        message=f"Output directory writable: {path}",
        details={"path": str(path)},
    )


def run_health_checks(
    output_dir: str | Path | None = None,
    extra_packages: list[str] | None = None,
) -> list[HealthCheckResult]:
    """Run all health checks and return the results.

    Parameters
    ----------
    output_dir:
        If provided, also check this directory for writability and space.
    extra_packages:
        Additional packages to verify (beyond the defaults).

    Returns
    -------
    list[HealthCheckResult]
        One result per check, in execution order.
    """
    results: list[HealthCheckResult] = []

    results.append(check_python_version())

    disk_path = Path(output_dir) if output_dir else Path(".")
    results.append(check_disk_space(disk_path))

    packages = None
    if extra_packages:
        base = ["gin", "numpy", "scipy", "cv2", "imageio", "trimesh", "tqdm"]
        packages = base + extra_packages
    results.append(check_required_packages(packages))

    results.append(check_gpu_availability())
    results.append(check_blender_availability())
    results.append(check_output_directory(output_dir))

    # Log summary
    counts = {s: 0 for s in HealthStatus}
    for r in results:
        counts[r.status] += 1
    logger.info(
        "Health check summary: %d pass, %d warn, %d fail, %d skip",
        counts[HealthStatus.PASS],
        counts[HealthStatus.WARN],
        counts[HealthStatus.FAIL],
        counts[HealthStatus.SKIP],
    )

    return results


def format_report(results: list[HealthCheckResult]) -> str:
    """Format health check results as a human-readable table.

    Returns
    -------
    str
        Multi-line string with one row per check.
    """
    lines = ["System Health Check Report", "=" * 50]
    status_icons = {
        HealthStatus.PASS: "✓",
        HealthStatus.WARN: "⚠",
        HealthStatus.FAIL: "✗",
        HealthStatus.SKIP: "–",
    }
    for r in results:
        icon = status_icons.get(r.status, "?")
        lines.append(f"  [{icon}] {r.name}: {r.message}")

    # Summary
    fail_count = sum(1 for r in results if r.status == HealthStatus.FAIL)
    warn_count = sum(1 for r in results if r.status == HealthStatus.WARN)
    lines.append("=" * 50)
    if fail_count:
        lines.append(f"RESULT: {fail_count} failure(s), {warn_count} warning(s)")
    elif warn_count:
        lines.append(f"RESULT: All passed with {warn_count} warning(s)")
    else:
        lines.append("RESULT: All checks passed")

    return "\n".join(lines)
