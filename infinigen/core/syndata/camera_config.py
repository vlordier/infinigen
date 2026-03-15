# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Drone camera configuration for the Infinigen rendering pipeline.

Configures camera optics (FoV, focal length, aspect ratio) and rig
layouts (stereo baseline, multi-drone positions) matching Infinigen's
``spawn_camera_rigs`` gin structure.

All helpers are pure Python — no ``bpy`` dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ASPECT_1_1",
    "ASPECT_16_9",
    "ASPECT_4_3",
    "CameraRigConfig",
    "DroneCamera",
]

# Standard drone FPV aspect ratios
ASPECT_4_3 = 4 / 3
ASPECT_16_9 = 16 / 9
ASPECT_1_1 = 1.0


@dataclass(frozen=True)
class DroneCamera:
    """Single camera specification for a drone agent.

    Parameters
    ----------
    fov_deg : float
        Horizontal field-of-view in degrees.  Typical drone cameras:
        90° (racing), 120° (inspection), 150° (wide-angle).
    aspect_ratio : float
        Width / height.  4/3 is standard FPV, 16/9 for cinematics.
    sensor_height_mm : float
        Sensor height in mm (used for Blender lens calculations).
        Infinigen defaults to 18 mm.
    """

    fov_deg: float = 90.0
    aspect_ratio: float = ASPECT_4_3
    sensor_height_mm: float = 18.0

    # Pre-computed (set in __post_init__)
    _focal_length_mm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 10.0 <= self.fov_deg <= 180.0:
            msg = f"fov_deg must be in [10, 180], got {self.fov_deg}"
            raise ValueError(msg)
        if self.aspect_ratio <= 0:
            msg = f"aspect_ratio must be positive, got {self.aspect_ratio}"
            raise ValueError(msg)
        if self.sensor_height_mm <= 0:
            msg = f"sensor_height_mm must be positive, got {self.sensor_height_mm}"
            raise ValueError(msg)
        sensor_width = self.sensor_height_mm * self.aspect_ratio
        object.__setattr__(
            self, "_focal_length_mm",
            sensor_width / (2.0 * math.tan(math.radians(self.fov_deg / 2.0))),
        )

    @property
    def focal_length_mm(self) -> float:
        """Focal length derived from FoV and sensor size (pre-computed)."""
        return self._focal_length_mm


@dataclass(frozen=True)
class CameraRigConfig:
    """Multi-camera rig for a single drone in the swarm.

    Matches Infinigen's ``camera.spawn_camera_rigs.camera_rig_config``
    gin binding, which expects a list of ``{loc, rot_euler}`` dicts.

    Parameters
    ----------
    cameras : tuple[dict[str, tuple[float, float, float]], ...]
        Each element is ``{"loc": (x,y,z), "rot_euler": (rx,ry,rz)}``
        relative to the rig parent.
    n_rigs : int
        Number of identical rigs to spawn (for multi-agent scenarios,
        each drone gets its own rig).
    stereo_baseline_m : float
        If > 0, a second camera is added at ``(baseline, 0, 0)`` from
        the first camera, forming a horizontal stereo pair.
    """

    cameras: tuple[dict[str, tuple[float, float, float]], ...] = (
        {"loc": (0, 0, 0), "rot_euler": (0, 0, 0)},
    )
    n_rigs: int = 1
    stereo_baseline_m: float = 0.0

    def __post_init__(self) -> None:
        if self.n_rigs < 1:
            msg = f"n_rigs must be >= 1, got {self.n_rigs}"
            raise ValueError(msg)
        if self.stereo_baseline_m < 0:
            msg = f"stereo_baseline_m must be non-negative, got {self.stereo_baseline_m}"
            raise ValueError(msg)
        if len(self.cameras) < 1:
            msg = "cameras must contain at least one camera"
            raise ValueError(msg)

    @property
    def effective_cameras(self) -> tuple[dict[str, tuple[float, float, float]], ...]:
        """Cameras including the auto-generated stereo pair if applicable."""
        if self.stereo_baseline_m <= 0:
            return self.cameras
        # Add a right-eye camera offset from the first camera
        first = self.cameras[0]
        lx, ly, lz = first["loc"]
        right_eye = {
            "loc": (lx + self.stereo_baseline_m, ly, lz),
            "rot_euler": first["rot_euler"],
        }
        return (*self.cameras, right_eye)

    def gin_overrides(self) -> dict[str, int | list[dict[str, tuple[float, float, float]]]]:
        """Return gin bindings for Infinigen's camera rig system."""
        cams = self.effective_cameras
        return {
            "camera.spawn_camera_rigs.n_camera_rigs": self.n_rigs,
            "camera.spawn_camera_rigs.camera_rig_config": list(cams),
        }

    @staticmethod
    def monocular(n_drones: int = 1) -> CameraRigConfig:
        """Single forward-facing camera per drone."""
        return CameraRigConfig(
            cameras=({"loc": (0, 0, 0), "rot_euler": (0, 0, 0)},),
            n_rigs=n_drones,
        )

    @staticmethod
    def stereo(baseline_m: float = 0.065, n_drones: int = 1) -> CameraRigConfig:
        """Horizontal stereo pair per drone (default: 65 mm human-eye baseline)."""
        return CameraRigConfig(
            cameras=({"loc": (0, 0, 0), "rot_euler": (0, 0, 0)},),
            n_rigs=n_drones,
            stereo_baseline_m=baseline_m,
        )
