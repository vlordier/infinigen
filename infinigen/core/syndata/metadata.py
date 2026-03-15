# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Metadata and annotation helpers for RL-relevant frame data.

:class:`FrameMetadata` is a lightweight container for per-frame annotations
that RL training loops commonly need: obstacle bounding boxes, depth
statistics, traversability masks, and camera pose.

All helpers are pure Python / NumPy — no ``bpy`` dependency.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["BBox3D", "DepthStats", "FrameMetadata"]


@dataclass(frozen=True, slots=True)
class BBox3D:
    """Axis-aligned 3-D bounding box.

    Parameters
    ----------
    center : tuple[float, float, float]
        Centre of the box ``(x, y, z)`` in world coordinates.
    extent : tuple[float, float, float]
        Half-extents ``(dx, dy, dz)``.
    label : str
        Semantic label (e.g. ``"tree"``, ``"building"``).
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0)
    label: str = "unknown"


@dataclass(frozen=True, slots=True)
class DepthStats:
    """Summary statistics for a depth map."""

    min_m: float = 0.0
    max_m: float = 100.0
    mean_m: float = 50.0
    median_m: float = 50.0
    std_m: float = 10.0

    @staticmethod
    def from_depth_array(
        depth: np.ndarray,
        *,
        clip_min: float = 0.0,
        clip_max: float = 1e4,
    ) -> DepthStats:
        """Compute stats from a depth array (any shape).

        Parameters
        ----------
        depth : np.ndarray
            Depth values in metres.  ``inf`` and ``nan`` are excluded.
        clip_min, clip_max : float
            Valid depth range — values outside are clipped before computing
            statistics.  This prevents outliers from dominating the stats.
        """
        flat = depth.ravel().astype(np.float64)
        # Exclude invalid (inf / nan) values
        valid = flat[np.isfinite(flat)]
        if valid.size == 0:
            logger.warning("from_depth_array: all depth values are inf/nan; returning defaults")
            return DepthStats()
        valid = np.clip(valid, clip_min, clip_max)
        return DepthStats(
            min_m=float(np.min(valid)),
            max_m=float(np.max(valid)),
            mean_m=float(np.mean(valid)),
            median_m=float(np.median(valid)),
            std_m=float(np.std(valid)),
        )


@dataclass
class FrameMetadata:
    """Per-frame annotation container for RL training.

    Parameters
    ----------
    frame_id : int
        Frame index within the episode.
    scene_seed : int
        Scene generation seed for reproducibility.
    camera_position : tuple[float, float, float]
        Camera world position ``(x, y, z)``.
    camera_rotation_euler : tuple[float, float, float]
        Camera Euler rotation ``(rx, ry, rz)`` in radians.
    obstacles : list[BBox3D]
        Bounding boxes of obstacles visible from the camera.
    depth_stats : DepthStats | None
        Depth-map summary if available.
    traversability_ratio : float
        Fraction of the frame area that is traversable (0–1).
    curriculum_stage : int
        Curriculum stage that generated this frame.
    velocity : tuple[float, float, float]
        Agent velocity ``(vx, vy, vz)`` in m/s.  Zero for static frames.
    nearest_obstacle_m : float
        Distance to the nearest obstacle surface in metres.
        ``inf`` if no obstacles are in range.  Critical for collision
        avoidance reward shaping.
    swarm_positions : list[tuple[float, float, float]]
        World positions of other agents in the swarm.  Empty list for
        single-agent scenarios.
    extra : dict[str, Any]
        Arbitrary additional metadata.
    """

    frame_id: int = 0
    scene_seed: int = 0
    camera_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_rotation_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacles: list[BBox3D] = field(default_factory=list)
    depth_stats: DepthStats | None = None
    traversability_ratio: float = 1.0
    curriculum_stage: int = 0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nearest_obstacle_m: float = float("inf")
    swarm_positions: list[tuple[float, float, float]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        ``float('inf')`` is serialised as the string ``"Infinity"`` so
        that standard JSON parsers don't choke.
        """
        d = asdict(self)
        # JSON has no infinity literal — encode as string
        if math.isinf(d.get("nearest_obstacle_m", 0)):
            d["nearest_obstacle_m"] = "Infinity"
        return d

    def save_json(self, path: str | Path) -> None:
        """Write metadata to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # Fields that are tuples in the dataclass but become lists in JSON.
    _TUPLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "camera_position", "camera_rotation_euler", "velocity",
    )

    @staticmethod
    def load_json(path: str | Path) -> FrameMetadata:
        """Load metadata from a JSON file."""
        data = json.loads(Path(path).read_text())
        obstacles = [
            BBox3D(
                center=tuple(o.get("center", (0, 0, 0))),
                extent=tuple(o.get("extent", (1, 1, 1))),
                label=o.get("label", "unknown"),
            )
            for o in data.pop("obstacles", [])
        ]
        ds = data.pop("depth_stats", None)
        depth_stats = DepthStats(**ds) if ds else None
        # Restore infinity from string encoding
        nom = data.get("nearest_obstacle_m")
        if nom == "Infinity":
            data["nearest_obstacle_m"] = float("inf")
        # JSON deserialises tuples as lists — convert them back
        for key in FrameMetadata._TUPLE_FIELDS:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        sp = data.pop("swarm_positions", [])
        swarm_positions = [tuple(p) for p in sp]
        return FrameMetadata(
            obstacles=obstacles,
            depth_stats=depth_stats,
            swarm_positions=swarm_positions,
            **data,
        )
