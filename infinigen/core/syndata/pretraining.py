# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Ultra-simple pre-training environments for early curriculum stages.

Provides procedural obstacle generators that produce trivially simple
3D scenes for bootstrapping RL agent training *before* exposing them
to full Infinigen-generated environments.  Think "3D Flappy Bird":
a corridor with column-gap obstacles where the agent learns basic
flight control and collision avoidance.

These are intentionally minimal — no photorealism, no domain
randomisation beyond obstacle placement.  The goal is to reach
a baseline policy quickly so later curriculum stages (which use
expensive Infinigen scenes) converge faster.

All helpers are pure Python / NumPy — no ``bpy`` or ``genesis``
dependency at import time.

Usage
-----
>>> from infinigen.core.syndata.pretraining import (
...     FlappyColumnConfig, generate_flappy_obstacles,
... )
>>> cfg = FlappyColumnConfig(num_columns=5, corridor_height=1.5)
>>> obstacles = generate_flappy_obstacles(cfg, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FlappyColumnConfig",
    "FlappyObstacle",
    "flappy_frame_metadata",
    "generate_flappy_obstacles",
]

# ---------------------------------------------------------------------------
# Flappy-bird column config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlappyColumnConfig:
    """Configuration for a "3D Flappy Bird" pre-training corridor.

    Generates a straight corridor with evenly-spaced column-gap
    obstacles.  The agent flies through gaps in the columns — this
    is the simplest possible collision-avoidance task.

    Parameters
    ----------
    corridor_length : float
        Total corridor length (metres) along the X axis.
    corridor_width : float
        Corridor width (metres) along the Y axis.
    corridor_height : float
        Flight ceiling (metres) along the Z axis.
    num_columns : int
        Number of column obstacles along the corridor.
    gap_height : float
        Vertical gap size (metres) that the agent must fly through.
        Must be at least 0.2 m.
    gap_height_variation : float
        Random variation in gap centre height (metres). 0 = all gaps
        at the same height.  Must be non-negative.
    column_width : float
        Width of each column obstacle (metres).
    column_depth : float
        Depth of each column obstacle (metres).
    min_gap_z : float
        Minimum gap centre z-coordinate (metres above ground).
    max_gap_z : float
        Maximum gap centre z-coordinate (metres above ground).
    """

    corridor_length: float = 8.0
    corridor_width: float = 2.0
    corridor_height: float = 2.0
    num_columns: int = 5
    gap_height: float = 0.6
    gap_height_variation: float = 0.2
    column_width: float = 0.3
    column_depth: float = 1.8
    min_gap_z: float = 0.5
    max_gap_z: float = 1.5

    def __post_init__(self) -> None:
        if self.corridor_length <= 0:
            msg = f"corridor_length must be positive, got {self.corridor_length}"
            raise ValueError(msg)
        if self.corridor_width <= 0:
            msg = f"corridor_width must be positive, got {self.corridor_width}"
            raise ValueError(msg)
        if self.corridor_height <= 0:
            msg = f"corridor_height must be positive, got {self.corridor_height}"
            raise ValueError(msg)
        if self.num_columns < 0:
            msg = f"num_columns must be non-negative, got {self.num_columns}"
            raise ValueError(msg)
        if self.gap_height < 0.2:
            msg = f"gap_height must be >= 0.2, got {self.gap_height}"
            raise ValueError(msg)
        if self.gap_height_variation < 0:
            msg = f"gap_height_variation must be non-negative, got {self.gap_height_variation}"
            raise ValueError(msg)
        if self.column_width <= 0:
            msg = f"column_width must be positive, got {self.column_width}"
            raise ValueError(msg)
        if self.column_depth <= 0:
            msg = f"column_depth must be positive, got {self.column_depth}"
            raise ValueError(msg)
        if self.min_gap_z <= 0:
            msg = f"min_gap_z must be positive, got {self.min_gap_z}"
            raise ValueError(msg)
        if self.max_gap_z < self.min_gap_z:
            msg = f"max_gap_z ({self.max_gap_z}) must be >= min_gap_z ({self.min_gap_z})"
            raise ValueError(msg)

    # ---- Preset factory methods ---------------------------------------------

    @staticmethod
    def easy(*, seed: int | None = None) -> FlappyColumnConfig:
        """Wide gaps, few columns, wide corridor — for initial learning.

        The agent barely needs to adjust altitude; this is the absolute
        simplest collision-avoidance task.
        """
        _ = seed  # reserved for future per-preset seed forwarding
        return FlappyColumnConfig(
            corridor_length=10.0,
            corridor_width=3.0,
            corridor_height=3.0,
            num_columns=3,
            gap_height=1.2,
            gap_height_variation=0.3,
            column_width=0.3,
            column_depth=2.5,
            min_gap_z=0.8,
            max_gap_z=2.2,
        )

    @staticmethod
    def medium(*, seed: int | None = None) -> FlappyColumnConfig:
        """Moderate gaps and more columns — for skill refinement.

        The agent must make deliberate altitude adjustments and plan
        ahead across several obstacles.
        """
        _ = seed  # reserved for future per-preset seed forwarding
        return FlappyColumnConfig(
            corridor_length=12.0,
            corridor_width=2.0,
            corridor_height=2.5,
            num_columns=5,
            gap_height=0.8,
            gap_height_variation=0.3,
            column_width=0.3,
            column_depth=1.8,
            min_gap_z=0.6,
            max_gap_z=1.9,
        )

    @staticmethod
    def hard(*, seed: int | None = None) -> FlappyColumnConfig:
        """Narrow gaps, many columns, narrow corridor — approaching real-world.

        Tight manoeuvring with minimal margin for error.  Agents that
        pass this are ready for full Infinigen scenes.
        """
        _ = seed  # reserved for future per-preset seed forwarding
        return FlappyColumnConfig(
            corridor_length=15.0,
            corridor_width=1.5,
            corridor_height=2.0,
            num_columns=8,
            gap_height=0.5,
            gap_height_variation=0.2,
            column_width=0.4,
            column_depth=1.3,
            min_gap_z=0.4,
            max_gap_z=1.6,
        )


# ---------------------------------------------------------------------------
# Obstacle dataclass (lightweight, independent of FrameMetadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlappyObstacle:
    """A single box obstacle in the flappy corridor.

    Parameters
    ----------
    center : tuple[float, float, float]
        Centre position ``(x, y, z)`` in world coordinates.
    half_extents : tuple[float, float, float]
        Half-size ``(dx, dy, dz)`` in each axis.
    label : str
        Semantic label (e.g. ``"column_upper_0"``).
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1)
    label: str = "obstacle"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "center": list(self.center),
            "half_extents": list(self.half_extents),
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# Procedural obstacle generation
# ---------------------------------------------------------------------------


def generate_flappy_obstacles(
    config: FlappyColumnConfig,
    *,
    seed: int | None = None,
) -> list[FlappyObstacle]:
    """Generate column-gap obstacles for a flappy-bird corridor.

    Each column consists of an upper and lower box with a gap between
    them.  The gap centre height is randomised within the configured
    bounds.

    Parameters
    ----------
    config : FlappyColumnConfig
        Corridor and obstacle parameters.
    seed : int | None
        RNG seed for reproducible gap placement.

    Returns
    -------
    list[FlappyObstacle]
        Upper and lower box obstacles for each column (2 × num_columns),
        plus floor and ceiling bounding boxes.

    Raises
    ------
    TypeError
        If *config* is not a :class:`FlappyColumnConfig`.
    """
    if not isinstance(config, FlappyColumnConfig):
        msg = f"config must be FlappyColumnConfig, got {type(config).__name__}"
        raise TypeError(msg)

    rng = np.random.default_rng(seed)
    obstacles: list[FlappyObstacle] = []

    # Column spacing along the corridor
    if config.num_columns > 0:
        margin = config.corridor_length * 0.1
        spacing = (config.corridor_length - 2 * margin) / max(1, config.num_columns)
    else:
        margin = 0.0
        spacing = 0.0

    half_cw = config.column_width / 2
    half_cd = config.column_depth / 2
    half_gap = config.gap_height / 2

    for i in range(config.num_columns):
        col_x = margin + spacing * (i + 0.5)
        col_y = 0.0  # centred in corridor

        # Randomise gap centre height
        gap_z = rng.uniform(
            max(config.min_gap_z, half_gap),
            min(config.max_gap_z, config.corridor_height - half_gap),
        )
        if config.gap_height_variation > 0:
            jitter = rng.uniform(
                -config.gap_height_variation / 2,
                config.gap_height_variation / 2,
            )
            gap_z = float(np.clip(
                gap_z + jitter,
                half_gap + 0.05,
                config.corridor_height - half_gap - 0.05,
            ))

        # Lower column: from ground (z=0) to gap bottom
        lower_top = gap_z - half_gap
        if lower_top > 0.01:
            lower_half_h = lower_top / 2
            obstacles.append(FlappyObstacle(
                center=(col_x, col_y, lower_half_h),
                half_extents=(half_cw, half_cd, lower_half_h),
                label=f"column_lower_{i}",
            ))

        # Upper column: from gap top to ceiling
        upper_bottom = gap_z + half_gap
        upper_height = config.corridor_height - upper_bottom
        if upper_height > 0.01:
            upper_half_h = upper_height / 2
            upper_z = upper_bottom + upper_half_h
            obstacles.append(FlappyObstacle(
                center=(col_x, col_y, upper_z),
                half_extents=(half_cw, half_cd, upper_half_h),
                label=f"column_upper_{i}",
            ))

    # Floor and ceiling
    floor_half_thick = 0.05
    obstacles.append(FlappyObstacle(
        center=(config.corridor_length / 2, 0.0, -floor_half_thick),
        half_extents=(config.corridor_length / 2, config.corridor_width / 2, floor_half_thick),
        label="floor",
    ))
    obstacles.append(FlappyObstacle(
        center=(config.corridor_length / 2, 0.0, config.corridor_height + floor_half_thick),
        half_extents=(config.corridor_length / 2, config.corridor_width / 2, floor_half_thick),
        label="ceiling",
    ))

    return obstacles


# ---------------------------------------------------------------------------
# Frame metadata helper
# ---------------------------------------------------------------------------


def flappy_frame_metadata(
    config: FlappyColumnConfig,
    *,
    seed: int | None = None,
    frame_id: int = 0,
    scene_seed: int = 0,
) -> dict[str, Any]:
    """Generate a FrameMetadata-compatible dict for a flappy corridor.

    Combines :func:`generate_flappy_obstacles` with basic depth and
    traversability estimates derived purely from corridor geometry.
    Useful for pre-flight validation and curriculum tracking without
    running a full Infinigen render.

    Parameters
    ----------
    config : FlappyColumnConfig
        Corridor and obstacle parameters.
    seed : int | None
        RNG seed for reproducible obstacle placement.
    frame_id : int
        Frame identifier.
    scene_seed : int
        Scene RNG seed.

    Returns
    -------
    dict[str, Any]
        FrameMetadata-compatible dict with obstacles, depth_stats,
        traversability_ratio, and camera position.

    Raises
    ------
    TypeError
        If *config* is not a :class:`FlappyColumnConfig`.
    """
    if not isinstance(config, FlappyColumnConfig):
        msg = f"config must be FlappyColumnConfig, got {type(config).__name__}"
        raise TypeError(msg)

    obstacles = generate_flappy_obstacles(config, seed=seed)

    # Estimate depth from camera (placed near corridor start) to obstacles
    cam_x = 0.3
    col_obstacles = [o for o in obstacles if "column" in o.label]
    if col_obstacles:
        dists = [abs(o.center[0] - cam_x) for o in col_obstacles]
        min_dist = max(0.01, min(dists))
        max_dist = config.corridor_length
    else:
        min_dist = 0.5
        max_dist = config.corridor_length

    traversability = min(1.0, config.gap_height / config.corridor_height)

    return {
        "frame_id": frame_id,
        "scene_seed": scene_seed,
        "camera_position": (cam_x, 0.0, config.corridor_height / 2),
        "camera_rotation_euler": (0.0, 0.0, 0.0),
        "obstacles": [o.to_dict() for o in obstacles],
        "depth_stats": {
            "min_m": round(min_dist, 3),
            "max_m": round(max_dist, 3),
            "mean_m": round((min_dist + max_dist) / 2, 3),
            "median_m": round((min_dist + max_dist) / 2, 3),
            "std_m": round((max_dist - min_dist) / 4, 3),
        },
        "traversability_ratio": round(traversability, 3),
        "curriculum_stage": 0,
        "velocity": (0.0, 0.0, 0.0),
        "nearest_obstacle_m": round(min_dist, 3),
        "swarm_positions": [],
    }
