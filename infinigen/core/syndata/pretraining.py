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
...     flappy_drone_env_config,
... )
>>> cfg = FlappyColumnConfig(num_columns=5, corridor_height=1.5)
>>> obstacles = generate_flappy_obstacles(cfg, seed=42)
>>> env = flappy_drone_env_config(cfg, num_envs=512)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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


# ---------------------------------------------------------------------------
# Obstacle dataclass (lightweight, independent of FrameMetadata)
# ---------------------------------------------------------------------------


@dataclass
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
# Pre-configured DroneEnvConfig for flappy pre-training
# ---------------------------------------------------------------------------


def flappy_drone_env_config(
    config: FlappyColumnConfig | None = None,
    *,
    seed: int | None = None,
    num_envs: int = 2048,
    dt: float = 0.01,
) -> dict[str, Any]:
    """Build a GenesisDroneEnv-compatible config dict for flappy pre-training.

    Returns a dict that can be passed to
    :class:`~infinigen.core.syndata.drone_env_bridge.DroneEnvConfig`
    or serialised to YAML for GenesisDroneEnv.

    Parameters
    ----------
    config : FlappyColumnConfig | None
        Corridor configuration.  If *None*, uses defaults.
    seed : int | None
        RNG seed for obstacle placement.
    num_envs : int
        Number of parallel environments.
    dt : float
        Physics timestep.

    Returns
    -------
    dict[str, Any]
        Configuration dict with keys matching
        :class:`~infinigen.core.syndata.drone_env_bridge.DroneEnvConfig`.

    Raises
    ------
    ValueError
        If *num_envs* < 1 or *dt* <= 0.
    """
    if num_envs < 1:
        msg = f"num_envs must be >= 1, got {num_envs}"
        raise ValueError(msg)
    if dt <= 0:
        msg = f"dt must be positive, got {dt}"
        raise ValueError(msg)

    if config is None:
        config = FlappyColumnConfig()

    obstacles = generate_flappy_obstacles(config, seed=seed)

    # Build obstacle entity dicts for DroneEnvConfig
    obstacle_entities: list[dict[str, Any]] = []
    for obs in obstacles:
        cx, cy, cz = obs.center
        dx, dy, dz = obs.half_extents
        obstacle_entities.append({
            "name": obs.label,
            "morph_type": "Box",
            "pos": (cx, cy, cz),
            "size": (dx * 2, dy * 2, dz * 2),
            "fixed": True,
            "collision": True,
        })

    # Simple init: start at the beginning of the corridor, mid-height
    init_z = config.corridor_height / 2
    half_w = config.corridor_width / 4

    return {
        "num_envs": num_envs,
        "dt": dt,
        "cam_res": (128, 128),
        "drone_init_pos": (0.3, 0.0, init_z),
        "map_size": (config.corridor_length + 1.0, config.corridor_width + 1.0),
        "obstacle_entities": obstacle_entities,
        "episode_length_s": 10.0,
        "max_episode_length": 1000,
        "init_pos_range": {
            "x": (-0.1, 0.1),
            "y": (-half_w, half_w),
            "z": (init_z - 0.1, init_z + 0.1),
        },
        # Simple reward: target at end of corridor, harsh crash penalty
        "reward_scales": {
            "target": 15.0,
            "smooth": -0.0001,
            "yaw": 0.0,
            "angular": -0.0001,
            "crash": -15.0,
        },
        # Tight command ranges — just fly forward
        "command_ranges": {
            "x": (config.corridor_length * 0.7, config.corridor_length * 0.9),
            "y": (-0.3, 0.3),
            "z": (init_z - 0.2, init_z + 0.2),
        },
        # Strict termination — any tumble = failure
        "termination_conditions": {
            "roll_deg": 60.0,
            "pitch_deg": 60.0,
            "ground_m": 0.08,
            "x_m": config.corridor_length + 0.5,
            "y_m": config.corridor_width / 2 + 0.5,
            "z_m": config.corridor_height + 0.3,
        },
    }


def flappy_genesis_entities(
    config: FlappyColumnConfig | None = None,
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate Genesis entity configs for the flappy corridor.

    Returns a list of dicts compatible with
    :class:`~infinigen.core.syndata.genesis_export.GenesisEntityConfig`
    constructor kwargs.

    Parameters
    ----------
    config : FlappyColumnConfig | None
        Corridor configuration.  If *None*, uses defaults.
    seed : int | None
        RNG seed for obstacle placement.

    Returns
    -------
    list[dict[str, Any]]
        Entity config dicts with ``name``, ``morph_type``, ``pos``,
        ``is_fixed``, and ``extra`` keys.
    """
    if config is None:
        config = FlappyColumnConfig()

    obstacles = generate_flappy_obstacles(config, seed=seed)

    entities: list[dict[str, Any]] = []
    for obs in obstacles:
        cx, cy, cz = obs.center
        dx, dy, dz = obs.half_extents
        entities.append({
            "name": obs.label,
            "morph_type": "Box",
            "pos": (cx, cy, cz),
            "is_fixed": True,
            "extra": {"size": (dx * 2, dy * 2, dz * 2)},
        })

    return entities
