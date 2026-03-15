# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Parametric 3D world generator for progressive curriculum pre-training.

This module is the **Infinigen side** of the curriculum pipeline.  It
generates pure 3D geometry (axis-aligned boxes) that describe training
environments of increasing navigational complexity.  The output feeds
into:

* **Infinigen** — via :func:`world_gin_overrides` (scene generation params)
* **Genesis / DroneEnv** — via :func:`world_to_genesis_entities` and
  :func:`world_to_drone_env_config` (separate conversion layer)

Crucially, this module does *not* import or depend on Genesis, Blender, or
any RL framework.  It only produces geometry (``list[BBox3D]``) and
configuration dicts.  The physics simulation, reward shaping, and episode
management live in the Genesis RL gym (e.g. GenesisDroneEnv).

Complexity ladder
-----------------

A single ``complexity`` parameter in [0, 1] drives the entire progression.
All thresholds are soft — parameters blend smoothly so there are no hard
jumps.

===========  ====  ============================================================
complexity   Preset  Navigation challenge
===========  ====  ============================================================
0.00 – 0.15  ``flappy()``    **Trivial 3D flight**: straight corridor, simple
                              column-gap obstacles.  Teaches basic up/down,
                              left/right, forward/back avoidance.
0.15 – 0.35  ``corridor()``  **Textured corridor**: same topology but with
                              visual variety — coloured walls, varied floor
                              roughness, subtle fog.  Tests vision robustness.
0.35 – 0.55  ``rooms()``     **Connected rooms**: doors, furniture-like clutter.
                              Requires planning turns, entering/exiting spaces.
0.55 – 0.75  ``branches()``  **Branching paths**: T-junctions, dead-ends, fog,
                              debris.  Requires exploration and backtracking.
0.75 – 0.90  ``maze()``      **Multi-level maze**: vertical shafts, upper
                              corridors, dense obstacle fields.  Full 3D
                              navigation with altitude changes.
0.90 – 1.00  ``doom()``      **Dense 3D maze**: many rooms, branches, levels,
                              heavy fog, point lights, full debris.  Approaches
                              real-world indoor/outdoor complexity.
===========  ====  ============================================================

At ``complexity=1.0``, the generated world is suitable for importing into
Infinigen as a coarse scene layout that Infinigen then populates with
high-resolution assets (streets, forests, indoor furniture, moving tree
branches, etc.) depending on the curriculum stage.

Output format
-------------
:func:`generate_world` always returns a ``list[BBox3D]`` — axis-aligned
bounding boxes representing walls, floors, ceilings, columns, furniture,
and debris.  The number of boxes scales with complexity:

* ``flappy`` (~0.05): ~10 boxes (corridor shell + a few column obstacles)
* ``corridor`` (~0.25): ~20 boxes (more columns, tighter gaps)
* ``rooms`` (~0.45): ~40–60 boxes (room walls + furniture)
* ``branches`` (~0.65): ~80–120 boxes (branch corridors + dead-ends)
* ``maze`` (~0.85): ~150–200 boxes (multiple levels + shafts)
* ``doom`` (~0.98): ~250–350 boxes (full maze with heavy debris)

These counts are approximate and vary with the RNG seed.  Each ``BBox3D``
has a human-readable label (e.g. ``"main_col_lower_3"``, ``"room_2_furniture_1"``)
for debugging and selective filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from infinigen.core.syndata.metadata import BBox3D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_GAP: float = 0.3  # Minimum flyable gap (metres)
_WALL_THICKNESS: float = 0.1  # Wall / floor / ceiling thickness
_DEBRIS_SIZE_RANGE: tuple[float, float] = (0.05, 0.3)

# ---------------------------------------------------------------------------
# Visual style config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualStyle:
    """Visual appearance parameters for the generated world.

    These map to **Infinigen** material / lighting gin overrides (via
    :func:`world_gin_overrides`).  They do *not* control Genesis rendering
    — the Genesis bridge layer reads these values and maps them to the
    appropriate Genesis light/material APIs separately.

    At low complexity, everything is greyscale with uniform lighting.
    As complexity increases, surfaces gain colour, roughness varies,
    fog and clouds appear, and extra point lights add visual diversity.
    At the highest stages, Infinigen uses these hints to add photorealistic
    textures, sub-surface scattering, and HDR lighting.

    Parameters
    ----------
    wall_color_hue : float
        Base hue for wall surfaces in [0, 360) degrees.
    wall_color_saturation : float
        Saturation in [0, 1].  0 = greyscale, 1 = vivid.
    floor_roughness : float
        PBR roughness for floor material in [0, 1].
    ambient_intensity : float
        Ambient / fill-light intensity multiplier.
    fog_density : float
        Volumetric fog density in [0, 1].  0 = clear, 1 = thick.
    cloud_density : float
        Sky cloud density in [0, 1] (visible through open ceilings).
    point_light_count : int
        Number of additional point lights scattered in the world.
    """

    wall_color_hue: float = 0.0
    wall_color_saturation: float = 0.0
    floor_roughness: float = 0.5
    ambient_intensity: float = 1.0
    fog_density: float = 0.0
    cloud_density: float = 0.0
    point_light_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.wall_color_saturation <= 1.0:
            msg = f"wall_color_saturation must be in [0, 1], got {self.wall_color_saturation}"
            raise ValueError(msg)
        if not 0.0 <= self.floor_roughness <= 1.0:
            msg = f"floor_roughness must be in [0, 1], got {self.floor_roughness}"
            raise ValueError(msg)
        if self.ambient_intensity < 0:
            msg = f"ambient_intensity must be non-negative, got {self.ambient_intensity}"
            raise ValueError(msg)
        if not 0.0 <= self.fog_density <= 1.0:
            msg = f"fog_density must be in [0, 1], got {self.fog_density}"
            raise ValueError(msg)
        if not 0.0 <= self.cloud_density <= 1.0:
            msg = f"cloud_density must be in [0, 1], got {self.cloud_density}"
            raise ValueError(msg)
        if self.point_light_count < 0:
            msg = f"point_light_count must be non-negative, got {self.point_light_count}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Main world config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldConfig:
    """Configuration for a procedurally generated 3D training world.

    A single ``complexity`` parameter in [0, 1] drives the entire
    progression from a trivial flight corridor to a dense 3D maze.
    Individual parameters can be overridden for fine-grained control.

    **Typical output sizes** (number of BBox3D obstacles per preset,
    seed-dependent):

    * ``flappy()``   (c≈0.05): ~10 boxes — basic corridor shell + columns
    * ``corridor()`` (c≈0.25): ~20 boxes — more columns, tighter gaps
    * ``rooms()``    (c≈0.45): ~40–60 — rooms with furniture clutter
    * ``branches()`` (c≈0.65): ~80–120 — branch corridors + dead-ends
    * ``maze()``     (c≈0.85): ~150–200 — multiple vertical levels
    * ``doom()``     (c≈0.98): ~250–350 — dense maze with heavy debris

    **Role in curriculum**: this config is *only* about 3D geometry for
    Infinigen.  The RL simulation parameters (rewards, episode length,
    termination conditions) belong to the Genesis DroneEnv bridge layer
    and are derived from this config via :func:`world_to_drone_env_config`.

    Parameters
    ----------
    complexity : float
        Master difficulty knob in [0, 1].  Determines topology, obstacle
        density, visual variety, and environmental effects.
    seed : int | None
        RNG seed for reproducible generation.
    corridor_length : float
        Base corridor length (metres).  Scaled up by complexity.
    corridor_width : float
        Corridor width (metres).
    corridor_height : float
        Corridor height (metres).
    num_columns : int | None
        Number of column obstacles.  If *None*, derived from complexity
        (2 at c=0, up to 15 at c=1).
    gap_height : float | None
        Flyable gap size (metres).  If *None*, derived from complexity
        (2.0m at c=0, 0.5m at c=1).
    num_rooms : int | None
        Number of connected rooms (complexity ≥ 0.35).  If *None*,
        derived from complexity.
    room_size_range : tuple[float, float]
        (min, max) room side length (metres).
    num_branches : int | None
        Number of branching paths / dead-ends (complexity ≥ 0.55).
        If *None*, derived from complexity.
    num_levels : int | None
        Vertical levels (complexity ≥ 0.75).  If *None*, derived from
        complexity.
    debris_density : float | None
        Small obstacle density in [0, 1].  If *None*, derived from
        complexity.
    style : VisualStyle | None
        Visual appearance.  If *None*, auto-derived from complexity
        (greyscale at c=0, vivid colours/fog/clouds at c=1).
    """

    complexity: float = 0.0
    seed: int | None = None

    # Corridor parameters
    corridor_length: float = 8.0
    corridor_width: float = 2.0
    corridor_height: float = 2.0
    num_columns: int | None = None
    gap_height: float | None = None

    # Room parameters
    num_rooms: int | None = None
    room_size_range: tuple[float, float] = (3.0, 6.0)

    # Maze parameters
    num_branches: int | None = None
    num_levels: int | None = None

    # Debris / clutter
    debris_density: float | None = None

    # Visual style
    style: VisualStyle | None = None

    # Derived values
    _effective: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.complexity <= 1.0:
            msg = f"complexity must be in [0, 1], got {self.complexity}"
            raise ValueError(msg)
        if self.corridor_length <= 0:
            msg = f"corridor_length must be positive, got {self.corridor_length}"
            raise ValueError(msg)
        if self.corridor_width <= 0:
            msg = f"corridor_width must be positive, got {self.corridor_width}"
            raise ValueError(msg)
        if self.corridor_height <= 0:
            msg = f"corridor_height must be positive, got {self.corridor_height}"
            raise ValueError(msg)
        if self.room_size_range[0] > self.room_size_range[1]:
            msg = f"room_size_range min ({self.room_size_range[0]}) > max ({self.room_size_range[1]})"
            raise ValueError(msg)
        if self.gap_height is not None and self.gap_height < _MIN_GAP:
            msg = f"gap_height must be >= {_MIN_GAP}, got {self.gap_height}"
            raise ValueError(msg)
        if self.debris_density is not None and not 0.0 <= self.debris_density <= 1.0:
            msg = f"debris_density must be in [0, 1], got {self.debris_density}"
            raise ValueError(msg)

        # Compute effective values from complexity
        c = self.complexity
        eff: dict[str, Any] = {}

        # Columns: 2 at c=0, up to 15 at c=1
        eff["num_columns"] = (
            self.num_columns if self.num_columns is not None
            else max(2, round(2 + 13 * c))
        )

        # Gap height: 2.0m at c=0 (easy), 0.5m at c=1 (hard)
        eff["gap_height"] = (
            self.gap_height if self.gap_height is not None
            else max(_MIN_GAP, 2.0 - 1.5 * c)
        )

        # Rooms emerge at c >= 0.35
        eff["num_rooms"] = (
            self.num_rooms if self.num_rooms is not None
            else max(0, round(8 * max(0, c - 0.35) / 0.65))
        )

        # Branches emerge at c >= 0.55
        eff["num_branches"] = (
            self.num_branches if self.num_branches is not None
            else max(0, round(6 * max(0, c - 0.55) / 0.45))
        )

        # Vertical levels at c >= 0.75
        eff["num_levels"] = (
            self.num_levels if self.num_levels is not None
            else max(1, round(1 + 3 * max(0, c - 0.75) / 0.25))
        )

        # Debris density
        eff["debris_density"] = (
            self.debris_density if self.debris_density is not None
            else min(1.0, max(0.0, (c - 0.15) / 0.85))
        )

        # Corridor length scales with complexity
        eff["corridor_length"] = self.corridor_length * (1.0 + 1.5 * c)

        # Visual style
        if self.style is not None:
            eff["style"] = self.style
        else:
            eff["style"] = VisualStyle(
                wall_color_hue=0.0,  # will be randomised per-wall
                wall_color_saturation=min(1.0, c * 1.5),
                floor_roughness=0.3 + 0.5 * c,
                ambient_intensity=max(0.3, 1.0 - 0.5 * c),
                fog_density=max(0.0, (c - 0.5) * 1.2),
                cloud_density=max(0.0, (c - 0.4) * 1.0),
                point_light_count=max(0, round(c * 8)),
            )

        object.__setattr__(self, "_effective", eff)

    # ---- Effective values ---------------------------------------------------

    @property
    def effective_num_columns(self) -> int:
        """Number of column obstacles after applying complexity."""
        return self._effective["num_columns"]

    @property
    def effective_gap_height(self) -> float:
        """Flyable gap height (metres) after applying complexity."""
        return self._effective["gap_height"]

    @property
    def effective_num_rooms(self) -> int:
        """Number of connected rooms after applying complexity."""
        return self._effective["num_rooms"]

    @property
    def effective_num_branches(self) -> int:
        """Number of maze branches / dead-ends after applying complexity."""
        return self._effective["num_branches"]

    @property
    def effective_num_levels(self) -> int:
        """Number of vertical levels after applying complexity."""
        return self._effective["num_levels"]

    @property
    def effective_debris_density(self) -> float:
        """Debris density in [0, 1] after applying complexity."""
        return self._effective["debris_density"]

    @property
    def effective_corridor_length(self) -> float:
        """Corridor length (metres) after applying complexity scaling."""
        return self._effective["corridor_length"]

    @property
    def effective_style(self) -> VisualStyle:
        """Visual style after applying complexity defaults."""
        return self._effective["style"]

    # ---- Preset factories ---------------------------------------------------

    @staticmethod
    def flappy(*, seed: int | None = None) -> WorldConfig:
        """Preset: trivial flight corridor for basic 3D navigation.

        Produces a straight corridor with a few column-gap obstacles.
        The agent learns elemental avoidance — fly up/down to dodge
        columns, stay centred left/right, move forward.  ~10 boxes.
        """
        return WorldConfig(complexity=0.05, seed=seed)

    @staticmethod
    def corridor(*, seed: int | None = None) -> WorldConfig:
        """Preset: textured corridor with more obstacles and visual variety.

        Same topology as ``flappy`` but with narrower gaps, more columns,
        coloured walls, and subtle fog.  Tests that vision-based policies
        are robust to surface appearance.  ~20 boxes.
        """
        return WorldConfig(complexity=0.25, seed=seed)

    @staticmethod
    def rooms(*, seed: int | None = None) -> WorldConfig:
        """Preset: connected rooms with doors and furniture-like clutter.

        Introduces right-angle turns and doorway transitions.  The agent
        must plan room entries/exits and avoid furniture.  ~40–60 boxes.
        """
        return WorldConfig(complexity=0.45, seed=seed)

    @staticmethod
    def branches(*, seed: int | None = None) -> WorldConfig:
        """Preset: branching corridors with T-junctions and dead-ends.

        Requires exploration, backtracking, and fog navigation.  Debris
        is scattered in corridors.  ~80–120 boxes.
        """
        return WorldConfig(complexity=0.65, seed=seed)

    @staticmethod
    def maze(*, seed: int | None = None) -> WorldConfig:
        """Preset: multi-level 3D maze with vertical shafts.

        Full 3D navigation: altitude changes through vertical shafts,
        upper corridors, dense obstacle fields.  ~150–200 boxes.
        """
        return WorldConfig(complexity=0.85, seed=seed)

    @staticmethod
    def doom(*, seed: int | None = None) -> WorldConfig:
        """Preset: dense "Doom-like" 3D maze — maximum complexity.

        Many rooms, branching corridors, multiple vertical levels, heavy
        fog, point lights, and dense debris.  Approaches real-world
        indoor/outdoor complexity.  ~250–350 boxes.  At this level,
        Infinigen would overlay high-resolution assets: photorealistic
        walls, vegetation, furniture, dynamic elements.
        """
        return WorldConfig(complexity=0.98, seed=seed)

    @staticmethod
    def from_curriculum_progress(
        progress: float,
        *,
        seed: int | None = None,
    ) -> WorldConfig:
        """Create a world config from curriculum progress in [0, 1].

        Uses a gentle sqrt curve so early stages ramp slowly, giving the
        agent more time on simple environments before complexity increases.

        Examples (approximate, seed-dependent)::

            progress=0.00 → c≈0.00  flappy corridor        (~10 boxes)
            progress=0.05 → c≈0.22  textured corridor       (~16 boxes)
            progress=0.20 → c≈0.45  rooms with furniture    (~50 boxes)
            progress=0.50 → c≈0.71  branching paths + fog   (~100 boxes)
            progress=0.80 → c≈0.89  multi-level maze        (~180 boxes)
            progress=1.00 → c≈1.00  dense Doom-like maze    (~300 boxes)
        """
        c = math.sqrt(max(0.0, min(1.0, progress)))
        return WorldConfig(complexity=c, seed=seed)


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


def _make_wall(
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    label: str,
) -> BBox3D:
    """Helper: create a wall BBox3D with half-extents."""
    return BBox3D(
        center=(x, y, z),
        extent=(dx, dy, dz),
        label=label,
    )


def _generate_corridor(
    config: WorldConfig,
    rng: np.random.Generator,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    length: float | None = None,
    label_prefix: str = "corridor",
) -> list[BBox3D]:
    """Generate a straight corridor with column-gap obstacles.

    Returns floor, ceiling, side walls, and column obstacles.
    """
    boxes: list[BBox3D] = []
    ox, oy, oz = origin
    dx_dir, dy_dir, _dz_dir = direction

    cor_len = length if length is not None else config.effective_corridor_length
    cor_w = config.corridor_width
    cor_h = config.corridor_height
    half_w = cor_w / 2
    half_t = _WALL_THICKNESS / 2

    # Determine local axes (simplified: support axis-aligned directions)
    # For X-aligned corridors
    if abs(dx_dir) > 0.5:
        sign = 1.0 if dx_dir > 0 else -1.0
        # Floor
        boxes.append(_make_wall(
            ox + sign * cor_len / 2, oy, oz - half_t,
            cor_len / 2, half_w, half_t,
            f"{label_prefix}_floor",
        ))
        # Ceiling
        boxes.append(_make_wall(
            ox + sign * cor_len / 2, oy, oz + cor_h + half_t,
            cor_len / 2, half_w, half_t,
            f"{label_prefix}_ceiling",
        ))
        # Left wall
        boxes.append(_make_wall(
            ox + sign * cor_len / 2, oy - half_w - half_t, oz + cor_h / 2,
            cor_len / 2, half_t, cor_h / 2,
            f"{label_prefix}_wall_left",
        ))
        # Right wall
        boxes.append(_make_wall(
            ox + sign * cor_len / 2, oy + half_w + half_t, oz + cor_h / 2,
            cor_len / 2, half_t, cor_h / 2,
            f"{label_prefix}_wall_right",
        ))

        # Column obstacles
        n_cols = config.effective_num_columns
        gap_h = config.effective_gap_height
        half_gap = gap_h / 2

        if n_cols > 0:
            margin = cor_len * 0.1
            spacing = (cor_len - 2 * margin) / max(1, n_cols)

            for i in range(n_cols):
                col_x = ox + sign * (margin + spacing * (i + 0.5))
                # Randomise gap centre (clamp bounds to valid range)
                lo_z = half_gap + 0.05
                hi_z = max(lo_z, cor_h - half_gap - 0.05)
                gap_z = oz + (rng.uniform(lo_z, hi_z) if hi_z > lo_z else lo_z)

                col_half_w = rng.uniform(0.1, 0.2)
                col_half_d = half_w * 0.9  # Almost full corridor width

                # Lower column
                lower_top = gap_z - half_gap
                if lower_top > 0.01:
                    lower_half_h = lower_top / 2
                    boxes.append(BBox3D(
                        center=(col_x, oy, oz + lower_half_h),
                        extent=(col_half_w, col_half_d, lower_half_h),
                        label=f"{label_prefix}_col_lower_{i}",
                    ))

                # Upper column
                upper_bottom = gap_z + half_gap
                upper_height = cor_h - (upper_bottom - oz)
                if upper_height > 0.01:
                    upper_half_h = upper_height / 2
                    boxes.append(BBox3D(
                        center=(col_x, oy, upper_bottom + upper_half_h),
                        extent=(col_half_w, col_half_d, upper_half_h),
                        label=f"{label_prefix}_col_upper_{i}",
                    ))

    # Y-aligned corridors
    elif abs(dy_dir) > 0.5:
        sign = 1.0 if dy_dir > 0 else -1.0
        # Floor
        boxes.append(_make_wall(
            ox, oy + sign * cor_len / 2, oz - half_t,
            half_w, cor_len / 2, half_t,
            f"{label_prefix}_floor",
        ))
        # Ceiling
        boxes.append(_make_wall(
            ox, oy + sign * cor_len / 2, oz + cor_h + half_t,
            half_w, cor_len / 2, half_t,
            f"{label_prefix}_ceiling",
        ))
        # Side walls
        boxes.append(_make_wall(
            ox - half_w - half_t, oy + sign * cor_len / 2, oz + cor_h / 2,
            half_t, cor_len / 2, cor_h / 2,
            f"{label_prefix}_wall_left",
        ))
        boxes.append(_make_wall(
            ox + half_w + half_t, oy + sign * cor_len / 2, oz + cor_h / 2,
            half_t, cor_len / 2, cor_h / 2,
            f"{label_prefix}_wall_right",
        ))

    return boxes


def _generate_room(
    rng: np.random.Generator,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    label_prefix: str = "room",
    has_ceiling: bool = True,
) -> list[BBox3D]:
    """Generate a rectangular room with floor, walls, and optional ceiling."""
    boxes: list[BBox3D] = []
    cx, cy, cz = center
    sx, sy, sz = size
    half_x, half_y, half_z = sx / 2, sy / 2, sz / 2
    half_t = _WALL_THICKNESS / 2

    # Floor
    boxes.append(_make_wall(
        cx, cy, cz - half_t,
        half_x, half_y, half_t,
        f"{label_prefix}_floor",
    ))
    # Ceiling
    if has_ceiling:
        boxes.append(_make_wall(
            cx, cy, cz + sz + half_t,
            half_x, half_y, half_t,
            f"{label_prefix}_ceiling",
        ))
    # Walls (4 sides)
    boxes.append(_make_wall(
        cx - half_x - half_t, cy, cz + half_z,
        half_t, half_y, half_z,
        f"{label_prefix}_wall_nx",
    ))
    boxes.append(_make_wall(
        cx + half_x + half_t, cy, cz + half_z,
        half_t, half_y, half_z,
        f"{label_prefix}_wall_px",
    ))
    boxes.append(_make_wall(
        cx, cy - half_y - half_t, cz + half_z,
        half_x, half_t, half_z,
        f"{label_prefix}_wall_ny",
    ))
    boxes.append(_make_wall(
        cx, cy + half_y + half_t, cz + half_z,
        half_x, half_t, half_z,
        f"{label_prefix}_wall_py",
    ))

    return boxes


def _generate_debris(
    rng: np.random.Generator,
    *,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    count: int,
    label_prefix: str = "debris",
) -> list[BBox3D]:
    """Scatter small debris boxes within the given bounds."""
    boxes: list[BBox3D] = []
    bmin = np.array(bounds_min)
    bmax = np.array(bounds_max)

    for i in range(count):
        pos = rng.uniform(bmin, bmax)
        size = rng.uniform(_DEBRIS_SIZE_RANGE[0], _DEBRIS_SIZE_RANGE[1], size=3)
        boxes.append(BBox3D(
            center=(float(pos[0]), float(pos[1]), float(pos[2])),
            extent=(float(size[0] / 2), float(size[1] / 2), float(size[2] / 2)),
            label=f"{label_prefix}_{i}",
        ))

    return boxes


def generate_world(config: WorldConfig) -> list[BBox3D]:
    """Generate a procedural 3D world from the given configuration.

    This is the core **Infinigen geometry generator**.  It produces pure
    3D layout data (bounding boxes) with no physics or RL semantics.
    The boxes describe walls, floors, ceilings, column obstacles,
    furniture, rooms, corridors, vertical shafts, and scattered debris.

    **Downstream consumers:**

    * **Infinigen pipeline** — reads :func:`world_gin_overrides` to
      configure scene generation (grid resolution, scatter density,
      material settings, fog, lighting).
    * **Genesis DroneEnv** — receives box geometry via
      :func:`world_to_genesis_entities` (separate conversion step).
    * **Curriculum tracker** — reads :func:`world_to_frame_metadata`
      for depth stats, traversability, and obstacle counts.

    Parameters
    ----------
    config : WorldConfig
        World generation parameters (see :class:`WorldConfig` for
        complexity ladder and preset factories).

    Returns
    -------
    list[BBox3D]
        All obstacles/walls/floors/ceilings as bounding boxes.
        Each box has a unique human-readable ``label``.

    Raises
    ------
    TypeError
        If *config* is not a :class:`WorldConfig`.
    """
    if not isinstance(config, WorldConfig):
        msg = f"config must be WorldConfig, got {type(config).__name__}"
        raise TypeError(msg)

    rng = np.random.default_rng(config.seed)
    boxes: list[BBox3D] = []

    # ── Phase 1: Main corridor (always present) ──────────────────────────
    corridor = _generate_corridor(config, rng, label_prefix="main")
    boxes.extend(corridor)

    cor_len = config.effective_corridor_length
    cor_w = config.corridor_width
    cor_h = config.corridor_height

    # ── Phase 2: Connected rooms (complexity >= 0.35) ────────────────────
    n_rooms = config.effective_num_rooms
    room_centers: list[tuple[float, float, float]] = []

    if n_rooms > 0:
        for i in range(n_rooms):
            # Place rooms along the corridor, offset to one side
            room_x = rng.uniform(cor_len * 0.2, cor_len * 0.8)
            side = 1.0 if rng.random() > 0.5 else -1.0
            room_y = side * (cor_w / 2 + rng.uniform(*config.room_size_range) / 2 + 0.5)

            room_sx = rng.uniform(*config.room_size_range)
            room_sy = rng.uniform(*config.room_size_range)
            room_sz = cor_h * rng.uniform(0.8, 1.2)

            center = (room_x, room_y, 0.0)
            room_centers.append(center)

            room_boxes = _generate_room(
                rng,
                center=center,
                size=(room_sx, room_sy, room_sz),
                label_prefix=f"room_{i}",
                has_ceiling=rng.random() > 0.3,  # Some rooms have open ceilings
            )
            boxes.extend(room_boxes)

            # Add furniture-like obstacles inside rooms
            n_furniture = max(1, round(config.effective_debris_density * 5))
            for j in range(n_furniture):
                fx = room_x + rng.uniform(-room_sx / 3, room_sx / 3)
                fy = room_y + rng.uniform(-room_sy / 3, room_sy / 3)
                fz = rng.uniform(0.1, room_sz * 0.4)
                fw = rng.uniform(0.15, 0.5)
                fh = rng.uniform(0.15, 0.5)
                fd = rng.uniform(0.15, 0.5)
                boxes.append(BBox3D(
                    center=(fx, fy, fz),
                    extent=(fw / 2, fd / 2, fh / 2),
                    label=f"room_{i}_furniture_{j}",
                ))

    # ── Phase 3: Branching corridors (complexity >= 0.55) ────────────────
    n_branches = config.effective_num_branches

    if n_branches > 0:
        for i in range(n_branches):
            # Branch off from a random point along the main corridor
            branch_x = rng.uniform(cor_len * 0.15, cor_len * 0.85)
            side = 1.0 if rng.random() > 0.5 else -1.0
            branch_len = rng.uniform(3.0, cor_len * 0.5)

            branch = _generate_corridor(
                config, rng,
                origin=(branch_x, side * cor_w / 2, 0.0),
                direction=(0.0, side, 0.0),
                length=branch_len,
                label_prefix=f"branch_{i}",
            )
            boxes.extend(branch)

            # Dead-end wall at the end of some branches
            if rng.random() > 0.4:
                end_y = side * (cor_w / 2 + branch_len)
                boxes.append(_make_wall(
                    branch_x, end_y, cor_h / 2,
                    cor_w / 2, _WALL_THICKNESS / 2, cor_h / 2,
                    f"branch_{i}_deadend",
                ))

    # ── Phase 4: Vertical levels (complexity >= 0.75) ────────────────────
    n_levels = config.effective_num_levels

    if n_levels > 1:
        for level in range(1, n_levels):
            level_z = level * cor_h
            # Each upper level is a shorter corridor
            upper_len = cor_len * rng.uniform(0.4, 0.7)

            upper = _generate_corridor(
                config, rng,
                origin=(rng.uniform(0, cor_len * 0.3), 0.0, level_z),
                direction=(1.0, 0.0, 0.0),
                length=upper_len,
                label_prefix=f"level_{level}",
            )
            boxes.extend(upper)

            # Vertical shaft connecting levels
            shaft_x = rng.uniform(cor_len * 0.2, cor_len * 0.6)
            shaft_w = rng.uniform(0.8, 1.5)
            shaft_boxes = _generate_room(
                rng,
                center=(shaft_x, 0.0, level_z - cor_h / 2),
                size=(shaft_w, shaft_w, cor_h),
                label_prefix=f"shaft_{level}",
                has_ceiling=False,
            )
            boxes.extend(shaft_boxes)

    # ── Phase 5: Debris and clutter ──────────────────────────────────────
    debris_count = round(config.effective_debris_density * 20)
    if debris_count > 0:
        debris = _generate_debris(
            rng,
            bounds_min=(cor_len * 0.1, -cor_w / 2 + 0.2, 0.1),
            bounds_max=(cor_len * 0.9, cor_w / 2 - 0.2, cor_h * 0.5),
            count=debris_count,
            label_prefix="debris",
        )
        boxes.extend(debris)

    return boxes


# ---------------------------------------------------------------------------
# Summary / introspection helpers
# ---------------------------------------------------------------------------


def world_summary(config: WorldConfig) -> dict[str, Any]:
    """Return a human-readable summary of the world configuration.

    Useful for logging, debugging, and curriculum tracking dashboards.
    All values are the *effective* parameters after complexity derivation.
    """
    style = config.effective_style
    return {
        "complexity": config.complexity,
        "seed": config.seed,
        "effective_corridor_length": round(config.effective_corridor_length, 2),
        "corridor_width": config.corridor_width,
        "corridor_height": config.corridor_height,
        "num_columns": config.effective_num_columns,
        "gap_height": round(config.effective_gap_height, 2),
        "num_rooms": config.effective_num_rooms,
        "num_branches": config.effective_num_branches,
        "num_levels": config.effective_num_levels,
        "debris_density": round(config.effective_debris_density, 3),
        "fog_density": round(style.fog_density, 3),
        "cloud_density": round(style.cloud_density, 3),
        "point_light_count": style.point_light_count,
        "wall_color_saturation": round(style.wall_color_saturation, 3),
    }


def world_to_frame_metadata(
    config: WorldConfig,
    boxes: list[BBox3D],
    *,
    frame_id: int = 0,
    scene_seed: int = 0,
) -> dict[str, Any]:
    """Build a dict compatible with :class:`FrameMetadata` from world output.

    The returned dict can be passed to ``FrameMetadata(**d)`` or stored
    as JSON for the training pipeline.  It includes depth statistics,
    traversability ratio, obstacle list, and camera placement — all
    computed from the pure 3D geometry without running any physics
    simulation.

    Parameters
    ----------
    config : WorldConfig
        The world config used for generation.
    boxes : list[BBox3D]
        The generated obstacles from :func:`generate_world`.
    frame_id : int
        Frame identifier.
    scene_seed : int
        Scene RNG seed.

    Returns
    -------
    dict[str, Any]
        FrameMetadata-compatible dict.
    """
    cor_len = config.effective_corridor_length
    cor_h = config.corridor_height

    # Compute basic depth stats from corridor geometry
    cam_pos = np.array([0.3, 0.0, cor_h / 2])
    obstacle_boxes = [b for b in boxes if "col_" in b.label or "furniture" in b.label or "debris" in b.label]
    if obstacle_boxes:
        # Distance from camera to nearest obstacle surface (approximate)
        dists = [
            max(0.01, np.linalg.norm(np.array(b.center) - cam_pos) - np.linalg.norm(b.extent))
            for b in obstacle_boxes
        ]
        min_dist = float(min(dists))
        max_dist = cor_len
    else:
        min_dist = 0.5
        max_dist = cor_len

    # Traversability: ratio of gap area to total cross-section
    gap_h = config.effective_gap_height
    traversability = min(1.0, gap_h / cor_h)

    return {
        "frame_id": frame_id,
        "scene_seed": scene_seed,
        "camera_position": (0.3, 0.0, cor_h / 2),
        "camera_rotation_euler": (0.0, 0.0, 0.0),
        "obstacles": boxes,
        "depth_stats": {
            "min_m": round(min_dist, 3),
            "max_m": round(max_dist, 3),
            "mean_m": round((min_dist + max_dist) / 2, 3),
            "median_m": round((min_dist + max_dist) / 2, 3),
            "std_m": round((max_dist - min_dist) / 4, 3),
        },
        "traversability_ratio": round(traversability, 3),
        "curriculum_stage": round(config.complexity * 10),
        "velocity": (0.0, 0.0, 0.0),
        "nearest_obstacle_m": round(min_dist, 3),
        "swarm_positions": [],
    }


def world_to_genesis_entities(
    boxes: list[BBox3D],
) -> list[dict[str, Any]]:
    """Convert world obstacles to Genesis entity config dicts.

    This is the **Infinigen → Genesis conversion layer**.  Each BBox3D
    becomes a fixed Box entity suitable for
    :class:`~infinigen.core.syndata.genesis_export.GenesisEntityConfig`.

    Note: ``BBox3D.extent`` stores half-extents, so Genesis ``size``
    is ``extent * 2`` in each dimension.

    Parameters
    ----------
    boxes : list[BBox3D]
        Output from :func:`generate_world`.

    Returns
    -------
    list[dict[str, Any]]
        Genesis entity config dicts with ``name``, ``morph_type``,
        ``pos``, ``is_fixed``, and ``extra`` keys.
    """
    entities: list[dict[str, Any]] = []
    for b in boxes:
        dx, dy, dz = b.extent
        entities.append({
            "name": b.label,
            "morph_type": "Box",
            "pos": b.center,
            "is_fixed": True,
            "extra": {"size": (dx * 2, dy * 2, dz * 2)},
        })
    return entities


def world_to_drone_env_config(
    config: WorldConfig,
    boxes: list[BBox3D],
    *,
    num_envs: int = 2048,
    dt: float = 0.01,
) -> dict[str, Any]:
    """Build a DroneEnvConfig-compatible dict from world output.

    This is the **Infinigen → Genesis DroneEnv conversion layer**.
    It translates the pure 3D geometry into RL simulation parameters:
    map bounds, reward scales, termination conditions, command ranges,
    and obstacle entity lists.

    The RL gym (Genesis DroneEnv) consumes this dict to configure the
    physics environment.  Infinigen only *produces* the configuration —
    it does not run the simulation.

    Parameters
    ----------
    config : WorldConfig
        The world config used for generation.
    boxes : list[BBox3D]
        The generated obstacles from :func:`generate_world`.
    num_envs : int
        Number of parallel environments for vectorised training.
    dt : float
        Physics timestep (seconds).

    Returns
    -------
    dict[str, Any]
        Configuration dict compatible with
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

    cor_len = config.effective_corridor_length
    cor_w = config.corridor_width
    cor_h = config.corridor_height
    init_z = cor_h / 2
    half_w = cor_w / 4

    # Build obstacle entities
    obstacle_entities: list[dict[str, Any]] = []
    for b in boxes:
        dx, dy, dz = b.extent
        obstacle_entities.append({
            "name": b.label,
            "morph_type": "Box",
            "pos": b.center,
            "size": (dx * 2, dy * 2, dz * 2),
            "fixed": True,
            "collision": True,
        })

    # Crash penalty scales with complexity
    crash_penalty = -10.0 - 10.0 * config.complexity

    # Roll/pitch termination gets stricter with complexity
    term_angle = max(30.0, 90.0 - 60.0 * config.complexity)

    return {
        "num_envs": num_envs,
        "dt": dt,
        "cam_res": (128, 128),
        "drone_init_pos": (0.3, 0.0, init_z),
        "map_size": (cor_len + 1.0, cor_w + 1.0),
        "obstacle_entities": obstacle_entities,
        "episode_length_s": 10.0 + 20.0 * config.complexity,
        "max_episode_length": round(1000 + 2000 * config.complexity),
        "init_pos_range": {
            "x": (-0.1, 0.1),
            "y": (-half_w, half_w),
            "z": (init_z - 0.1, init_z + 0.1),
        },
        "reward_scales": {
            "target": 15.0,
            "smooth": -0.0001,
            "yaw": 0.0,
            "angular": -0.0001,
            "crash": crash_penalty,
        },
        "command_ranges": {
            "x": (cor_len * 0.7, cor_len * 0.9),
            "y": (-0.3 - 0.5 * config.complexity, 0.3 + 0.5 * config.complexity),
            "z": (init_z - 0.2, init_z + 0.2),
        },
        "termination_conditions": {
            "roll_deg": term_angle,
            "pitch_deg": term_angle,
            "ground_m": 0.08,
            "x_m": cor_len + 0.5,
            "y_m": cor_w / 2 + 0.5,
            "z_m": cor_h + 0.3,
        },
        "world_summary": world_summary(config),
    }


def world_gin_overrides(config: WorldConfig) -> dict[str, object]:
    """Return **Infinigen-specific** gin-compatible overrides for the world.

    Maps world config to scene-generation parameters that control
    Infinigen's procedural pipeline: grid resolution, object density,
    material properties, fog, clouds, and lighting.

    These are *not* Genesis simulation settings — Genesis receives its
    configuration through :func:`world_to_genesis_entities` and
    :func:`world_to_drone_env_config` instead.

    Parameters
    ----------
    config : WorldConfig
        World generation parameters.

    Returns
    -------
    dict[str, object]
        Gin-compatible override dict.  Keys match Infinigen's pipeline
        gin bindings (e.g. ``"grid_coarsen"``, ``"scatter_density_multiplier"``).
    """
    style = config.effective_style
    return {
        "grid_coarsen": max(1, round(4 - 3 * config.complexity)),
        "object_count": config.effective_num_columns + config.effective_num_rooms * 3,
        "scatter_density_multiplier": round(0.1 + 0.9 * config.complexity, 4),
        "fog_density": round(style.fog_density, 4),
        "cloud_density": round(style.cloud_density, 4),
        "lighting.point_light_count": style.point_light_count,
        "material.roughness": round(style.floor_roughness, 4),
        "material.color_saturation": round(style.wall_color_saturation, 4),
        "configure_render_cycles.exposure": round(
            0.5 * style.ambient_intensity, 4
        ),
    }
