# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Parametric 3D world generator for progressive curriculum pre-training.

Generates pure 3D geometry (axis-aligned bounding boxes) describing
training environments of increasing navigational complexity.  Output
is ``list[BBox3D]`` — pure geometry, no physics or RL types.

:class:`InfinigenOverlayHints` describes which Infinigen asset categories
(vegetation, furniture, vehicles, weather) and render quality (texture
resolution, material complexity) to activate at each complexity level.

Curriculum progression
----------------------

A single ``complexity`` parameter in [0, 1] drives the entire progression
from trivial flight corridors to dense, photorealistic 3D environments.
The goal is to start **extremely simple** so RL agents can learn basic
navigation policies, then **gradually increase** realism and difficulty.

===========  =========  =============================================================
complexity   Preset     Environment
===========  =========  =============================================================
0.00 – 0.15  ``flappy``   **Basic 3D flight**: straight corridor with column-gap
                          obstacles (~10 BBox3D boxes for walls/floor/ceiling +
                          column obstacles).  Agent learns: up/down, left/right,
                          forward/back.  Flat-shaded, uniform lighting, 256px
                          textures.
0.15 – 0.35  ``corridor`` **Textured corridor**: same topology but with visual
                          variety (~20 boxes).  Coloured PBR walls, varied floor
                          roughness, subtle fog.  Tests vision robustness.
0.35 – 0.55  ``rooms``    **Indoor rooms**: connected rooms with doors, furniture-
                          like clutter (~40–60 boxes).  Requires planning turns,
                          entering/exiting spaces.  Multi-light setup.
0.55 – 0.75  ``branches`` **Mixed environment**: branching corridors, T-junctions,
                          dead-ends, fog, vegetation, dynamic objects like swinging
                          doors and tree branches (~80–120 boxes).
0.75 – 0.90  ``maze``     **Streets / forests / indoor**: multi-level vertical maze
                          with vehicles, pedestrians, HDR environment lighting,
                          full PBR materials (~150–200 boxes).  2048px textures.
0.90 – 1.00  ``doom``     **Full photorealism**: dense 3D maze with all Infinigen
                          assets — subsurface materials, weather particles, moving
                          branches, opening doors, 4096px textures (~250–350 boxes).
===========  =========  =============================================================

At ``complexity=1.0``, Infinigen overlays the coarse box layout with its
full asset library: streets, forests, indoor furniture, moving tree
branches, opening doors, vehicles, pedestrians, and weather effects —
with extensive domain randomisation of lighting, fog, wind, and materials.

Box counts are approximate and vary with the RNG seed.  "~10 boxes" at
``complexity=0.05`` means the corridor shell (floor, ceiling, 2 walls)
plus a handful of column obstacles — the simplest possible 3D environment
for curriculum learning.

Output format
-------------
:func:`generate_world` always returns a ``list[BBox3D]`` — axis-aligned
bounding boxes representing walls, floors, ceilings, columns, furniture,
and debris.  Each ``BBox3D`` has a human-readable label (e.g.
``"main_col_lower_3"``, ``"room_2_furniture_1"``) for debugging and
selective filtering.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from infinigen.core.syndata.metadata import BBox3D

logger = logging.getLogger(__name__)

__all__ = [
    "COMPLEXITY_BRANCHES",
    "COMPLEXITY_CORRIDOR",
    "COMPLEXITY_DOOM",
    "COMPLEXITY_MAZE",
    "COMPLEXITY_ROOMS",
    "InfinigenOverlayHints",
    "VisualStyle",
    "WorldConfig",
    "generate_world",
    "overlay_hints_for_complexity",
    "world_gin_overrides",
    "world_summary",
    "world_to_frame_metadata",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_GAP: float = 0.3  # Minimum flyable gap (metres)
_WALL_THICKNESS: float = 0.1  # Wall / floor / ceiling thickness
_DEBRIS_SIZE_RANGE: tuple[float, float] = (0.05, 0.3)

# Label substrings used to identify obstacle boxes in world_to_frame_metadata.
_OBSTACLE_LABEL_TAGS: frozenset[str] = frozenset({"col_", "furniture", "debris"})

# Complexity thresholds defining curriculum stage boundaries.
# These are shared between InfinigenOverlayHints.from_complexity() and
# WorldConfig effective value derivation.  Adjust here to reshape the
# progression globally.
COMPLEXITY_CORRIDOR: float = 0.15   # c < this → flat corridor
COMPLEXITY_ROOMS: float = 0.35      # c < this → textured corridor
COMPLEXITY_BRANCHES: float = 0.55   # c < this → indoor rooms
COMPLEXITY_MAZE: float = 0.75       # c < this → branching corridors
COMPLEXITY_DOOM: float = 0.90       # c < this → multi-level maze
# c >= COMPLEXITY_DOOM → full photorealism

# ---------------------------------------------------------------------------
# Infinigen overlay hints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfinigenOverlayHints:
    """Hints describing what Infinigen should render at a given complexity.

    The box layout from :func:`generate_world` is the **coarse 3D skeleton**.
    Infinigen overlays it with photorealistic assets — the *kind* of assets
    depends on the curriculum stage.  This dataclass communicates those
    intentions to the Infinigen pipeline (or to a human configuring gin).

    These hints are purely informational.  They do **not** run Blender or
    generate any assets themselves.  They describe *what category* of
    Infinigen assets should be used, letting the pipeline (or a curriculum
    controller) select the right gin configs.

    The ``enabled_*`` booleans indicate which asset categories to activate.
    ``texture_resolution`` and ``subdiv_level`` control render quality.

    Parameters
    ----------
    stage_description : str
        Human-readable description of the curriculum stage.
    environment_type : str
        One of ``"corridor"``, ``"indoor"``, ``"outdoor_street"``,
        ``"outdoor_forest"``, ``"mixed"``.
    texture_resolution : int
        Target texture resolution (pixels, power-of-2).
    subdiv_level : int
        Mesh subdivision level (0 = coarse, 3 = very fine).
    enabled_vegetation : bool
        If True, scatter vegetation (trees, bushes, grass).
    enabled_furniture : bool
        If True, place indoor furniture assets.
    enabled_vehicles : bool
        If True, add parked/moving vehicles.
    enabled_dynamic_objects : bool
        If True, add moving elements (swinging doors, tree branches,
        wind-blown debris).
    enabled_weather : bool
        If True, add weather effects (rain, snow, dust particles).
    enabled_pedestrians : bool
        If True, add pedestrian/human assets.
    material_complexity : str
        One of ``"flat"``, ``"basic_pbr"``, ``"full_pbr"``,
        ``"subsurface"`` — controls shader complexity.
    lighting_complexity : str
        One of ``"uniform"``, ``"single_sun"``, ``"multi_light"``,
        ``"hdr_environment"`` — controls lighting setup.
    """

    stage_description: str = "flat corridor"
    environment_type: str = "corridor"
    texture_resolution: int = 256
    subdiv_level: int = 0
    enabled_vegetation: bool = False
    enabled_furniture: bool = False
    enabled_vehicles: bool = False
    enabled_dynamic_objects: bool = False
    enabled_weather: bool = False
    enabled_pedestrians: bool = False
    material_complexity: str = "flat"
    lighting_complexity: str = "uniform"

    _VALID_ENV_TYPES: ClassVar[frozenset[str]] = frozenset({
        "corridor", "indoor", "outdoor_street", "outdoor_forest", "mixed",
    })
    _VALID_MATERIALS: ClassVar[frozenset[str]] = frozenset({
        "flat", "basic_pbr", "full_pbr", "subsurface",
    })
    _VALID_LIGHTING: ClassVar[frozenset[str]] = frozenset({
        "uniform", "single_sun", "multi_light", "hdr_environment",
    })

    def __post_init__(self) -> None:
        if self.environment_type not in self._VALID_ENV_TYPES:
            msg = (
                f"environment_type must be one of {self._VALID_ENV_TYPES}, "
                f"got {self.environment_type!r}"
            )
            raise ValueError(msg)
        if self.material_complexity not in self._VALID_MATERIALS:
            msg = (
                f"material_complexity must be one of {self._VALID_MATERIALS}, "
                f"got {self.material_complexity!r}"
            )
            raise ValueError(msg)
        if self.lighting_complexity not in self._VALID_LIGHTING:
            msg = (
                f"lighting_complexity must be one of {self._VALID_LIGHTING}, "
                f"got {self.lighting_complexity!r}"
            )
            raise ValueError(msg)
        if self.texture_resolution < 1:
            msg = f"texture_resolution must be >= 1, got {self.texture_resolution}"
            raise ValueError(msg)
        if self.subdiv_level < 0:
            msg = f"subdiv_level must be >= 0, got {self.subdiv_level}"
            raise ValueError(msg)

    @staticmethod
    def from_complexity(complexity: float) -> InfinigenOverlayHints:
        """Derive overlay hints from a complexity value in [0, 1].

        These hints tell the Infinigen pipeline which categories of
        photorealistic assets to activate at this curriculum stage.
        Early stages use flat shading for fast RL bootstrapping;
        later stages enable full Infinigen asset variety.

        Progression:

        * **c < 0.15**: Flat-shaded corridor.  No assets.  For learning
          basic obstacle avoidance (up/down, left/right, forward/back).
        * **c 0.15–0.35**: Basic PBR materials, single directional light.
          Coloured walls, varied floor textures.
        * **c 0.35–0.55**: Indoor environment with furniture, doors.
          Multi-light setup.  Teaches room-to-room navigation.
        * **c 0.55–0.75**: Mixed indoor/outdoor.  Vegetation (potted
          plants → trees), dynamic objects (swinging doors, moving tree
          branches), fog, basic weather.
        * **c 0.75–0.90**: Outdoor streets/forest.  Vehicles, pedestrians,
          full PBR materials, HDR environment lighting.  2048px textures.
        * **c 0.90–1.00**: Full Infinigen photorealism — subsurface
          materials, dense vegetation, weather particles, opening doors,
          vehicles, moving branches.  4096px textures.
        """
        c = max(0.0, min(1.0, complexity))

        if c < COMPLEXITY_CORRIDOR:
            return InfinigenOverlayHints(
                stage_description="Flat corridor — basic avoidance training",
                environment_type="corridor",
                texture_resolution=256,
                subdiv_level=0,
                material_complexity="flat",
                lighting_complexity="uniform",
            )
        if c < COMPLEXITY_ROOMS:
            return InfinigenOverlayHints(
                stage_description="Textured corridor — visual robustness",
                environment_type="corridor",
                texture_resolution=512,
                subdiv_level=1,
                material_complexity="basic_pbr",
                lighting_complexity="single_sun",
            )
        if c < COMPLEXITY_BRANCHES:
            return InfinigenOverlayHints(
                stage_description="Indoor rooms — furniture, doors, multi-light",
                environment_type="indoor",
                texture_resolution=1024,
                subdiv_level=1,
                enabled_furniture=True,
                material_complexity="basic_pbr",
                lighting_complexity="multi_light",
            )
        if c < COMPLEXITY_MAZE:
            return InfinigenOverlayHints(
                stage_description="Mixed indoor/outdoor — vegetation, dynamic objects, fog",
                environment_type="mixed",
                texture_resolution=1024,
                subdiv_level=2,
                enabled_vegetation=True,
                enabled_furniture=True,
                enabled_dynamic_objects=True,
                enabled_weather=True,
                material_complexity="full_pbr",
                lighting_complexity="multi_light",
            )
        if c < COMPLEXITY_DOOM:
            return InfinigenOverlayHints(
                stage_description="Outdoor streets/forest — vehicles, HDR, high-res",
                environment_type="outdoor_street",
                texture_resolution=2048,
                subdiv_level=2,
                enabled_vegetation=True,
                enabled_furniture=True,
                enabled_vehicles=True,
                enabled_dynamic_objects=True,
                enabled_weather=True,
                enabled_pedestrians=True,
                material_complexity="full_pbr",
                lighting_complexity="hdr_environment",
            )
        return InfinigenOverlayHints(
            stage_description="Full photorealism — all Infinigen assets, max quality",
            environment_type="mixed",
            texture_resolution=4096,
            subdiv_level=3,
            enabled_vegetation=True,
            enabled_furniture=True,
            enabled_vehicles=True,
            enabled_dynamic_objects=True,
            enabled_weather=True,
            enabled_pedestrians=True,
            material_complexity="subsurface",
            lighting_complexity="hdr_environment",
        )

    def to_gin_hints(self) -> dict[str, object]:
        """Return Infinigen gin-compatible hints for the overlay.

        These are suggestions for the Infinigen pipeline — the actual
        gin bindings depend on the specific scene recipe.
        """
        return {
            "environment_type": self.environment_type,
            "texture_resolution": self.texture_resolution,
            "subdiv_level": self.subdiv_level,
            "enable_vegetation": self.enabled_vegetation,
            "enable_furniture": self.enabled_furniture,
            "enable_vehicles": self.enabled_vehicles,
            "enable_dynamic_objects": self.enabled_dynamic_objects,
            "enable_weather": self.enabled_weather,
            "enable_pedestrians": self.enabled_pedestrians,
            "material_complexity": self.material_complexity,
            "lighting_complexity": self.lighting_complexity,
        }


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
# Derived effective values (typed container replacing dict[str, Any])
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EffectiveValues:
    """Typed container for complexity-derived parameters.

    Replaces the previous ``dict[str, Any]`` for type safety and
    faster attribute access via ``__slots__``.
    """

    num_columns: int
    gap_height: float
    num_rooms: int
    num_branches: int
    num_levels: int
    debris_density: float
    corridor_length: float
    style: VisualStyle


# ---------------------------------------------------------------------------
# Main world config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldConfig:
    """Configuration for a procedurally generated 3D training world.

    A single ``complexity`` parameter in [0, 1] drives the entire
    progression from a trivial flight corridor to a dense 3D maze.
    Individual parameters can be overridden for fine-grained control.

    **Box counts per preset** (approximate, seed-dependent):

    * ``flappy()``   (c≈0.05): ~10 boxes — corridor shell (floor, ceiling,
      2 walls) + a few column obstacles.  Simplest 3D environment.
    * ``corridor()`` (c≈0.25): ~20 boxes — more columns, tighter gaps,
      coloured walls.
    * ``rooms()``    (c≈0.45): ~40–60 — rooms with doors + furniture clutter.
    * ``branches()`` (c≈0.65): ~80–120 — branch corridors + dead-ends + fog.
    * ``maze()``     (c≈0.85): ~150–200 — multiple vertical levels + shafts.
    * ``doom()``     (c≈0.98): ~250–350 — dense maze with heavy debris.

    **Separation of concerns**: this config describes **3D geometry only**
    (Infinigen's job).  The RL simulation parameters (rewards, episode
    length, termination, drone dynamics) belong to the separate
    orchestration project that consumes this package's outputs.

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

    # Derived values (populated by __post_init__, never None after construction)
    _effective: _EffectiveValues = field(init=False, repr=False)

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

        eff = _EffectiveValues(
            # Columns: 2 at c=0, up to 15 at c=1
            num_columns=(
                self.num_columns if self.num_columns is not None
                else max(2, round(2 + 13 * c))
            ),
            # Gap height: 2.0m at c=0 (easy), 0.5m at c=1 (hard)
            gap_height=(
                self.gap_height if self.gap_height is not None
                else max(_MIN_GAP, 2.0 - 1.5 * c)
            ),
            # Rooms emerge at c >= COMPLEXITY_ROOMS
            num_rooms=(
                self.num_rooms if self.num_rooms is not None
                else max(0, round(8 * max(0, c - COMPLEXITY_ROOMS) / (1.0 - COMPLEXITY_ROOMS)))
            ),
            # Branches emerge at c >= COMPLEXITY_BRANCHES
            num_branches=(
                self.num_branches if self.num_branches is not None
                else max(0, round(6 * max(0, c - COMPLEXITY_BRANCHES) / (1.0 - COMPLEXITY_BRANCHES)))
            ),
            # Vertical levels at c >= COMPLEXITY_MAZE
            num_levels=(
                self.num_levels if self.num_levels is not None
                else max(1, round(1 + 3 * max(0, c - COMPLEXITY_MAZE) / (1.0 - COMPLEXITY_MAZE)))
            ),
            # Debris density emerges at c >= COMPLEXITY_CORRIDOR
            debris_density=(
                self.debris_density if self.debris_density is not None
                else min(1.0, max(0.0, (c - COMPLEXITY_CORRIDOR) / (1.0 - COMPLEXITY_CORRIDOR)))
            ),
            # Corridor length scales with complexity
            corridor_length=self.corridor_length * (1.0 + 1.5 * c),
            # Visual style
            style=(
                self.style if self.style is not None
                else VisualStyle(
                    wall_color_hue=0.0,  # will be randomised per-wall
                    wall_color_saturation=min(1.0, c * 1.5),
                    floor_roughness=0.3 + 0.5 * c,
                    ambient_intensity=max(0.3, 1.0 - 0.5 * c),
                    fog_density=max(0.0, (c - 0.5) * 1.2),
                    cloud_density=max(0.0, (c - 0.4) * 1.0),
                    point_light_count=max(0, round(c * 8)),
                )
            ),
        )

        object.__setattr__(self, "_effective", eff)

    # ---- Effective values ---------------------------------------------------

    @property
    def effective_num_columns(self) -> int:
        """Number of column obstacles after applying complexity."""
        return self._effective.num_columns

    @property
    def effective_gap_height(self) -> float:
        """Flyable gap height (metres) after applying complexity."""
        return self._effective.gap_height

    @property
    def effective_num_rooms(self) -> int:
        """Number of connected rooms after applying complexity."""
        return self._effective.num_rooms

    @property
    def effective_num_branches(self) -> int:
        """Number of maze branches / dead-ends after applying complexity."""
        return self._effective.num_branches

    @property
    def effective_num_levels(self) -> int:
        """Number of vertical levels after applying complexity."""
        return self._effective.num_levels

    @property
    def effective_debris_density(self) -> float:
        """Debris density in [0, 1] after applying complexity."""
        return self._effective.debris_density

    @property
    def effective_corridor_length(self) -> float:
        """Corridor length (metres) after applying complexity scaling."""
        return self._effective.corridor_length

    @property
    def effective_style(self) -> VisualStyle:
        """Visual style after applying complexity defaults."""
        return self._effective.style

    @property
    def overlay_hints(self) -> InfinigenOverlayHints:
        """Infinigen asset overlay hints derived from complexity.

        Describes which categories of Infinigen assets (vegetation,
        furniture, vehicles, weather, etc.) and what render quality
        (texture resolution, subdivision level, material complexity)
        should be used at this complexity level.
        """
        return InfinigenOverlayHints.from_complexity(self.complexity)

    def __repr__(self) -> str:
        """Concise representation with complexity and key effective values."""
        hints = self.overlay_hints
        return (
            f"WorldConfig(complexity={self.complexity}, seed={self.seed}, "
            f"cols={self.effective_num_columns}, rooms={self.effective_num_rooms}, "
            f"branches={self.effective_num_branches}, levels={self.effective_num_levels}, "
            f"env={hints.environment_type!r})"
        )

    # ---- Preset factories ---------------------------------------------------

    @staticmethod
    def flappy(*, seed: int | None = None) -> WorldConfig:
        """Preset: trivial flight corridor for basic 3D navigation.

        Produces a straight corridor with a few column-gap obstacles —
        the simplest possible 3D environment for curriculum learning.
        The agent learns elemental avoidance: fly up/down to dodge
        columns, stay centred left/right, move forward.

        Output: ~10 BBox3D boxes (floor, ceiling, 2 walls + columns).
        Infinigen overlay: flat shading, uniform lighting, 256px textures.
        """
        return WorldConfig(complexity=0.05, seed=seed)

    @staticmethod
    def corridor(*, seed: int | None = None) -> WorldConfig:
        """Preset: textured corridor with more obstacles and visual variety.

        Same topology as ``flappy`` but with narrower gaps, more columns,
        coloured PBR walls, and subtle fog.  Tests that vision-based
        policies are robust to surface appearance changes.

        Output: ~20 BBox3D boxes.
        Infinigen overlay: basic PBR materials, single sun, 512px textures.
        """
        return WorldConfig(complexity=0.25, seed=seed)

    @staticmethod
    def rooms(*, seed: int | None = None) -> WorldConfig:
        """Preset: connected rooms with doors and furniture-like clutter.

        Introduces right-angle turns, doorway transitions, and indoor
        obstacles.  The agent must plan room entries/exits and navigate
        around furniture.

        Output: ~40–60 BBox3D boxes (rooms + furniture).
        Infinigen overlay: furniture assets, multi-light setup, 1024px.
        """
        return WorldConfig(complexity=0.45, seed=seed)

    @staticmethod
    def branches(*, seed: int | None = None) -> WorldConfig:
        """Preset: branching corridors with T-junctions and dead-ends.

        Mixed indoor/outdoor environment with vegetation, dynamic objects
        (swinging doors, moving tree branches), fog, and debris.  Requires
        exploration and backtracking.

        Output: ~80–120 BBox3D boxes.
        Infinigen overlay: vegetation, dynamic objects, fog, full PBR.
        """
        return WorldConfig(complexity=0.65, seed=seed)

    @staticmethod
    def maze(*, seed: int | None = None) -> WorldConfig:
        """Preset: multi-level 3D maze with vertical shafts.

        Full 3D navigation: altitude changes through vertical shafts,
        upper corridors, dense obstacle fields.  Approaching outdoor
        streets/forest complexity with vehicles, pedestrians, and HDR
        environment lighting.

        Output: ~150–200 BBox3D boxes.
        Infinigen overlay: vehicles, pedestrians, HDR, 2048px textures.
        """
        return WorldConfig(complexity=0.85, seed=seed)

    @staticmethod
    def doom(*, seed: int | None = None) -> WorldConfig:
        """Preset: dense Doom-like 3D maze — maximum complexity.

        Many rooms, branching corridors, multiple vertical levels, heavy
        fog, point lights, and dense debris.  At this level, Infinigen
        overlays the full asset library: photorealistic walls, vegetation,
        furniture, moving tree branches, opening doors, vehicles,
        pedestrians, weather particles, and subsurface materials.

        Output: ~250–350 BBox3D boxes.
        Infinigen overlay: all assets, subsurface materials, 4096px.
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
        This is the recommended entry-point for automatic curriculum
        scheduling.

        Typical progression (approximate, seed-dependent)::

            progress=0.00 → c≈0.00  trivial corridor     (~10 boxes, flat)
            progress=0.05 → c≈0.22  textured corridor     (~16 boxes, PBR)
            progress=0.20 → c≈0.45  rooms with furniture  (~50 boxes)
            progress=0.50 → c≈0.71  branching + fog       (~100 boxes)
            progress=0.80 → c≈0.89  multi-level maze      (~180 boxes)
            progress=1.00 → c≈1.00  full photorealism     (~300 boxes)

        The output ``list[BBox3D]`` from :func:`generate_world` describes
        pure 3D geometry for Infinigen.  Downstream consumers (Genesis
        World, GenesisDroneEnv) convert these via separate bridge projects.
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
    """Create a wall/surface BBox3D centered at ``(x, y, z)`` with half-extents ``(dx, dy, dz)`` per axis."""
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

    Produces floor, ceiling, two side walls, and column-gap obstacles
    along the corridor axis.  Supports both X-aligned and Y-aligned
    directions for branching corridors.

    Parameters
    ----------
    config : WorldConfig
        World parameters (corridor dimensions, column count, gap height).
    rng : np.random.Generator
        Random number generator for obstacle placement.
    origin : tuple[float, float, float]
        Starting position of the corridor.
    direction : tuple[float, float, float]
        Axis-aligned direction vector (X or Y).
    length : float | None
        Override corridor length.  If *None*, uses ``config.effective_corridor_length``.
    label_prefix : str
        Prefix for box labels (e.g. ``"main"``, ``"branch_0"``).
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
    """Generate a rectangular room with floor, four walls, and optional ceiling.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator (reserved for future furniture randomisation).
    center : tuple[float, float, float]
        Room centre ``(x, y, z)`` in world coordinates.
    size : tuple[float, float, float]
        Full room dimensions ``(width, depth, height)``.
    label_prefix : str
        Label prefix for box identification (e.g. ``"room_0"``).
    has_ceiling : bool
        If *False*, the room is open-topped (visible from above).
    """
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
    """Scatter small debris boxes randomly within an axis-aligned bounding region.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for position and size sampling.
    bounds_min, bounds_max : tuple[float, float, float]
        Axis-aligned bounding region for debris placement.
    count : int
        Number of debris boxes to generate.
    label_prefix : str
        Label prefix for identification (e.g. ``"debris"``).
    """
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

    **Infinigen geometry generator** — produces pure 3D layout data
    (bounding boxes) with no physics, RL, or Genesis semantics.

    The boxes describe walls, floors, ceilings, column obstacles,
    furniture, rooms, corridors, vertical shafts, and scattered debris.
    Box counts scale with complexity: ~10 at c=0.05, ~300 at c=0.98.

    **Downstream consumers** (all separate projects):

    * **Infinigen pipeline**: reads :func:`world_gin_overrides` to
      configure scene generation (textures, materials, lighting, fog).
    * **Genesis World / GenesisDroneEnv**: consume the ``list[BBox3D]``
      output via separate bridge projects for physics simulation and
      RL environment setup.

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

    logger.debug("generate_world: complexity=%.2f → %d boxes", config.complexity, len(boxes))
    return boxes


# ---------------------------------------------------------------------------
# Summary / introspection helpers
# ---------------------------------------------------------------------------


def overlay_hints_for_complexity(complexity: float) -> InfinigenOverlayHints:
    """Return Infinigen overlay hints for a given complexity value.

    Convenience wrapper around :meth:`InfinigenOverlayHints.from_complexity`
    for use without constructing a full :class:`WorldConfig`.

    Parameters
    ----------
    complexity : float
        Complexity value in [0, 1].

    Returns
    -------
    InfinigenOverlayHints
        Hints describing which Infinigen asset categories and render
        quality to use at the given complexity level.

    Examples
    --------
    >>> hints = overlay_hints_for_complexity(0.5)
    >>> hints.environment_type
    'indoor'
    >>> hints.enabled_furniture
    True
    """
    return InfinigenOverlayHints.from_complexity(complexity)


def world_summary(config: WorldConfig) -> dict[str, Any]:
    """Return a human-readable summary of the world configuration.

    Useful for logging, debugging, and curriculum tracking dashboards.
    All values are the *effective* parameters after complexity derivation.
    """
    style = config.effective_style
    hints = config.overlay_hints
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
        "overlay": {
            "stage_description": hints.stage_description,
            "environment_type": hints.environment_type,
            "texture_resolution": hints.texture_resolution,
            "subdiv_level": hints.subdiv_level,
            "material_complexity": hints.material_complexity,
            "lighting_complexity": hints.lighting_complexity,
            "enabled_vegetation": hints.enabled_vegetation,
            "enabled_furniture": hints.enabled_furniture,
            "enabled_vehicles": hints.enabled_vehicles,
            "enabled_dynamic_objects": hints.enabled_dynamic_objects,
            "enabled_weather": hints.enabled_weather,
            "enabled_pedestrians": hints.enabled_pedestrians,
        },
    }


def world_to_frame_metadata(
    config: WorldConfig,
    boxes: list[BBox3D],
    *,
    frame_id: int = 0,
    scene_seed: int = 0,
) -> dict[str, Any]:
    """Build a FrameMetadata-compatible dict from world geometry.

    Computes depth statistics, traversability ratio, obstacle list, and
    camera placement from the pure 3D geometry — no physics simulation
    needed.  Useful for pre-flight validation, curriculum tracking, and
    metadata logging.

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

    # Filter obstacle boxes using module-level label tag set
    cam_pos = np.array([0.3, 0.0, cor_h / 2])
    obstacle_boxes = [
        b for b in boxes
        if any(tag in b.label for tag in _OBSTACLE_LABEL_TAGS)
    ]
    if obstacle_boxes:
        # Vectorized distance computation — batch all centers/extents
        centers = np.array([b.center for b in obstacle_boxes])
        extents = np.array([b.extent for b in obstacle_boxes])
        center_dists = np.linalg.norm(centers - cam_pos, axis=1)
        extent_norms = np.linalg.norm(extents, axis=1)
        surface_dists = np.maximum(0.01, center_dists - extent_norms)
        min_dist = float(surface_dists.min())
        max_dist = cor_len
    else:
        min_dist = 0.5
        max_dist = cor_len

    # Traversability: ratio of gap area to total cross-section
    gap_h = config.effective_gap_height
    traversability = min(1.0, gap_h / cor_h)

    logger.debug("world_to_frame_metadata: %d obstacle boxes, min_dist=%.3f", len(obstacle_boxes), min_dist)
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


def world_gin_overrides(config: WorldConfig) -> dict[str, object]:
    """Return Infinigen gin-compatible overrides for the world.

    Maps world config to scene-generation parameters that control
    Infinigen's procedural pipeline: grid resolution, object density,
    material properties, fog, clouds, and lighting.

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
    hints = config.overlay_hints
    overrides: dict[str, object] = {
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
    # Merge overlay hints so the Infinigen pipeline knows which asset
    # categories and render quality to use at this complexity level.
    overrides.update(hints.to_gin_hints())
    return overrides
