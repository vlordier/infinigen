# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Parametric 3D world generation for curriculum learning.

This bpy-free package controls Infinigen's procedural generation pipeline
to produce 3D environments of **incrementally increasing complexity** —
from trivial flight corridors to dense photorealistic worlds.

**Infinigen's role:**
    Generate parametric 3D worlds and assets (geometry, textures, lighting,
    materials).  Output: ``list[BBox3D]`` layouts, Gin override dicts, and
    exported meshes (OBJ/PLY/MJCF/URDF).

**What this package does NOT do:**
    - It does not run Blender (bpy-free, fast CI)
    - It does not run physics simulation
    - It does not implement RL gym environments
    - It does not implement curriculum control logic

The public API is organised into four concern groups:

1. **Scene generation config** — :class:`CurriculumConfig`,
   :func:`drone_preset`, :class:`SceneBudget`, :func:`resolution_for_stage`,
   :class:`DomainRandomiser`, density/stage/metadata/validation configs,
   :func:`to_gin_bindings`.

2. **Procedural world generation** — :class:`WorldConfig` with a single
   ``complexity`` knob [0, 1] drives :func:`generate_world` →
   ``list[BBox3D]`` layouts.  :class:`InfinigenOverlayHints` describes
   which Infinigen assets to overlay at each complexity level.

3. **Simple pre-training geometry** — :class:`FlappyColumnConfig` for
   ultra-simple corridor environments (stage 0 bootstrapping).

4. **Rendering pipeline config** — :class:`DroneCamera`,
   :class:`CameraRigConfig`, :class:`EpisodeConfig`,
   :class:`ObservationConfig` for Infinigen's Blender rendering.

Typical usage
-------------
>>> from infinigen.core.syndata import WorldConfig, generate_world
>>> cfg = WorldConfig.from_curriculum_progress(0.3, seed=42)
>>> boxes = generate_world(cfg)  # pure 3D geometry
>>> overrides = world_gin_overrides(cfg)  # Infinigen pipeline params
"""

from infinigen.core.syndata.camera_config import CameraRigConfig, DroneCamera
from infinigen.core.syndata.complexity import CurriculumConfig, curriculum_overrides
from infinigen.core.syndata.density_scaling import DensityScaler
from infinigen.core.syndata.episode import EpisodeConfig
from infinigen.core.syndata.metadata import FrameMetadata
from infinigen.core.syndata.metrics import SceneBudget
from infinigen.core.syndata.observation import ObservationConfig, SensorNoiseModel
from infinigen.core.syndata.parallel_stages import StageGraph
from infinigen.core.syndata.pretraining import (
    FlappyColumnConfig,
    FlappyObstacle,
    generate_flappy_obstacles,
)
from infinigen.core.syndata.quality_presets import drone_preset, to_gin_bindings
from infinigen.core.syndata.randomisation import DomainRandomiser
from infinigen.core.syndata.resolution import resolution_for_stage
from infinigen.core.syndata.validation import SceneValidator
from infinigen.core.syndata.world_gen import (
    InfinigenOverlayHints,
    VisualStyle,
    WorldConfig,
    generate_world,
    world_gin_overrides,
    world_summary,
    world_to_frame_metadata,
)

__all__ = [
    # ── 1. Scene generation config & gin bindings ──
    #    Configure Infinigen's Blender-based rendering pipeline.
    "CameraRigConfig",
    "CurriculumConfig",
    "DensityScaler",
    "DomainRandomiser",
    "DroneCamera",
    "EpisodeConfig",
    "FrameMetadata",
    "ObservationConfig",
    "SceneBudget",
    "SceneValidator",
    "SensorNoiseModel",
    "StageGraph",
    "curriculum_overrides",
    "drone_preset",
    "resolution_for_stage",
    "to_gin_bindings",
    # ── 2. Procedural world generation (Infinigen geometry) ──
    #    WorldConfig(complexity=[0,1]) → generate_world() → list[BBox3D]
    #    Pure 3D layout: corridors → rooms → streets → forests → indoor.
    #    InfinigenOverlayHints describes which asset categories to activate.
    "InfinigenOverlayHints",
    "VisualStyle",
    "WorldConfig",
    "generate_world",
    "world_gin_overrides",
    "world_summary",
    "world_to_frame_metadata",
    # ── 3. Simple pre-training geometry (ultra-simple corridor) ──
    #    Even simpler than WorldConfig(complexity=0) — a straight corridor
    #    with column-gap obstacles for bootstrapping RL agents.
    "FlappyColumnConfig",
    "FlappyObstacle",
    "generate_flappy_obstacles",
]
