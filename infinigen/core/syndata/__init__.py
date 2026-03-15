# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Synthetic data utilities for curriculum learning of RL agents.

This bpy-free package provides configuration, scheduling, and validation
tools that sit *outside* the Blender render loop.  Every module can be
imported and tested without ``bpy`` so that CI stays fast and portable.

The public API is organised into **five concern groups**, cleanly
separating what Infinigen owns from what Genesis/DroneEnv owns:

1. **Infinigen pipeline** — scene generation config, gin bindings,
   resource budgeting, domain randomisation.  No dependency on Genesis
   or any RL framework.

2. **Procedural world generation** — :class:`WorldConfig` +
   :func:`generate_world` produce pure 3D geometry (``list[BBox3D]``)
   with progressive complexity: from trivial "3D Flappy Bird" corridors
   to dense multi-level "Doom-like" mazes.  This is **Infinigen's** job:
   the 3D world and assets.  Genesis then imports these for physics
   simulation.

3. **Simple pre-training** — :class:`FlappyColumnConfig` for an even
   simpler bootstrapping environment (superseded by :class:`WorldConfig`
   at ``complexity=0.05``).

4. **Genesis World bridge** — converters mapping Infinigen config/assets
   to Genesis entity/camera/light types and generating runnable scripts.
   This is the **export interface** from Infinigen to Genesis.

5. **GenesisDroneEnv bridge** — bidirectional data exchange between
   Infinigen (scene export) and GenesisDroneEnv (training feedback →
   curriculum adjustment).  Curriculum *control* logic lives in a
   separate project — this package only provides data contracts.

Typical usage
-------------
>>> from infinigen.core.syndata import WorldConfig, generate_world
>>> cfg = WorldConfig.from_curriculum_progress(0.3, seed=42)
>>> boxes = generate_world(cfg)  # pure 3D geometry for Infinigen
>>> # Separately, convert to Genesis entities for RL simulation:
>>> from infinigen.core.syndata import world_to_genesis_entities
>>> genesis_ents = world_to_genesis_entities(boxes)
"""

from infinigen.core.syndata.camera_config import CameraRigConfig, DroneCamera
from infinigen.core.syndata.complexity import CurriculumConfig, curriculum_overrides
from infinigen.core.syndata.density_scaling import DensityScaler
from infinigen.core.syndata.drone_env_bridge import (
    DroneEnvConfig,
    TrainingOutcome,
    apply_curriculum_adjustment,
    outcome_to_curriculum_params,
    scene_to_drone_entities,
    syndata_to_drone_env_config,
)
from infinigen.core.syndata.episode import EpisodeConfig
from infinigen.core.syndata.genesis_export import (
    GenesisCamera,
    GenesisEntityConfig,
    GenesisEpisodeConfig,
    GenesisLight,
    GenesisObservationConfig,
    GenesisSceneConfig,
    GenesisSceneManifest,
    build_genesis_config,
    episode_to_genesis,
    observation_to_genesis,
    to_genesis_script,
)
from infinigen.core.syndata.metadata import FrameMetadata
from infinigen.core.syndata.metrics import SceneBudget
from infinigen.core.syndata.observation import ObservationConfig, SensorNoiseModel
from infinigen.core.syndata.parallel_stages import StageGraph
from infinigen.core.syndata.pretraining import (
    FlappyColumnConfig,
    FlappyObstacle,
    flappy_drone_env_config,
    flappy_genesis_entities,
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
    world_to_drone_env_config,
    world_to_frame_metadata,
    world_to_genesis_entities,
)

__all__ = [
    # ── 1. Infinigen pipeline — scene generation config & gin bindings ──
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
    #    WorldConfig → generate_world() → list[BBox3D]
    #    Pure 3D layout, no Genesis/RL types.
    "InfinigenOverlayHints",
    "VisualStyle",
    "WorldConfig",
    "generate_world",
    "world_gin_overrides",
    "world_summary",
    "world_to_drone_env_config",
    "world_to_frame_metadata",
    "world_to_genesis_entities",
    # ── 3. Simple pre-training (flappy-bird corridor) ──
    "FlappyColumnConfig",
    "FlappyObstacle",
    "flappy_drone_env_config",
    "flappy_genesis_entities",
    "generate_flappy_obstacles",
    # ── 4. Genesis World bridge (Infinigen → Genesis conversion) ──
    "GenesisCamera",
    "GenesisEntityConfig",
    "GenesisEpisodeConfig",
    "GenesisLight",
    "GenesisObservationConfig",
    "GenesisSceneConfig",
    "GenesisSceneManifest",
    "build_genesis_config",
    "episode_to_genesis",
    "observation_to_genesis",
    "to_genesis_script",
    # ── 5. GenesisDroneEnv bridge (bidirectional data exchange) ──
    "DroneEnvConfig",
    "TrainingOutcome",
    "apply_curriculum_adjustment",
    "outcome_to_curriculum_params",
    "scene_to_drone_entities",
    "syndata_to_drone_env_config",
]
