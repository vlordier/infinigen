# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Synthetic data utilities for curriculum learning of RL drone swarms.

This bpy-free package provides **data contracts and converters** between
three independent systems:

**Three-system architecture**::

    ┌──────────────────────────────────────────────────────────────────┐
    │  INFINIGEN  (this repo)                                         │
    │  Parametric 3D world + asset generation                         │
    │  • Geometry: corridors → rooms → streets → forests → indoor     │
    │  • Assets:   flat colour → PBR textures → 4K photorealistic     │
    │  • Output:   list[BBox3D] + gin overrides + OBJ/PLY/MJCF/URDF  │
    └──────────────────────┬───────────────────────────────────────────┘
                           │  world_to_genesis_entities()
                           │  build_genesis_config()
                           │  to_genesis_script()
                           ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  GENESIS WORLD  (genesis-world.readthedocs.io)                  │
    │  Physics engine + differentiable renderer                       │
    │  • Imports Infinigen meshes via gs.morphs.Mesh/MJCF/URDF/USD   │
    │  • Runs physics:  scene.step(dt), collision, rigid-body dynamics│
    │  • Renders:       cam.render(rgb, depth, segmentation, normal)  │
    │  • Episode loop:  scene.step() × N, vectorised env resets       │
    └──────────────────────┬───────────────────────────────────────────┘
                           │  DroneEnvConfig / TrainingOutcome
                           ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  GENESIS DRONE ENV  (github.com/KafuuChikai/GenesisDroneEnv)    │
    │  RL gym for drone swarm training                                │
    │  • Gym interface:  obs, reward, done, info = env.step(action)   │
    │  • Drone control:  attitude, velocity, position commands        │
    │  • Reward shaping: collision penalty, target tracking, smooth   │
    │  • Curriculum:     outcome_to_curriculum_params() → advance/    │
    │                    hold/regress (controller lives elsewhere)     │
    └──────────────────────────────────────────────────────────────────┘

**What this package does:**

- Generates parametric 3D worlds with progressive complexity (Infinigen)
- Converts Infinigen geometry/assets to Genesis scene configs (bridge)
- Exchanges structured data with GenesisDroneEnv (bridge)

**What this package does NOT do:**

- It does not run Blender (bpy-free)
- It does not run Genesis physics (genesis-free)
- It does not implement curriculum control logic (separate project)
- It does not train RL agents

The public API is organised into five concern groups:

1. **Infinigen pipeline** — scene generation config, gin bindings,
   resource budgeting, domain randomisation.

2. **Procedural world generation** — :class:`WorldConfig` +
   :func:`generate_world` produce ``list[BBox3D]`` layouts with
   progressive complexity.  :class:`InfinigenOverlayHints` describes
   which Infinigen assets (vegetation, furniture, vehicles, weather)
   to overlay at each complexity level.

3. **Simple pre-training** — :class:`FlappyColumnConfig` for ultra-simple
   corridor environments before full world generation kicks in.

4. **Genesis World bridge** — converters mapping Infinigen config/assets
   to Genesis entity/camera/light types and generating runnable scripts.

5. **GenesisDroneEnv bridge** — bidirectional data exchange between
   Infinigen (scene export) and GenesisDroneEnv (training feedback).

Typical usage
-------------
>>> from infinigen.core.syndata import WorldConfig, generate_world
>>> cfg = WorldConfig.from_curriculum_progress(0.3, seed=42)
>>> boxes = generate_world(cfg)  # pure 3D geometry (Infinigen side)
>>>
>>> # Convert to Genesis entities (bridge layer):
>>> from infinigen.core.syndata import world_to_genesis_entities
>>> genesis_ents = world_to_genesis_entities(boxes)
>>>
>>> # Or build a full Genesis scene config:
>>> from infinigen.core.syndata import build_genesis_config, to_genesis_script
>>> cfg = build_genesis_config(frame_metadata=..., drone_camera=...)
>>> script = to_genesis_script(cfg)  # runnable Python script
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
    #    These configure Infinigen's Blender-based rendering pipeline.
    #    No Genesis or RL dependency.
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
    "world_to_drone_env_config",
    "world_to_frame_metadata",
    "world_to_genesis_entities",
    # ── 3. Simple pre-training (ultra-simple corridor) ──
    #    Even simpler than WorldConfig(complexity=0) — a straight corridor
    #    with column-gap obstacles for bootstrapping RL agents.
    "FlappyColumnConfig",
    "FlappyObstacle",
    "flappy_drone_env_config",
    "flappy_genesis_entities",
    "generate_flappy_obstacles",
    # ── 4. Genesis World bridge (Infinigen → Genesis conversion) ──
    #    Converts Infinigen types to Genesis entity/camera/light configs.
    #    Generates runnable Python scripts that `import genesis`.
    #    See: https://genesis-world.readthedocs.io/
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
    #    Infinigen → DroneEnv: scene export as YAML-compatible configs.
    #    DroneEnv → Infinigen: TrainingOutcome → curriculum hints.
    #    See: https://github.com/KafuuChikai/GenesisDroneEnv
    "DroneEnvConfig",
    "TrainingOutcome",
    "apply_curriculum_adjustment",
    "outcome_to_curriculum_params",
    "scene_to_drone_entities",
    "syndata_to_drone_env_config",
]
