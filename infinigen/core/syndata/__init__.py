# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Synthetic data utilities for curriculum learning of RL agents.

This bpy-free package provides configuration, scheduling, and validation
tools that sit *outside* the Blender render loop.  Every module can be
imported and tested without ``bpy`` so that CI stays fast and portable.

Typical usage
-------------
>>> from infinigen.core.syndata import complexity, quality_presets
>>> cfg = complexity.CurriculumConfig(stage=3, total_stages=10)
>>> overrides = quality_presets.drone_preset("fast")
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
from infinigen.core.syndata.quality_presets import drone_preset, to_gin_bindings
from infinigen.core.syndata.randomisation import DomainRandomiser
from infinigen.core.syndata.resolution import resolution_for_stage
from infinigen.core.syndata.validation import SceneValidator

__all__ = [
    "CameraRigConfig",
    "CurriculumConfig",
    "DensityScaler",
    "DomainRandomiser",
    "DroneCamera",
    "DroneEnvConfig",
    "EpisodeConfig",
    "FrameMetadata",
    "GenesisCamera",
    "GenesisEntityConfig",
    "GenesisEpisodeConfig",
    "GenesisLight",
    "GenesisObservationConfig",
    "GenesisSceneConfig",
    "GenesisSceneManifest",
    "ObservationConfig",
    "SceneBudget",
    "SceneValidator",
    "SensorNoiseModel",
    "StageGraph",
    "TrainingOutcome",
    "apply_curriculum_adjustment",
    "build_genesis_config",
    "curriculum_overrides",
    "drone_preset",
    "episode_to_genesis",
    "observation_to_genesis",
    "outcome_to_curriculum_params",
    "resolution_for_stage",
    "scene_to_drone_entities",
    "syndata_to_drone_env_config",
    "to_genesis_script",
    "to_gin_bindings",
]
