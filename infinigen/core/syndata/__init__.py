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

from infinigen.core.syndata.complexity import CurriculumConfig, curriculum_overrides
from infinigen.core.syndata.density_scaling import DensityScaler
from infinigen.core.syndata.metadata import FrameMetadata
from infinigen.core.syndata.metrics import SceneBudget
from infinigen.core.syndata.parallel_stages import StageGraph
from infinigen.core.syndata.quality_presets import drone_preset
from infinigen.core.syndata.randomisation import DomainRandomiser
from infinigen.core.syndata.resolution import resolution_for_stage
from infinigen.core.syndata.validation import SceneValidator

__all__ = [
    "CurriculumConfig",
    "DensityScaler",
    "DomainRandomiser",
    "FrameMetadata",
    "SceneBudget",
    "SceneValidator",
    "StageGraph",
    "curriculum_overrides",
    "drone_preset",
    "resolution_for_stage",
]
