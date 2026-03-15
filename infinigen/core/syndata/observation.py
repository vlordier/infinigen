# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Observation-space configuration for the Infinigen rendering pipeline.

Defines which Blender render passes to include in the agent's observation
and what post-processing (sensor noise) is applied.  Maps directly to
Infinigen's ``render_image`` gin bindings.

All helpers are pure Python — no ``bpy`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Render passes available in Infinigen (via Cycles)
# ---------------------------------------------------------------------------

#: Passes that produce the visual RGB observation.
PASS_RGB = "Image"

#: Geometric / metric passes useful for RL state estimation.
PASS_DEPTH = "z"
PASS_NORMAL = "normal"
PASS_FLOW = "vector"  # optical flow (forward + backward)

#: Semantic passes for object/instance awareness.
PASS_OBJECT_INDEX = "object_index"
PASS_MATERIAL_INDEX = "material_index"

#: Lighting decomposition passes (useful for domain-randomisation analysis).
PASS_DIFFUSE_DIRECT = "diffuse_direct"
PASS_DIFFUSE_COLOR = "diffuse_color"

#: Passes available via the ``flat/render_image`` path (cheap to render).
PASSES_FLAT_AVAILABLE: frozenset[str] = frozenset({
    PASS_DEPTH, PASS_NORMAL, PASS_FLOW, PASS_OBJECT_INDEX,
})
PASSES_MINIMAL: frozenset[str] = frozenset({PASS_DEPTH, PASS_OBJECT_INDEX})
PASSES_NAVIGATION: frozenset[str] = frozenset({PASS_DEPTH, PASS_NORMAL, PASS_OBJECT_INDEX})
PASSES_FULL: frozenset[str] = frozenset({
    PASS_DEPTH, PASS_NORMAL, PASS_FLOW,
    PASS_OBJECT_INDEX, PASS_MATERIAL_INDEX,
    PASS_DIFFUSE_DIRECT, PASS_DIFFUSE_COLOR,
})


@dataclass(frozen=True)
class SensorNoiseModel:
    """Simple parametric sensor noise for sim-to-real transfer.

    Applied *after* rendering as post-processing on the RGB observation.

    Parameters
    ----------
    gaussian_std : float
        Standard deviation of additive Gaussian noise (0 = none).
        Typical drone camera: 0.01–0.03 (normalised to [0, 1] image).
    salt_pepper_prob : float
        Probability of a salt-and-pepper pixel (0 = none).
    motion_blur_px : float
        Simulated motion blur kernel size in pixels (0 = disabled).
        Already handled by Cycles when ``motion_blur=True`` in presets,
        but this field lets you add *extra* post-render blur for fast
        drone manoeuvres that exceed the shutter window.
    exposure_jitter : float
        Random exposure offset in stops (0 = none).
    """

    gaussian_std: float = 0.0
    salt_pepper_prob: float = 0.0
    motion_blur_px: float = 0.0
    exposure_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.gaussian_std < 0:
            msg = f"gaussian_std must be non-negative, got {self.gaussian_std}"
            raise ValueError(msg)
        if not 0.0 <= self.salt_pepper_prob <= 1.0:
            msg = f"salt_pepper_prob must be in [0, 1], got {self.salt_pepper_prob}"
            raise ValueError(msg)
        if self.motion_blur_px < 0:
            msg = f"motion_blur_px must be non-negative, got {self.motion_blur_px}"
            raise ValueError(msg)

    @staticmethod
    def drone_default(difficulty: float = 0.5) -> SensorNoiseModel:
        """Noise model scaled by curriculum difficulty.

        At ``difficulty=0`` the sensor is near-perfect; at ``difficulty=1``
        noise approaches real-world drone camera levels.
        """
        d = max(0.0, min(1.0, difficulty))
        return SensorNoiseModel(
            gaussian_std=0.03 * d,
            salt_pepper_prob=0.001 * d,
            motion_blur_px=2.0 * d,
            exposure_jitter=0.5 * d,
        )


@dataclass(frozen=True)
class ObservationConfig:
    """What the RL agent observes from each rendered frame.

    Parameters
    ----------
    passes : frozenset[str]
        Set of Blender render passes to include (use the ``PASS_*``
        constants or the ``PASSES_*`` preset sets).
    include_rgb : bool
        Whether to include the composited RGB image.
    depth_clip_m : float
        Maximum depth in metres before clamping to ``inf``.
        Real drone depth sensors clip at ~30–100 m.
    noise : SensorNoiseModel
        Post-render sensor noise parameters.
    """

    passes: frozenset[str] = PASSES_NAVIGATION
    include_rgb: bool = True
    depth_clip_m: float = 100.0
    noise: SensorNoiseModel = field(default_factory=SensorNoiseModel)

    def __post_init__(self) -> None:
        if self.depth_clip_m <= 0:
            msg = f"depth_clip_m must be positive, got {self.depth_clip_m}"
            raise ValueError(msg)

    @property
    def channel_names(self) -> list[str]:
        """Ordered list of data channels the agent receives."""
        channels: list[str] = []
        if self.include_rgb:
            channels.extend(["R", "G", "B"])
        for p in sorted(self.passes):
            channels.append(p)
        return channels

    @property
    def num_channels(self) -> int:
        """Total number of observation channels."""
        return len(self.channel_names)

    def gin_overrides(self) -> dict[str, Any]:
        """Return gin-compatible overrides for render-pass selection.

        Maps to Infinigen's ``full/render_image`` and ``flat/render_image``
        gin configuration patterns.
        """
        overrides: dict[str, Any] = {}
        # Flat passes (always enabled for RL — cheap to render)
        flat_passes = sorted(self.passes & PASSES_FLAT_AVAILABLE)
        if flat_passes:
            overrides["flat/render_image.passes_to_save"] = tuple(flat_passes)
        return overrides
