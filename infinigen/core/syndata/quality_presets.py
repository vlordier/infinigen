# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Adaptive quality presets for render speed vs. fidelity trade-off.

During RL training it is often desirable to iterate quickly with low-fidelity
renders and only switch to full quality for evaluation or final data capture.
This module defines named quality tiers and the logic to derive gin overrides
from them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class QualityPreset(StrEnum):
    """Named quality tiers controlling render fidelity and speed."""

    PREVIEW = "preview"
    FAST = "fast"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass(frozen=True)
class QualityConfig:
    """Immutable render configuration for a quality tier.

    Parameters
    ----------
    num_samples : int
        Cycles sample count.
    resolution_x : int
        Horizontal output resolution.
    resolution_y : int
        Vertical output resolution.
    adaptive_threshold : float
        Cycles adaptive sampling noise threshold (lower = cleaner).
    use_denoising : bool
        Whether to enable the denoiser.
    time_limit : int
        Maximum render time per frame in seconds (0 = unlimited).
    motion_blur : bool
        Whether to enable motion blur.
    volume_step_rate : float
        Step rate for volumetric rendering.
    """

    num_samples: int = 64
    resolution_x: int = 320
    resolution_y: int = 180
    adaptive_threshold: float = 0.1
    use_denoising: bool = False
    time_limit: int = 30
    motion_blur: bool = False
    volume_step_rate: float = 1.0


_QUALITY_PRESETS: dict[QualityPreset, QualityConfig] = {
    QualityPreset.PREVIEW: QualityConfig(
        num_samples=16,
        resolution_x=320,
        resolution_y=180,
        adaptive_threshold=0.2,
        use_denoising=False,
        time_limit=10,
        motion_blur=False,
        volume_step_rate=2.0,
    ),
    QualityPreset.FAST: QualityConfig(
        num_samples=64,
        resolution_x=640,
        resolution_y=360,
        adaptive_threshold=0.1,
        use_denoising=True,
        time_limit=30,
        motion_blur=False,
        volume_step_rate=1.0,
    ),
    QualityPreset.MEDIUM: QualityConfig(
        num_samples=512,
        resolution_x=1280,
        resolution_y=720,
        adaptive_threshold=0.05,
        use_denoising=True,
        time_limit=120,
        motion_blur=False,
        volume_step_rate=0.5,
    ),
    QualityPreset.HIGH: QualityConfig(
        num_samples=2048,
        resolution_x=1920,
        resolution_y=1080,
        adaptive_threshold=0.02,
        use_denoising=True,
        time_limit=300,
        motion_blur=True,
        volume_step_rate=0.2,
    ),
    QualityPreset.ULTRA: QualityConfig(
        num_samples=8192,
        resolution_x=3840,
        resolution_y=2160,
        adaptive_threshold=0.005,
        use_denoising=True,
        time_limit=0,  # unlimited
        motion_blur=True,
        volume_step_rate=0.1,
    ),
}


def get_quality_config(preset: str | QualityPreset) -> QualityConfig:
    """Return the :class:`QualityConfig` for the named *preset*.

    Parameters
    ----------
    preset : str or QualityPreset
        One of ``"preview"``, ``"fast"``, ``"medium"``, ``"high"``, ``"ultra"``.

    Returns
    -------
    QualityConfig

    Raises
    ------
    ValueError
        If *preset* is not recognised.
    """
    if isinstance(preset, str):
        try:
            preset = QualityPreset(preset.lower())
        except ValueError:
            valid = [p.value for p in QualityPreset]
            raise ValueError(
                f"Unknown quality preset '{preset}'. Valid presets: {valid}"
            ) from None
    return _QUALITY_PRESETS[preset]


def quality_gin_overrides(preset: str | QualityPreset) -> list[str]:
    """Return gin override strings that implement the given quality *preset*.

    These can be fed directly to ``gin.parse_config``.
    """
    cfg = get_quality_config(preset)
    return [
        f"configure_render_cycles.num_samples = {cfg.num_samples}",
        f"configure_render_cycles.adaptive_threshold = {cfg.adaptive_threshold}",
        f"configure_render_cycles.denoise = {cfg.use_denoising}",
        f"configure_render_cycles.time_limit = {cfg.time_limit}",
        f"execute_tasks.generate_resolution = ({cfg.resolution_x}, {cfg.resolution_y})",
        f"configure_blender.motion_blur = {cfg.motion_blur}",
    ]
