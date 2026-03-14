# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Quality presets optimised for drone-swarm RL training.

Each preset returns a dict of Gin-compatible overrides that control Cycles
render settings (samples, denoising, resolution) and scene-level knobs.
The presets are deliberately conservative on the *preview* / *fast* end so
that a single GPU can produce hundreds of training frames per minute.

No ``bpy`` import — everything is a plain dict.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "preview": {
        "configure_render_cycles.num_samples": 16,
        "configure_render_cycles.min_samples": 4,
        "configure_render_cycles.adaptive_threshold": 0.1,
        "configure_render_cycles.time_limit": 5,
        "configure_render_cycles.denoise": False,
        "execute_tasks.generate_resolution": (128, 128),
        "render.motion_blur": False,
        "render.volume_scatter": False,
    },
    "fast": {
        "configure_render_cycles.num_samples": 64,
        "configure_render_cycles.min_samples": 8,
        "configure_render_cycles.adaptive_threshold": 0.05,
        "configure_render_cycles.time_limit": 15,
        "configure_render_cycles.denoise": True,
        "execute_tasks.generate_resolution": (256, 256),
        "render.motion_blur": False,
        "render.volume_scatter": False,
    },
    "medium": {
        "configure_render_cycles.num_samples": 128,
        "configure_render_cycles.min_samples": 16,
        "configure_render_cycles.adaptive_threshold": 0.02,
        "configure_render_cycles.time_limit": 30,
        "configure_render_cycles.denoise": True,
        "execute_tasks.generate_resolution": (512, 512),
        "render.motion_blur": True,
        "render.volume_scatter": True,
    },
    "high": {
        "configure_render_cycles.num_samples": 512,
        "configure_render_cycles.min_samples": 32,
        "configure_render_cycles.adaptive_threshold": 0.005,
        "configure_render_cycles.time_limit": 120,
        "configure_render_cycles.denoise": True,
        "execute_tasks.generate_resolution": (1024, 1024),
        "render.motion_blur": True,
        "render.volume_scatter": True,
    },
}

VALID_PRESETS = frozenset(_PRESETS)


def drone_preset(
    name: str,
    *,
    resolution_override: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return Gin-compatible overrides for the named quality preset.

    Parameters
    ----------
    name : str
        One of ``"preview"``, ``"fast"``, ``"medium"``, ``"high"``.
    resolution_override : tuple[int, int] | None
        If given, replaces the preset's default resolution.

    Returns
    -------
    dict[str, Any]
        Keys are Gin scope/function.param strings; values are Python literals.

    Raises
    ------
    ValueError
        If *name* is not a recognised preset.
    """
    if name not in _PRESETS:
        msg = f"Unknown preset {name!r}; choose from {sorted(_PRESETS)}"
        raise ValueError(msg)

    overrides = dict(_PRESETS[name])
    if resolution_override is not None:
        overrides["execute_tasks.generate_resolution"] = resolution_override
    return overrides
