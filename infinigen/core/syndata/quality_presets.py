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

__all__ = [
    "VALID_PRESETS",
    "drone_preset",
    "to_gin_bindings",
]

# Precise type for gin override values — covers all value types produced
# by _make_preset and consumed by to_gin_bindings.
GinValue = int | float | str | bool | tuple[int, int]

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

def _make_preset(
    width: int,
    height: int,
    num_samples: int,
    min_samples: int,
    adaptive_threshold: float,
    time_limit: int,
    denoise: bool,
    exposure: float,
    motion_blur: bool,
    volume_scatter: bool,
) -> dict[str, GinValue]:
    """Build a preset dict with resolution *and* camera-intrinsics sync.

    Infinigen's ``get_sensor_coords`` has its own ``H`` / ``W`` gin
    bindings that **must** match ``execute_tasks.generate_resolution``.
    Failing to sync them silently produces wrong depth / projection
    data — fatal for any vision-based RL agent.
    """
    return {
        "configure_render_cycles.num_samples": num_samples,
        "configure_render_cycles.min_samples": min_samples,
        "configure_render_cycles.adaptive_threshold": adaptive_threshold,
        "configure_render_cycles.time_limit": time_limit,
        "configure_render_cycles.denoise": denoise,
        "configure_render_cycles.exposure": exposure,
        "execute_tasks.generate_resolution": (width, height),
        # Camera intrinsics must match render resolution
        "get_sensor_coords.H": height,
        "get_sensor_coords.W": width,
        "configure_blender.motion_blur": motion_blur,
        "render.volume_scatter": volume_scatter,
    }


_PRESETS: dict[str, dict[str, GinValue]] = {
    "preview": _make_preset(
        width=128, height=128, num_samples=16, min_samples=4,
        adaptive_threshold=0.1, time_limit=5,
        denoise=False, exposure=0.8,
        motion_blur=False, volume_scatter=False,
    ),
    "fast": _make_preset(
        width=256, height=256, num_samples=64, min_samples=8,
        adaptive_threshold=0.05, time_limit=15,
        denoise=True, exposure=1.0,
        motion_blur=False, volume_scatter=False,
    ),
    "medium": _make_preset(
        width=512, height=512, num_samples=128, min_samples=16,
        adaptive_threshold=0.02, time_limit=30,
        denoise=True, exposure=1.0,
        motion_blur=True, volume_scatter=True,
    ),
    "high": _make_preset(
        width=1024, height=1024, num_samples=512, min_samples=32,
        adaptive_threshold=0.005, time_limit=120,
        denoise=True, exposure=1.0,
        motion_blur=True, volume_scatter=True,
    ),
}

VALID_PRESETS = frozenset(_PRESETS)


def drone_preset(
    name: str,
    *,
    resolution_override: tuple[int, int] | None = None,
) -> dict[str, GinValue]:
    """Return Gin-compatible overrides for the named quality preset.

    Parameters
    ----------
    name : str
        One of ``"preview"``, ``"fast"``, ``"medium"``, ``"high"``.
    resolution_override : tuple[int, int] | None
        If given, replaces the preset's default resolution.

    Returns
    -------
    dict[str, GinValue]
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
        w, h = resolution_override
        if w < 32 or h < 32:
            msg = f"resolution too small: {resolution_override} (min 32×32)"
            raise ValueError(msg)
        if w > 8192 or h > 8192:
            msg = f"resolution too large: {resolution_override} (max 8192×8192)"
            raise ValueError(msg)
        overrides["execute_tasks.generate_resolution"] = resolution_override
        overrides["get_sensor_coords.H"] = h
        overrides["get_sensor_coords.W"] = w
    return overrides


def to_gin_bindings(overrides: dict[str, object]) -> list[str]:
    """Convert an overrides dict to a list of gin-parseable binding strings.

    Each returned string is a valid gin binding such as
    ``"configure_render_cycles.num_samples = 64"``.  Strings are quoted,
    tuples become parenthesised, booleans become ``True``/``False``.

    Parameters
    ----------
    overrides : dict[str, object]
        Dict of ``"scope/function.param"`` → value pairs, as returned by
        :func:`drone_preset`, :meth:`DomainRandomiser.gin_overrides`, etc.

    Returns
    -------
    list[str]
        Gin binding strings ready for ``gin.parse_config``.
    """
    lines: list[str] = []
    for key, value in sorted(overrides.items()):
        lines.append(f"{key} = {_gin_repr(value)}")
    return lines


def _gin_repr(value: object) -> str:
    """Format a Python value as a gin-compatible literal."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, tuple):
        inner = ", ".join(_gin_repr(v) for v in value)
        return f"({inner})"
    return str(value)
