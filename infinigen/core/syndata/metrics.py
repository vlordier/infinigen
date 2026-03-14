# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Scene-level budget estimation and tracking.

Provides :class:`SceneBudget` to pre-flight-check whether a planned scene is
likely to fit within render-time and memory budgets *before* committing to
expensive Blender geometry nodes or Cycles rendering.

All methods are pure Python / NumPy — no ``bpy`` dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SceneBudget:
    """Estimate resource usage for a planned scene.

    The heuristics intentionally over-estimate so that a scene that *passes*
    budget checks will almost certainly render within the target envelope.

    Parameters
    ----------
    poly_count : int
        Total triangle count across all meshes.
    texture_pixels : int
        Sum of width×height across all unique texture images.
    num_lights : int
        Number of emissive objects / area lights.
    num_samples : int
        Cycles sample count.
    resolution : tuple[int, int]
        Render resolution (width, height).
    """

    poly_count: int = 0
    texture_pixels: int = 0
    num_lights: int = 1
    num_samples: int = 64
    resolution: tuple[int, int] = (256, 256)

    # ------ heuristic constants (empirically tuned) --------------------------
    _BYTES_PER_TRI: int = 120  # Cycles BVH ≈ 120 B/tri
    _BYTES_PER_TEX_PX: int = 16  # RGBA float32
    _MS_PER_SAMPLE_PER_MPIX: float = 0.35  # rough GPU throughput

    @property
    def estimated_vram_mb(self) -> float:
        """Conservative VRAM estimate in MiB."""
        geo = self.poly_count * self._BYTES_PER_TRI
        tex = self.texture_pixels * self._BYTES_PER_TEX_PX
        fb_pixels = self.resolution[0] * self.resolution[1]
        framebuffer = fb_pixels * 4 * 4  # 4 channels, float32
        return (geo + tex + framebuffer) / (1024 * 1024)

    @property
    def estimated_render_seconds(self) -> float:
        """Rough render-time estimate in seconds (single GPU)."""
        mpix = (self.resolution[0] * self.resolution[1]) / 1e6
        base = self.num_samples * mpix * self._MS_PER_SAMPLE_PER_MPIX
        # Light complexity scales sub-linearly
        light_factor = 1.0 + 0.1 * math.log2(max(self.num_lights, 1))
        return base * light_factor / 1000.0

    def fits(self, *, max_vram_mb: float = 4096, max_seconds: float = 60) -> bool:
        """Return *True* if the scene fits the given resource envelope."""
        return (
            self.estimated_vram_mb <= max_vram_mb
            and self.estimated_render_seconds <= max_seconds
        )

    def summary(self) -> dict[str, float]:
        """Return a JSON-serialisable summary dict."""
        return {
            "poly_count": self.poly_count,
            "texture_pixels": self.texture_pixels,
            "estimated_vram_mb": round(self.estimated_vram_mb, 1),
            "estimated_render_seconds": round(self.estimated_render_seconds, 2),
        }
