# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Domain-randomisation controls for RL training.

:class:`DomainRandomiser` centralises the ranges for lighting, weather,
material perturbation, and camera jitter.  Each parameter range can be
tightened (early curriculum stages) or widened (later stages) via a single
``difficulty`` knob in [0, 1].

All helpers are pure Python / NumPy — no ``bpy`` dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _Range:
    """A single randomisation range that grows with difficulty."""

    lo_easy: float
    hi_easy: float
    lo_hard: float
    hi_hard: float

    def at(self, difficulty: float) -> tuple[float, float]:
        """Return (lo, hi) interpolated at *difficulty* ∈ [0, 1]."""
        d = max(0.0, min(1.0, difficulty))
        lo = self.lo_easy + d * (self.lo_hard - self.lo_easy)
        hi = self.hi_easy + d * (self.hi_hard - self.hi_easy)
        return (lo, hi)


@dataclass
class DomainRandomiser:
    """Centralised domain-randomisation parameter source.

    Parameters
    ----------
    difficulty : float
        Difficulty in [0, 1].  0 = tight ranges (easy), 1 = wide ranges (hard).
    seed : int | None
        Optional RNG seed for reproducibility.

    The default ranges are tuned for outdoor drone-swarm scenarios.
    Override individual ``_Range`` fields for indoor or specialised scenes.
    """

    difficulty: float = 0.5
    seed: int | None = None

    # ---- light & exposure ---------------------------------------------------
    sun_elevation: _Range = field(
        default_factory=lambda: _Range(lo_easy=30, hi_easy=60, lo_hard=5, hi_hard=85)
    )
    sun_intensity: _Range = field(
        default_factory=lambda: _Range(lo_easy=0.8, hi_easy=1.2, lo_hard=0.3, hi_hard=2.5)
    )
    exposure: _Range = field(
        default_factory=lambda: _Range(lo_easy=-0.5, hi_easy=0.5, lo_hard=-2.0, hi_hard=2.0)
    )

    # ---- weather / atmosphere -----------------------------------------------
    cloud_density: _Range = field(
        default_factory=lambda: _Range(lo_easy=0.0, hi_easy=0.3, lo_hard=0.0, hi_hard=0.9)
    )
    fog_density: _Range = field(
        default_factory=lambda: _Range(lo_easy=0.0, hi_easy=0.0, lo_hard=0.0, hi_hard=0.5)
    )
    wind_speed: _Range = field(
        default_factory=lambda: _Range(lo_easy=0.0, hi_easy=2.0, lo_hard=0.0, hi_hard=15.0)
    )

    # ---- camera jitter (simulates drone vibration) --------------------------
    camera_rotation_jitter_deg: _Range = field(
        default_factory=lambda: _Range(lo_easy=0.0, hi_easy=1.0, lo_hard=0.0, hi_hard=5.0)
    )
    camera_translation_jitter_m: _Range = field(
        default_factory=lambda: _Range(lo_easy=0.0, hi_easy=0.02, lo_hard=0.0, hi_hard=0.1)
    )

    # ---- material perturbation ----------------------------------------------
    material_roughness_offset: _Range = field(
        default_factory=lambda: _Range(lo_easy=-0.05, hi_easy=0.05, lo_hard=-0.3, hi_hard=0.3)
    )
    material_color_hue_shift: _Range = field(
        default_factory=lambda: _Range(lo_easy=-5.0, hi_easy=5.0, lo_hard=-30.0, hi_hard=30.0)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            msg = "difficulty must be in [0.0, 1.0]"
            raise ValueError(msg)

    # ---- convenience accessors ----------------------------------------------

    def ranges(self) -> dict[str, tuple[float, float]]:
        """Return all randomisation ranges at the current difficulty."""
        result: dict[str, tuple[float, float]] = {}
        for name in (
            "sun_elevation",
            "sun_intensity",
            "exposure",
            "cloud_density",
            "fog_density",
            "wind_speed",
            "camera_rotation_jitter_deg",
            "camera_translation_jitter_m",
            "material_roughness_offset",
            "material_color_hue_shift",
        ):
            r: _Range = getattr(self, name)
            result[name] = r.at(self.difficulty)
        return result

    def gin_overrides(self) -> dict[str, object]:
        """Return Gin-compatible overrides derived from current ranges."""
        r = self.ranges()
        sun_lo, sun_hi = r["sun_elevation"]
        return {
            "lighting.sun_elevation_range": (sun_lo, sun_hi),
            "camera.rotation_jitter": r["camera_rotation_jitter_deg"][1],
            "camera.translation_jitter": r["camera_translation_jitter_m"][1],
        }

    @staticmethod
    def from_curriculum_progress(progress: float, *, seed: int | None = None) -> DomainRandomiser:
        """Create a randomiser whose difficulty is proportional to progress.

        This is a convenience constructor for coupling with
        :class:`~infinigen.core.syndata.complexity.CurriculumConfig`.

        Uses a gentle sqrt curve so that some randomisation is present
        even at very early stages.
        """
        difficulty = math.sqrt(max(0.0, min(1.0, progress)))
        return DomainRandomiser(difficulty=difficulty, seed=seed)
