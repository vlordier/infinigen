# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Structured metadata export for generated scenes.

Provides a rich, JSON-serialisable record of everything that was generated
in a scene, suitable for:
  - RL agent observation augmentation (object relationships, spatial stats).
  - Dataset filtering and balancing (e.g. "give me all scenes with > 5 trees").
  - Provenance tracking (which seeds / configs produced a given output).

The metadata is designed to be written alongside rendered images and can be
loaded independently of Blender.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ObjectRecord:
    """Metadata for a single placed object in the scene.

    Parameters
    ----------
    name : str
        Object name (e.g. ``"TreeFactory_0"``).
    category : str
        Semantic category (e.g. ``"vegetation"``, ``"furniture"``).
    position : tuple[float, float, float]
        World-space (x, y, z) location.
    polygon_count : int
        Number of polygons in the object's mesh.
    material_names : list[str]
        Materials assigned to the object.
    bounding_box_min : tuple[float, float, float]
        Axis-aligned bounding box minimum corner.
    bounding_box_max : tuple[float, float, float]
        Axis-aligned bounding box maximum corner.
    """

    name: str = ""
    category: str = "unknown"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    polygon_count: int = 0
    material_names: list[str] = field(default_factory=list)
    bounding_box_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box_max: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SceneMetadata:
    """Top-level metadata container for a generated scene.

    Parameters
    ----------
    scene_seed : int
        The base random seed used for generation.
    complexity_level : int
        The curriculum complexity level (1-5), or 0 if not using curriculum.
    quality_preset : str
        The render quality preset used (e.g. ``"medium"``).
    generation_time_s : float
        Total wall-clock time for scene generation (seconds).
    objects : list[ObjectRecord]
        Per-object metadata records.
    gin_overrides : list[str]
        Gin configuration overrides applied at generation time.
    stage_timings : dict[str, float]
        Per-stage wall-clock timings in seconds.
    extra : dict
        Arbitrary additional metadata.
    """

    scene_seed: int = 0
    complexity_level: int = 0
    quality_preset: str = "medium"
    generation_time_s: float = 0.0
    objects: list[ObjectRecord] = field(default_factory=list)
    gin_overrides: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def total_polygon_count(self) -> int:
        return sum(o.polygon_count for o in self.objects)

    @property
    def categories(self) -> set[str]:
        return {o.category for o in self.objects}

    def add_object(self, record: ObjectRecord) -> None:
        """Append an :class:`ObjectRecord` to the scene metadata."""
        self.objects.append(record)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain JSON-serialisable dictionary."""
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        """Write metadata to a JSON file."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info("Saved scene metadata to %s", path)
        except OSError as exc:
            logger.error("Failed to save metadata to %s: %s", path, exc)
            raise

    @classmethod
    def load_json(cls, path: Path | str) -> SceneMetadata:
        """Load metadata from a JSON file."""
        path = Path(path)
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Failed to load metadata from %s: %s", path, exc)
            raise
        objects = [ObjectRecord(**o) for o in data.pop("objects", [])]
        # Filter to only known fields to tolerate schema evolution
        import dataclasses as _dc

        allowed = {f.name for f in _dc.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        unexpected = set(data.keys()) - allowed
        if unexpected:
            logger.warning("Ignoring unexpected metadata fields: %s", unexpected)
        return cls(objects=objects, **filtered)


class MetadataCollector:
    """Accumulates metadata during scene generation.

    Typically instantiated once at the start of a scene and populated
    throughout the pipeline.  Call :meth:`finalise` to get the completed
    :class:`SceneMetadata`.
    """

    def __init__(
        self,
        scene_seed: int = 0,
        complexity_level: int = 0,
        quality_preset: str = "medium",
    ):
        self._start_time = time.monotonic()
        self._metadata = SceneMetadata(
            scene_seed=scene_seed,
            complexity_level=complexity_level,
            quality_preset=quality_preset,
        )

    def record_object(self, record: ObjectRecord) -> None:
        self._metadata.add_object(record)

    def record_stage_timing(self, stage_name: str, elapsed_s: float) -> None:
        self._metadata.stage_timings[stage_name] = elapsed_s

    def record_gin_overrides(self, overrides: list[str]) -> None:
        self._metadata.gin_overrides = list(overrides)

    def set_extra(self, key: str, value: Any) -> None:
        self._metadata.extra[key] = value

    def finalise(self) -> SceneMetadata:
        """Return the completed metadata, filling in generation time."""
        self._metadata.generation_time_s = time.monotonic() - self._start_time
        return self._metadata
