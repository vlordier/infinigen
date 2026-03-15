# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Scene statistics collection and reporting.

While :func:`save_polycounts` in ``core/util/logging.py`` captures basic
polygon counts via ``bpy``, this module provides a richer, structured
representation of scene metadata that can be serialised to JSON for
downstream analysis, dashboards, or regression testing.

All functions in this module require ``bpy`` to be available.

Usage::

    from infinigen.core.util.scene_stats import collect_scene_stats

    stats = collect_scene_stats()
    stats.save_json(output_folder / "scene_stats.json")
    print(stats.summary())
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import bpy
import psutil

logger = logging.getLogger(__name__)


@dataclass
class SceneStats:
    """Structured snapshot of scene-level statistics.

    All counts are computed at the moment of collection and reflect the
    current state of the Blender scene.
    """

    # Object counts
    total_objects: int = 0
    mesh_objects: int = 0
    light_objects: int = 0
    camera_objects: int = 0
    empty_objects: int = 0
    curve_objects: int = 0
    other_objects: int = 0

    # Geometry
    total_polygons: int = 0
    total_vertices: int = 0
    total_edges: int = 0

    # Materials and textures
    total_materials: int = 0
    total_images: int = 0

    # Collections
    total_collections: int = 0
    collection_names: list[str] = field(default_factory=list)

    # Per-collection polygon counts
    collection_polycounts: dict[str, int] = field(default_factory=dict)

    # Memory
    process_memory_mb: float = 0.0

    # Timing (optional, filled in by callers)
    collection_timestamp: float = 0.0

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            "Scene Statistics",
            "=" * 40,
            f"  Objects:     {self.total_objects:,} total",
            f"    Mesh:      {self.mesh_objects:,}",
            f"    Light:     {self.light_objects:,}",
            f"    Camera:    {self.camera_objects:,}",
            f"    Empty:     {self.empty_objects:,}",
            f"    Curve:     {self.curve_objects:,}",
            f"    Other:     {self.other_objects:,}",
            f"  Geometry:    {self.total_polygons:,} polys, "
            f"{self.total_vertices:,} verts, {self.total_edges:,} edges",
            f"  Materials:   {self.total_materials:,}",
            f"  Images:      {self.total_images:,}",
            f"  Collections: {self.total_collections:,}",
            f"  Memory:      {self.process_memory_mb:.1f} MB",
        ]
        if self.collection_polycounts:
            lines.append("  Per-collection polycounts:")
            for name, count in sorted(
                self.collection_polycounts.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {name}: {count:,}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a plain dict (JSON-serialisable)."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write statistics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Scene stats saved to %s", path)


def collect_scene_stats() -> SceneStats:
    """Collect statistics from the current Blender scene.

    Returns
    -------
    SceneStats
        Populated statistics dataclass.
    """
    stats = SceneStats()
    stats.collection_timestamp = time.time()

    # Object counts by type
    type_counts: dict[str, int] = {}
    for obj in bpy.data.objects:
        obj_type = obj.type
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

    stats.total_objects = len(bpy.data.objects)
    stats.mesh_objects = type_counts.get("MESH", 0)
    stats.light_objects = type_counts.get("LIGHT", 0)
    stats.camera_objects = type_counts.get("CAMERA", 0)
    stats.empty_objects = type_counts.get("EMPTY", 0)
    stats.curve_objects = type_counts.get("CURVE", 0)
    stats.other_objects = stats.total_objects - (
        stats.mesh_objects
        + stats.light_objects
        + stats.camera_objects
        + stats.empty_objects
        + stats.curve_objects
    )

    # Geometry totals
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.data is not None:
            mesh = obj.data
            stats.total_polygons += len(mesh.polygons)
            stats.total_vertices += len(mesh.vertices)
            stats.total_edges += len(mesh.edges)

    # Materials and images
    stats.total_materials = len(bpy.data.materials)
    stats.total_images = len(bpy.data.images)

    # Collections
    stats.total_collections = len(bpy.data.collections)
    stats.collection_names = [col.name for col in bpy.data.collections]

    # Per-collection polycounts
    for col in bpy.data.collections:
        polycount = sum(
            len(obj.data.polygons)
            for obj in col.all_objects
            if obj.type == "MESH" and obj.data is not None
        )
        stats.collection_polycounts[col.name] = polycount

    # Memory
    stats.process_memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    return stats
