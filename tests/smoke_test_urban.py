#!/usr/bin/env python3
"""Smoke test for DCEL-based road generation.

Run with: blender -b -P tests/smoke_test_urban.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bpy
from pathlib import Path
from infinigen.core.init import configure_blender
from infinigen.assets.urban.compose_urban import compose_urban


def main():
    configure_blender()
    output_folder = Path("/tmp/smoke_test_urban")
    output_folder.mkdir(parents=True, exist_ok=True)
    compose_urban(output_folder, scene_seed=42)
    total = len(bpy.data.objects)
    road_count = sum(1 for o in bpy.data.objects if o.name.startswith("road_"))
    sidewalk_count = sum(1 for o in bpy.data.objects if o.name.startswith("sidewalk_"))
    intersection_count = sum(1 for o in bpy.data.objects if o.name.startswith("intersection"))
    building_count = sum(1 for o in bpy.data.objects if o.name.startswith("building_shell"))
    streetlight_count = sum(1 for o in bpy.data.objects if o.name.startswith("streetlight_"))
    landmark_count = sum(1 for o in bpy.data.objects if o.name.startswith("landmark_"))
    print(f"Total objects: {total}")
    print(f"Roads: {road_count}, Sidewalks: {sidewalk_count}, Intersections: {intersection_count}")
    print(f"Buildings: {building_count}, Streetlights: {streetlight_count}, Landmarks: {landmark_count}")
    assert road_count > 0, "No road meshes"
    assert building_count > 0, "No buildings"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
