#!/usr/bin/env python3
"""Smoke test for DCEL-based road generation.

Run with: blender -b -P tests/smoke_test_urban.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bpy
from infinigen.core.init import configure_blender
from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.graph_parser import GraphParser
from infinigen.assets.urban.road_mesher import RoadMesher
from infinigen.assets.urban.block_subdivision import subdivide_lots


def main():
    configure_blender()
    dcel = GraphGenerator.generate(200, 200, seed=42)
    parser = GraphParser(dcel)
    print(f"Generated {len(parser.road_segments)} road segments")
    print(f"Generated {len(parser.city_areas)} city areas")
    assert len(parser.road_segments) > 0, "No road segments generated"
    assert len(parser.city_areas) > 0, "No city areas generated"
    mesher = RoadMesher()
    road_objs = mesher.mesh_roads(parser.road_segments)
    sidewalk_objs = mesher.mesh_sidewalks(parser.road_segments)
    print(f"Created {len(road_objs)} road meshes")
    print(f"Created {len(sidewalk_objs)} sidewalk meshes")
    assert len(road_objs) > 0, "No road meshes created"
    lots = subdivide_lots(parser.city_areas, seed=42)
    print(f"Created {len(lots)} building lots")
    assert len(lots) > 0, "No building lots created"
    total_objs = len(bpy.data.objects)
    print(f"Total objects in scene: {total_objs}")
    assert total_objs >= len(road_objs) + len(sidewalk_objs)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
