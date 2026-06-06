"""End-to-end urban pipeline test.

Runs with Blender: blender -b -P tests/smoke_test_urban_e2e.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bpy

from infinigen.assets.urban.city_presets import load_preset
from infinigen.assets.urban.skeleton import RadialGenerator, GridGenerator
from infinigen.assets.urban.templates import RectangularGridTemplate, DistrictTemplateConfig
from infinigen.assets.urban.road_to_dcel import RoadToDCEL
from infinigen.assets.urban.graph_parser import GraphParser
from infinigen.assets.urban.road_mesher import RoadMesher
from infinigen.assets.urban.intersection import IntersectionMesher
from infinigen.assets.urban.road_markings import RoadMarkingMesher
from infinigen.assets.urban.block_subdivision import subdivide_block_fill
from infinigen.assets.urban.buildings.building_generator import generate_building_shell
from infinigen.assets.urban.infrastructure.streetlights import place_streetlights
from infinigen.assets.urban.buildings.landmarks import place_landmarks
from infinigen.assets.urban.regional_styles import get_regional_style
from infinigen.assets.urban.opendrive_exporter import export_opendrive

import random, math


def test_pipeline():
    seed = 42
    size = 200
    rng = random.Random(seed)

    # Step 1: Generate skeleton
    skeleton = RadialGenerator.generate(size=size, seed=seed)
    assert len(skeleton.road_segments) > 0, "No road segments"
    assert len(skeleton.blocks) > 0, "No blocks"

    # Step 2: Fill blocks with templates
    all_segments = list(skeleton.road_segments)
    all_lots = []
    for block in skeleton.blocks:
        config = DistrictTemplateConfig()
        fill = RectangularGridTemplate.fill(block.boundary, config, rng)
        all_segments.extend(fill.road_segments)
        all_lots.extend(fill.building_lots)

    # Step 3: Subdivide any remaining blocks
    dcel = RoadToDCEL.build(all_segments)
    parser = GraphParser(dcel)
    assert len(parser.road_segments) > 0, "No parsed road segments"

    if not all_lots:
        for block in skeleton.blocks:
            all_lots.extend(subdivide_block_fill(block.boundary, rng=rng))
    assert len(all_lots) > 0, "No building lots"

    # Step 4: Mesh roads
    mesher = RoadMesher()
    road_objs = mesher.mesh_roads(parser.road_segments)
    assert len(road_objs) > 0, "No road meshes"
    sidewalk_objs = mesher.mesh_sidewalks(parser.road_segments)

    # Step 5: Mesh intersections
    inter_mesher = IntersectionMesher()
    inter_objs = inter_mesher.mesh_intersections(dcel, parser.road_segments)

    # Step 6: Road markings
    mark_mesher = RoadMarkingMesher()
    mark_objs = mark_mesher.mesh_markings(parser.road_segments)
    cross_objs = mark_mesher.mesh_crosswalks(dcel, parser.road_segments)

    # Step 7: Buildings
    bldg_objs = []
    for idx, lot in enumerate(all_lots):
        h = max(6.0, min(30.0, lot.area ** 0.5 * 0.5))
        obj = generate_building_shell(lot.boundary, h, name_suffix=str(idx))
        bpy.context.scene.collection.objects.link(obj)
        bldg_objs.append(obj)
    assert len(bldg_objs) > 0, "No building meshes"

    # Step 8: Streetlights
    light_positions = [
        ((s.source[0]+s.target[0])*0.5, (s.source[1]+s.target[1])*0.5)
        for s in parser.road_segments if s.sidewalk
    ]
    if light_positions:
        sl, lo = place_streetlights(light_positions, spacing=30, seed=seed + 4)
        assert len(sl) > 0, "No streetlight meshes"

    # Step 9: Landmarks
    regional_style = get_regional_style("mediterranean")
    bounds = (size, size)
    landmarks = place_landmarks(bounds, regional_style, count=5, seed=seed + 5)
    assert len(landmarks) > 0, "No landmarks placed"

    # Step 10: OpenDRIVE export
    export_path = "/tmp/test_e2e.xodr"
    export_opendrive(parser.road_segments, export_path)
    assert os.path.exists(export_path), "OpenDRIVE not exported"

    total = len(bpy.data.objects)
    print(f"PASS: Roads={len(road_objs)} Sidewalks={len(sidewalk_objs)} Intersections={len(inter_objs)} "
          f"Buildings={len(bldg_objs)} Markings={len(mark_objs)} Crosswalks={len(cross_objs)} "
          f"Streetlights={len(sl) if light_positions else 0} Landmarks={len(landmarks)} "
          f"Total={total}")
    return True


if __name__ == "__main__":
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()
    success = test_pipeline()
    sys.exit(0 if success else 1)