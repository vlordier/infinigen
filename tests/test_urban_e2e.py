"""Step-by-step end-to-end urban pipeline test.

Usage:
    blender -b -P tests/test_urban_e2e.py              # all steps
    blender -b -P tests/test_urban_e2e.py -- --step 3  # single step
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bpy


def setup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()


STEP_HELP = {
    1: "city skeleton",
    2: "fill blocks (templates + lots)",
    3: "DCEL + GraphParser",
    4: "mesh roads + sidewalks",
    5: "mesh intersections",
    6: "road markings + crosswalks",
    7: "buildings",
    8: "streetlights",
    9: "landmarks",
    10: "OpenDRIVE export",
}


def run_step(step, pipe):
    if step == 1:
        sk = pipe.step1_skeleton()
        assert len(sk.road_segments) > 0, "No road segments"
        assert len(sk.blocks) > 0, "No blocks"
        print(f"  skeleton: {len(sk.road_segments)} roads, {len(sk.blocks)} blocks")
        return sk

    if step == 2:
        sk = run_step(1, pipe)
        segs, lots = pipe.step2_fill_blocks(sk)
        assert len(segs) > 0, "No segments after fill"
        print(f"  fill: {len(segs)} segments, {len(lots)} lots")
        return segs, lots

    if step == 3:
        segs, lots = run_step(2, pipe)
        dcel, parser = pipe.step3_dcel(segs)
        assert len(parser.road_segments) > 0, "No parsed segments"
        assert len(parser.city_areas) > 0, "No city areas"
        print(f"  DCEL: {len(dcel.nodes)} nodes, {len(dcel.faces)} faces")
        print(f"  parser: {len(parser.road_segments)} road segments, {len(parser.city_areas)} areas")
        return dcel, parser, lots

    if step == 4:
        dcel, parser, lots = run_step(3, pipe)
        roads, sw = pipe.step4_mesh_roads(parser)
        assert len(roads) > 0, "No road meshes"
        print(f"  roads: {len(roads)} meshes, {len(sw)} sidewalk meshes")
        return roads, sw, dcel, parser, lots

    if step == 5:
        roads, sw, dcel, parser, lots = run_step(4, pipe)
        intersections = pipe.step5_mesh_intersections(dcel, parser)
        print(f"  intersections: {len(intersections)} meshes")
        return roads, sw, dcel, parser, lots, intersections

    if step == 6:
        roads, sw, dcel, parser, lots, intersections = run_step(5, pipe)
        marks, crosses = pipe.step6_markings(dcel, parser)
        print(f"  markings: {len(marks)} lines, {len(crosses)} crosswalk strips")
        return roads, sw, dcel, parser, lots, intersections, marks, crosses

    if step == 7:
        r = run_step(6, pipe)
        roads, sw, dcel, parser, lots, intersections, marks, crosses = r
        buildings = pipe.step7_buildings(lots)
        assert len(buildings) > 0, "No building meshes"
        print(f"  buildings: {len(buildings)} shells")
        return r + (buildings,)

    if step == 8:
        r = run_step(7, pipe)
        roads, sw, dcel, parser, lots, intersections, marks, crosses, buildings = r
        sl, lt = pipe.step8_streetlights(parser)
        print(f"  streetlights: {len(sl)} mesh objects, {len(lt)} light objects")
        return r + (sl, lt)

    if step == 9:
        r = run_step(8, pipe)
        *_, sl, lt = r
        landmarks = pipe.step9_landmarks()
        print(f"  landmarks: {len(landmarks)} objects")
        return r + (landmarks,)

    if step == 10:
        r = run_step(9, pipe)
        parser = r[3]
        xodr = pipe.step10_opendrive(parser)
        assert os.path.exists(xodr), f"OpenDRIVE not written: {xodr}"
        sz = os.path.getsize(xodr)
        print(f"  OpenDRIVE: {xodr} ({sz} bytes)")
        return r + (xodr,)

    raise ValueError(f"Unknown step {step}")


if __name__ == "__main__":
    import os
    max_step = 10
    single_step = None
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        extra = sys.argv[idx + 1:]
        for e in extra:
            if e.startswith("--step="):
                single_step = int(e.split("=")[1])
                max_step = single_step

    setup_scene()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")

    steps = [single_step] if single_step else range(1, max_step + 1)
    for s in steps:
        print(f"\n--- Step {s}: {STEP_HELP[s]} ---")
        run_step(s, pipe)

    total = sum(1 for _ in bpy.data.objects)
    print(f"\n{'='*40}\nALL {len(steps)} STEP(S) PASSED  (total objects: {total})\n{'='*40}")