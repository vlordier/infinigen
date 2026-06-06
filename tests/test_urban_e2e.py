"""End-to-end tests for each urban pipeline step.

Each test function is self-contained — it generates its own required inputs
and validates its own outputs.  Steps 1-3 are pure Python (run via pytest).
Steps 4-10 need Blender (run via blender -P).

Run all 10 steps:
    blender -b -P tests/test_urban_e2e.py

Run a single step:
    blender -b -P tests/test_urban_e2e.py -- --step=4

Run pure-Python steps with pytest (no Blender):
    PYTHONPATH=. pytest tests/test_urban_e2e.py -v --noconftest
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

STEP_HELP = {
    1: "generate city skeleton",
    2: "fill blocks with templates and lots",
    3: "build DCEL and GraphParser",
    4: "mesh roads and sidewalks",
    5: "mesh intersections",
    6: "road markings and crosswalks",
    7: "buildings",
    8: "streetlights",
    9: "landmarks",
    10: "OpenDRIVE export",
}

# ---------------------------------------------------------------------------
# Pure-Python steps (no Blender needed)
# ---------------------------------------------------------------------------
import random
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.block_subdivision import BuildingLot


def _make_square_block():
    return [(0, 0), (100, 0), (100, 100), (0, 100)]


def _make_cycle_segments():
    """Return 4 RoadSegments forming a 100x100 square."""
    pts = _make_square_block()
    segs = []
    for i in range(4):
        j = (i + 1) % 4
        segs.append(RoadSegment(source=pts[i], target=pts[j],
                                road_type="local", lane_count=2, width=12.0,
                                sidewalk=True))
    return segs


def test_step1():
    """step1_skeleton() produces a CitySkeleton with roads and blocks."""
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    assert len(sk.road_segments) > 0
    assert len(sk.blocks) > 0


def test_step2():
    """step2_fill_blocks() adds template roads and building lots."""
    sk = _make_sk_for_test()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    segs, lots = pipe.step2_fill_blocks(sk)
    assert len(segs) > len(sk.road_segments)
    assert len(lots) > 0


def test_step3():
    """step3_dcel() builds a valid DCEL and GraphParser."""
    sk = _make_sk_for_test()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    segs, _ = pipe.step2_fill_blocks(sk)
    dcel, parser = pipe.step3_dcel(segs)
    assert len(dcel.nodes) > 0
    assert len(parser.road_segments) > 0
    assert len(parser.city_areas) > 0


def _make_sk_for_test():
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    return pipe.step1_skeleton()


# ---------------------------------------------------------------------------
# Blender-dependent steps
# ---------------------------------------------------------------------------


def _setup():
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()
    # Purge material caches so stale references don't survive scene resets
    cache_attrs = ('_MATERIALS_CACHE', '_MATERIALS', '_MATERIAL', 'material_cache')
    for mod_name in ('road_mesher', 'intersection', 'road_markings',
                     'streetlights', 'trees', 'cars',
                     'buildings.building_generator',
                     'buildings.landmarks'):
        mod = sys.modules.get(f'infinigen.assets.urban.{mod_name}')
        if mod:
            for attr in cache_attrs:
                if hasattr(mod, attr):
                    obj = getattr(mod, attr)
                    if isinstance(obj, dict):
                        obj.clear()


def test_step4():
    """step4_mesh_roads() creates Blender road and sidewalk meshes."""
    _setup()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    segs, _ = pipe.step2_fill_blocks(sk)
    dcel, parser = pipe.step3_dcel(segs)
    roads, sw = pipe.step4_mesh_roads(parser)
    assert len(roads) > 0
    assert len(sw) > 0
    print(f"  roads={len(roads)} sidewalks={len(sw)}")


def test_step5():
    """step5_mesh_intersections() creates junction meshes."""
    _setup()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    segs, _ = pipe.step2_fill_blocks(sk)
    dcel, parser = pipe.step3_dcel(segs)
    roads, sw = pipe.step4_mesh_roads(parser)
    inters = pipe.step5_mesh_intersections(dcel, parser)
    assert len(inters) > 0, "No intersection meshes"
    print(f"  roads={len(roads)} intersections={len(inters)}")
    return roads, sw, dcel, parser, inters


def test_step6():
    """step6_markings() creates lane lines and crosswalks."""
    _setup()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    segs, _ = pipe.step2_fill_blocks(sk)
    dcel, parser = pipe.step3_dcel(segs)
    roads, sw = pipe.step4_mesh_roads(parser)
    inters = pipe.step5_mesh_intersections(dcel, parser)
    marks, crosses = pipe.step6_markings(dcel, parser)
    assert len(marks) > 0
    assert len(crosses) > 0
    print(f"  roads={len(roads)} markings={len(marks)} crosswalks={len(crosses)}")


def test_step7():
    """step7_buildings() creates building shell meshes from lots."""
    _setup()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    segs, lots = pipe.step2_fill_blocks(sk)
    bldgs = pipe.step7_buildings(lots)
    assert len(bldgs) > 0
    assert all(o.name.startswith("building_shell_") for o in bldgs)
    print(f"  buildings={len(bldgs)}")


def test_step8():
    """step8_streetlights() creates pole meshes + point lights."""
    _setup()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    segs, _ = pipe.step2_fill_blocks(sk)
    _, parser = pipe.step3_dcel(segs)
    sl, lt = pipe.step8_streetlights(parser)
    assert len(sl) > 0
    assert len(lt) > 0
    assert all(o.name.startswith("streetlight_") for o in sl)
    assert all(o.name.startswith("streetlight_light_") for o in lt)
    print(f"  streetlights={len(sl)} lights={len(lt)}")


def test_step9():
    """step9_landmarks() places building landmarks."""
    _setup()
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    lms = pipe.step9_landmarks()
    assert len(lms) > 0
    assert all(o.name.startswith("landmark_") for o in lms)
    print(f"  landmarks={len(lms)}")


def test_step10():
    """step10_opendrive() exports valid OpenDRIVE XML."""
    from infinigen.assets.urban.pipeline import UrbanPipeline
    pipe = UrbanPipeline(seed=42, city_size=200, preset_name="european_old")
    sk = pipe.step1_skeleton()
    segs, _ = pipe.step2_fill_blocks(sk)
    _, parser = pipe.step3_dcel(segs)
    path = pipe.step10_opendrive(parser, "/tmp/test_step10.xodr")
    assert os.path.exists(path)
    sz = os.path.getsize(path)
    assert sz > 1000
    with open(path) as f:
        content = f.read()
    assert "<OpenDRIVE>" in content
    assert "<road" in content
    assert "<laneSection" in content
    print(f"  OpenDRIVE: {path} ({sz} bytes)")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
_TEST_FN = {
    1: lambda: test_step1() or True,
    2: lambda: test_step2() or True,
    3: lambda: test_step3() or True,
    4: test_step4,
    5: test_step5,
    6: test_step6,
    7: test_step7,
    8: test_step8,
    9: test_step9,
    10: test_step10,
}


if __name__ == "__main__":
    max_step = 10
    single_step = None
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        for e in sys.argv[idx + 1:]:
            if e.startswith("--step="):
                single_step = int(e.split("=")[1])
                max_step = single_step

    steps = [single_step] if single_step else range(1, max_step + 1)
    for s in steps:
        print(f"--- Step {s}: {STEP_HELP[s]} ---")
        _TEST_FN[s]()
        print(f"  PASS")

    print(f"\n{'='*40}\nALL {len(steps)} STEP(S) PASSED\n{'='*40}")