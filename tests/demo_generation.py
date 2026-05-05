#!/usr/bin/env python3
"""Demo generation: creates urban scene with seasonal effects and ISR camera, renders frames."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy
import gin
from mathutils import Vector

gin.enter_interactive_mode()

from infinigen.assets.weather.season_system import create_season_state
from infinigen.assets.weather.time_of_day import TimeOfDay
from infinigen.assets.urban.regional_styles import get_regional_style
from infinigen.assets.urban.road_network import RoadGraph
from infinigen.assets.urban.buildings.building_generator import generate_buildings_from_lots
from infinigen.assets.urban.urban_surface import create_urban_materials
from infinigen.core.placement.flight_camera import get_platform_rig_spec
from infinigen.core.placement.flight_trajectories import get_flight_policy

random.seed(42)
outdir = os.path.join(os.path.dirname(__file__), "..", "output", "demo")
os.makedirs(outdir, exist_ok=True)

print("=" * 60)
print("INFINIGEN VISPOS DEMO GENERATION")
print("=" * 60)

# Setup
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.resolution_percentage = 100

# Clear default
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# 1. Create terrain proxy
bpy.ops.mesh.primitive_plane_add(size=1000, location=(0,0,0))
terrain = bpy.context.active_object
terrain.name = "Terrain"

# 2. Season
season = create_season_state("winter")
print(f"Season: {season.season}")

# 3. Road network
road_graph = RoadGraph(bounds=(800, 800))
road_graph.generate_grid(spacing=100, jitter=10)
print(f"Roads: {len(road_graph.edges)} edges, {len(road_graph.nodes)} nodes")

# 4. Buildings
style = get_regional_style("soviet")
lots = []
for i in range(5):
    for j in range(5):
        x = -300 + i * 150 + random.uniform(-20, 20)
        y = -300 + j * 150 + random.uniform(-20, 20)
        lots.append([(x-30, y-20), (x+30, y-20), (x+30, y+20), (x-30, y+20)])

scene_collection = bpy.context.scene.collection
buildings = generate_buildings_from_lots(lots, style, seed=42)
for b in buildings:
    scene_collection.objects.link(b)
print(f"Buildings: {len(buildings)}")

# 5. Landmarks — needs bmesh lookup table fix, skip for now
print("Landmarks: skipped (bmesh fix pending)")

# 6. Camera rig (ISR orbit)
rig_spec = get_platform_rig_spec("isr_orbit")
policy = get_flight_policy("isr_orbit", rig_spec, None)
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 300))
rig = bpy.context.active_object
rig.name = "ISR_Rig"
bpy.ops.object.camera_add(location=(0, 0, 0))
cam = bpy.context.active_object
cam.name = "ISR_Cam"
cam.parent = rig
scene.camera = cam

# 7. Lighting
bpy.context.scene.world = bpy.data.worlds.new("Sky")
bpy.context.scene.world.use_nodes = True

# 8. Render frames
n_frames = 3
for frame in range(n_frames):
    bpy.context.scene.frame_set(frame)
    policy(None, rig, frame)
    outpath = os.path.join(outdir, f"demo_frame_{frame:03d}.png")
    scene.render.filepath = outpath
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered frame {frame}/{n_frames} -> {outpath}")

print(f"\nDemo complete. Output: {outdir}")
