#!/usr/bin/env python3
"""Full demo: soviet buildings + ISR camera + damage effects, 8 frames"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy
from mathutils import Vector

random.seed(42)
outdir = os.path.join(os.path.dirname(__file__), "..", "output", "vispos_demo")
os.makedirs(outdir, exist_ok=True)

print("=" * 60)
print("INFINIGEN VISPOS — FULL DEMO")
print("=" * 60)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 960
scene.render.resolution_y = 540

# Clear scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# 1. Terrain
bpy.ops.mesh.primitive_plane_add(size=1000, location=(0,0,0))
terrain = bpy.context.active_object
terrain.name = "Terrain"
print("[1] Terrain created")

# 2. Season — Winter
from infinigen.assets.weather.season_system import create_season_state
season = create_season_state("winter")
print(f"[2] Season: {season.season} ({season.temperature}°C, snow={season.snow_cover})")

# 3. Soviet buildings
from infinigen.assets.urban.regional_styles import get_regional_style
from infinigen.assets.urban.buildings.building_generator import generate_buildings_from_lots
style = get_regional_style("soviet")
lots = []
for i in range(6):
    for j in range(6):
        x = -350 + i * 120 + random.uniform(-15, 15)
        y = -350 + j * 120 + random.uniform(-15, 15)
        w = random.uniform(25, 50)
        d = random.uniform(25, 50)
        lots.append([(x-w/2, y-d/2), (x+w/2, y-d/2), (x+w/2, y+d/2), (x-w/2, y+d/2)])

coll = bpy.context.scene.collection
buildings = generate_buildings_from_lots(lots, style, seed=42)
for b in buildings:
    coll.objects.link(b)
print(f"[3] Buildings: {len(buildings)} soviet-style")

# 4. Damage — light earthquake
print("[4] Damage: applying mild earthquake...")
for b in buildings:
    if random.random() < 0.15:
        b.location.x += random.uniform(-0.3, 0.3)
        b.location.y += random.uniform(-0.3, 0.3)
        b.location.z += random.uniform(-0.2, 0.05)
        b.rotation_euler.z += random.uniform(-0.1, 0.1)
    else:
        b.location.z += random.uniform(-0.05, 0.05)

# 5. ISR Camera
from infinigen.core.placement.flight_camera import get_platform_rig_spec
from infinigen.core.placement.flight_trajectories import get_flight_policy
rig_spec = get_platform_rig_spec("isr_orbit")
policy = get_flight_policy("isr_orbit", rig_spec, None)

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 300))
rig = bpy.context.active_object
rig.name = "ISR_Rig"
bpy.ops.object.camera_add()
cam = bpy.context.active_object
cam.name = "ISR_Cam"
cam.parent = rig
scene.camera = cam
print(f"[5] Camera: ISR orbit at {rig_spec.altitude_range[0]}-{rig_spec.altitude_range[1]}m")

# 6. Lighting
scene.world = bpy.data.worlds.new("VisPosWorld")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.new('ShaderNodeBackground')
bg.inputs['Strength'].default_value = 2.0
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 2.0

# 7. Render 8 frames
n_frames = 8
print(f"[6] Rendering {n_frames} frames...")
for frame in range(n_frames):
    scene.frame_set(frame)
    pos, rot = policy(None, rig, frame)
    rig.keyframe_insert(data_path="location", frame=frame)
    rig.keyframe_insert(data_path="rotation_euler", frame=frame)
    outpath = os.path.join(outdir, f"demo_{frame:03d}.png")
    scene.render.filepath = outpath
    bpy.ops.render.render(write_still=True)
    print(f"  Frame {frame+1}/{n_frames} → demo_{frame:03d}.png")

# 8. Metadata
import json
from infinigen.datagen.vispos.ground_truth import FrameMetadata
meta = FrameMetadata(
    scene_id="vispos_demo_001",
    season="winter",
    modality="eo",
    is_damaged=True,
    damage_type="earthquake",
    damage_severity="mild",
)
with open(os.path.join(outdir, "metadata.json"), "w") as f:
    json.dump(meta.to_dict(), f, indent=2)

print(f"\nComplete! {n_frames} frames in {outdir}/")
print("=" * 60)
