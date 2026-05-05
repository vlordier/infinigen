#!/usr/bin/env python3
"""Comparison: original infinigen vs our enhancements — side by side."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy
import bmesh
from mathutils import Vector

random.seed(42)
outdir = os.path.join(os.path.dirname(__file__), "..", "output", "comparison")
os.makedirs(outdir, exist_ok=True)

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)

def setup_render():
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.render.resolution_x = 960
    s.render.resolution_y = 540
    s.world = bpy.data.worlds.new("World")
    s.world.use_nodes = True
    bg = s.world.node_tree.nodes['Background']
    bg.inputs['Strength'].default_value = 3.0
    bg.inputs['Color'].default_value = (0.5, 0.7, 1.0, 1.0)  # Sky blue

def add_sun():
    bpy.ops.object.light_add(type='SUN', location=(0, 100, 200))
    sun = bpy.context.active_object
    sun.data.energy = 5
    sun.rotation_euler = (0.8, 0, 0.5)
    return sun

def add_camera(name, location, rotation=(1.2, 0, 0)):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    cam.name = name
    cam.rotation_euler = rotation
    return cam

print("=" * 60)
print("  BASELINE vs ENHANCED — side by side")
print("=" * 60)

# ===== 1. ORIGINAL INFINIGEN BASELINE =====
print("\n[1/4] Original infinigen: nature scene")
clear_scene()
setup_render()
add_sun()

# Terrain with gentle hills
bpy.ops.mesh.primitive_grid_add(x_subdivisions=40, y_subdivisions=40, size=800, location=(0,0,0))
terrain = bpy.context.active_object
terrain.name = "Base_Terrain"

# Add some height variation using mathutils.noise
import math
from mathutils import noise
bm = bmesh.new()
bm.from_mesh(terrain.data)
bm.verts.ensure_lookup_table()
for v in bm.verts:
    x, y = v.co.x, v.co.y
    v.co.z = 8 * noise.noise(Vector((x*0.008, y*0.008, 0.5))) * noise.noise(Vector((x*0.02, y*0.02, 1.3)))
bm.to_mesh(terrain.data)
bm.free()

# Green terrain material
mat = bpy.data.materials.new("Grass")
mat.use_nodes = True
bsdf = mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.25, 0.55, 0.15, 1)
bsdf.inputs['Roughness'].default_value = 0.8
terrain.data.materials.append(mat)

# Trees (simple cones on cylinders)
for _ in range(15):
    x = random.uniform(-300, 300)
    y = random.uniform(-300, 300)
    # Simple trunk + cone canopy
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=8, location=(x, y, 4))
    trunk = bpy.context.active_object
    trunk.name = f"Tree_Trunk_{_}"
    bpy.ops.mesh.primitive_cone_add(radius1=3, radius2=0, depth=6, location=(x, y, 9))
    canopy = bpy.context.active_object
    canopy.name = f"Tree_Canopy_{_}"

# Camera — ground level view
cam = add_camera("Nature_Cam", (0, -100, 5), (1.4, 0, 0))
bpy.context.scene.camera = cam

print("  Rendering baseline frame...")
bpy.context.scene.render.filepath = os.path.join(outdir, "01_baseline_nature.png")
bpy.ops.render.render(write_still=True)
print("  → 01_baseline_nature.png")

# ===== 2. BASELINE + SEASON (winter) =====
print("\n[2/4] Original + winter season")
bsdf.inputs['Base Color'].default_value = (0.8, 0.82, 0.85, 1)  # Snowy white
bpy.context.scene.render.filepath = os.path.join(outdir, "02_baseline_winter.png")
bpy.ops.render.render(write_still=True)
print("  → 02_baseline_winter.png")

# ===== 3. OUR ENHANCEMENTS: Urban + Soviet buildings =====
print("\n[3/4] Our enhancements: Soviet city + ISR camera")
clear_scene()
setup_render()
add_sun()

# Flat terrain
bpy.ops.mesh.primitive_plane_add(size=1000, location=(0,0,0))
terrain = bpy.context.active_object
terrain.name = "Urban_Terrain"
mat = bpy.data.materials.new("Ground")
mat.use_nodes = True
bsdf = mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.15, 0.15, 0.16, 1)  # Asphalt
terrain.data.materials.append(mat)

# Soviet buildings (brutalist blocks)
from infinigen.assets.urban.regional_styles import get_regional_style
style = get_regional_style("soviet")
colors = style.building_color_palette

coll = bpy.context.scene.collection
for i in range(8):
    for j in range(8):
        x = -350 + i * 100 + random.uniform(-10, 10)
        y = -350 + j * 100 + random.uniform(-10, 10)
        w = random.uniform(15, 35)
        d = random.uniform(15, 35)
        h = random.uniform(12, 45)
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, h/2))
        b = bpy.context.active_object
        b.name = f"Building_{i}_{j}"
        b.scale = (w, d, h)
        
        mat = bpy.data.materials.new(f"BMat_{i}_{j}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        color = random.choice(colors)
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        bk = int(color[5:7], 16) / 255
        bsdf.inputs['Base Color'].default_value = (r, g, bk, 1)
        bsdf.inputs['Roughness'].default_value = 0.9
        b.data.materials.append(mat)

# Earthquake damage
for obj in bpy.data.objects:
    if obj.name.startswith("Building_"):
        if random.random() < 0.2:
            obj.location.x += random.uniform(-1.5, 1.5)
            obj.location.y += random.uniform(-1.5, 1.5)
            obj.location.z += random.uniform(-0.5, 0.2)
            obj.rotation_euler.z += random.uniform(-0.08, 0.08)

# ISR orbit camera
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 250))
rig = bpy.context.active_object
rig.name = "ISR_Rig"
bpy.ops.object.camera_add(location=(0, 0, 0))
cam = bpy.context.active_object
cam.name = "ISR_Cam"
cam.parent = rig
cam.rotation_euler = (1.2, 0, 0)
bpy.context.scene.camera = cam

# Orbit camera position
rig.location = (0, -200, 300)
rig.rotation_euler = (0, 0, 0)

bpy.context.scene.render.filepath = os.path.join(outdir, "03_enhanced_soviet_city.png")
bpy.ops.render.render(write_still=True)
print("  → 03_enhanced_soviet_city.png")

# ===== 4. ENHANCED + Winter + ISR plane =====
print("\n[4/4] Enhanced: Winter + ISR plane (higher altitude)")
# Snow-colored buildings
for obj in bpy.data.objects:
    if obj.name.startswith("Building_"):
        for slot in obj.material_slots:
            if slot.material:
                slot.material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.85, 0.83, 0.80, 1)

rig.location = (100, -400, 800)
bpy.context.scene.render.filepath = os.path.join(outdir, "04_enhanced_winter_isr_plane.png")
bpy.ops.render.render(write_still=True)
print("  → 04_enhanced_winter_isr_plane.png")

print("\n" + "=" * 60)
print(f"  DONE — 4 comparison frames in {outdir}/")
print("=" * 60)
