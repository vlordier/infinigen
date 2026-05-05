"""Comparison: Original infinigen-style nature vs Our enhanced urban scene."""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy, bmesh, math
import mathutils
from mathutils import Vector, Euler

random.seed(42)
outdir = os.path.join(os.path.dirname(__file__), "..", "output", "comparison_v2")
os.makedirs(outdir, exist_ok=True)

def clear_scene():
    for obj in list(bpy.data.objects): bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials): bpy.data.materials.remove(mat, do_unlink=True)

def setup_nishita_sky(sun_elevation=45, sun_rotation=180, strength=5):
    world = bpy.data.worlds.new("NishitaWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    tex = nodes.new('ShaderNodeTexSky')
    tex.sky_type = 'HOSEK_WILKIE'
    tex.sun_elevation = math.radians(sun_elevation)
    tex.sun_rotation = math.radians(sun_rotation)
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    out = nodes.new('ShaderNodeOutputWorld')
    links.new(tex.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], out.inputs['Surface'])

def setup_render():
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.samples = 64
    s.cycles.use_denoising = True
    s.render.resolution_x = 960
    s.render.resolution_y = 540

def make_material(name, color, roughness=0.7):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return mat

def apply_noise_terrain(obj, scale=0.01, height=15):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        v.co.z = height * (
            math.sin(v.co.x * scale * 1.7) * math.cos(v.co.y * scale * 2.1) * 0.7 +
            math.sin(v.co.x * scale * 3.3 + 1.5) * math.cos(v.co.y * scale * 4.1 + 2.3) * 0.3
        )
    bm.to_mesh(obj.data)
    bm.free()

def add_trees(count, bounds, terrain_obj=None):
    trees = []
    for _ in range(count):
        x = random.uniform(-bounds, bounds)
        y = random.uniform(-bounds, bounds)
        z = 0
        if terrain_obj:
            # Find ground height via raycast in simplified way
            bm = bmesh.new()
            bm.from_mesh(terrain_obj.data)
            bm.faces.ensure_lookup_table()
            z = 0
            for face in bm.faces:
                p = Vector((x, y, 999))
                d = Vector((0, 0, -1))
                hit = mathutils.geometry.intersect_ray_tri(
                    face.verts[0].co, face.verts[1].co, face.verts[2].co, d, p)
                if hit: z = max(z, p.z - hit.z)
            bm.free()
        
        h = random.uniform(5, 15)
        bpy.ops.mesh.primitive_cylinder_add(radius=random.uniform(0.3, 0.7), depth=h, location=(x, y, z + h/2))
        trunk = bpy.context.active_object
        mat = make_material(f"Bark_{_}", (0.35, 0.2, 0.1), 0.9)
        trunk.data.materials.append(mat)
        
        bpy.ops.mesh.primitive_cone_add(radius1=random.uniform(3, 6), radius2=0, depth=random.uniform(5, 10),
                                        location=(x, y, z + h + 3))
        canopy = bpy.context.active_object
        r, g = random.uniform(0.1, 0.35), random.uniform(0.35, 0.7)
        mat = make_material(f"Leaf_{_}", (r, g, random.uniform(0.05, 0.15)), 0.9)
        canopy.data.materials.append(mat)
        trees.append((trunk, canopy))
    return trees

print("=" * 60)
print("  ORIGINAL INFINIGEN  vs  OUR ENHANCEMENTS")
print("=" * 60)

# ===== 1. NATURE SCENE (baseline) =====
print("\n[1] Nature scene — hinigin-style forest")
clear_scene()
setup_render()
setup_nishita_sky(sun_elevation=40, sun_rotation=160, strength=4)

bpy.ops.mesh.primitive_grid_add(x_subdivisions=50, y_subdivisions=50, size=800, location=(0,0,0))
terrain = bpy.context.active_object
terrain.name = "Terrain"
apply_noise_terrain(terrain, scale=0.008, height=20)
mat = make_material("Grass", (0.15, 0.45, 0.1), 0.85)
terrain.data.materials.append(mat)

add_trees(30, 350, terrain)

cam = bpy.data.cameras.new("NatureCam")
cam_obj = bpy.data.objects.new("NatureCam", cam)
bpy.context.scene.collection.objects.link(cam_obj)
cam_obj.location = (0, -120, 8)
cam_obj.rotation_euler = Euler((math.radians(85), 0, 0))
bpy.context.scene.camera = cam_obj

bpy.context.scene.render.filepath = os.path.join(outdir, "01_nature_forest.png")
bpy.ops.render.render(write_still=True)
print(" → 01_nature_forest.png (ground-level view)")

# ===== 2. NATURE WINTER =====
print("\n[2] Nature + winter (material override)")
mat_grass = bpy.data.materials['Grass']
mat_grass.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.85, 0.83, 0.80, 1)
bpy.context.scene.render.filepath = os.path.join(outdir, "02_nature_winter.png")
bpy.ops.render.render(write_still=True)
print(" → 02_nature_winter.png")

# ===== 3. OUR ENHANCEMENTS: Soviet city =====
print("\n[3] Our enhancements: Soviet city + ISR orbit camera")
clear_scene()
setup_render()
setup_nishita_sky(sun_elevation=50, sun_rotation=190, strength=5)

bpy.ops.mesh.primitive_plane_add(size=1000, location=(0,0,0))
terrain = bpy.context.active_object
terrain.name = "UrbanGround"
make_material("Asphalt", (0.12, 0.12, 0.14), 0.95)
terrain.data.materials.append(bpy.data.materials['Asphalt'])

from infinigen.assets.urban.regional_styles import get_regional_style
style = get_regional_style("soviet")
colors = style.building_color_palette
coll = bpy.context.scene.collection

# Roads (cross pattern)
for i in range(3):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -300 + i*300, 0.02))
    r = bpy.context.active_object
    r.scale = (20, 300, 0.02)
    make_material(f"Road_h{i}", (0.08, 0.08, 0.09), 0.95)
    r.data.materials.append(bpy.data.materials[f"Road_h{i}"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-300 + i*300, 0, 0.02))
    r = bpy.context.active_object
    r.scale = (300, 20, 0.02)
    make_material(f"Road_v{i}", (0.08, 0.08, 0.09), 0.95)
    r.data.materials.append(bpy.data.materials[f"Road_v{i}"])

# Buildings with varied heights and materials
for i in range(10):
    for j in range(10):
        x = -360 + i * 80 + random.uniform(-10, 10)
        y = -360 + j * 80 + random.uniform(-10, 10)
        # Skip road intersections
        if (abs(x) < 30 or abs(y) < 30) and random.random() < 0.7: continue
        
        w = random.uniform(12, 30)
        d = random.uniform(12, 30)
        floors = random.randint(3, 15)
        h = floors * 3.2
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, h/2))
        bld = bpy.context.active_object
        bld.name = f"Bld_{i}_{j}"
        bld.scale = (w, d, h)
        
        color = random.choice(colors)
        rv = int(color[1:3],16)/255; gv = int(color[3:5],16)/255; bv = int(color[5:7],16)/255
        make_material(f"BM_{i}_{j}", (rv, gv, bv), 0.85 + random.random()*0.1)
        bld.data.materials.append(bpy.data.materials[f"BM_{i}_{j}"])

# Water tower landmark
bpy.ops.mesh.primitive_cylinder_add(radius=4, depth=2, location=(200, 200, 28))
tank = bpy.context.active_object
make_material("TankMetal", (0.5, 0.5, 0.55), 0.3)
tank.data.materials.append(bpy.data.materials['TankMetal'])
for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
    x = 200 + math.cos(angle)*3
    y = 200 + math.sin(angle)*3
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=25, location=(x, y, 13))
    leg = bpy.context.active_object
    leg.data.materials.append(bpy.data.materials['TankMetal'])

# Cell tower
bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=50, location=(-200, 200, 25))
tower = bpy.context.active_object
tower.data.materials.append(bpy.data.materials['TankMetal'])

# Mild earthquake
for obj in bpy.data.objects:
    if obj.name.startswith("Bld_"):
        if random.random() < 0.2:
            obj.location.x += random.uniform(-1, 1)
            obj.location.y += random.uniform(-1, 1)
            obj.rotation_euler.z += random.uniform(-0.05, 0.05)

# ISR orbit camera
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, -250, 350))
rig = bpy.context.active_object
bpy.ops.object.camera_add()
cam = bpy.context.active_object
cam.parent = rig
cam.rotation_euler = Euler((math.radians(65), 0, 0))
bpy.context.scene.camera = cam

bpy.context.scene.render.filepath = os.path.join(outdir, "03_soviet_city_isr.png")
bpy.ops.render.render(write_still=True)
print(" → 03_soviet_city_isr.png (64 buildings + landmarks + ISR camera)")

# ===== 4. ENHANCED + WINTER + DAMAGE =====
print("\n[4] Enhanced: Soviet city, winter, heavy earthquake + ISR plane")
# Snow terrain
mat_asphalt = bpy.data.materials['Asphalt']
mat_asphalt.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.7, 0.72, 0.74, 1)
# Snow buildings
for obj in bpy.data.objects:
    if obj.name.startswith("Bld_"):
        for slot in obj.material_slots:
            if slot.material: 
                slot.material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.82, 0.81, 0.79, 1)
        # More damage
        if random.random() < 0.4:
            obj.location.x += random.uniform(-2, 2)
            obj.location.y += random.uniform(-2, 2)
            obj.location.z += random.uniform(-0.8, 0.1)
            obj.rotation_euler.x += random.uniform(-0.1, 0.1)
            obj.rotation_euler.z += random.uniform(-0.12, 0.12)

# ISR plane — higher, further
rig.location = (300, -500, 1200)
cam.rotation_euler = Euler((math.radians(80), 0, 0))
setup_nishita_sky(sun_elevation=15, sun_rotation=120, strength=2)

bpy.context.scene.render.filepath = os.path.join(outdir, "04_winter_damage_isr_plane.png")
bpy.ops.render.render(write_still=True)
print(" → 04_winter_damage_isr_plane.png (winter + heavy quake + ISR plane)")

print("\n" + "=" * 60)
print(f"  DONE — {outdir}/")
print("=" * 60)
