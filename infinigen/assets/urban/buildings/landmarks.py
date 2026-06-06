import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector


class LandmarkType:
    WATER_TOWER = "water_tower"
    CHURCH = "church"
    CELL_TOWER = "cell_tower"
    STADIUM = "stadium"
    SILO = "silo"
    WIND_TURBINE = "wind_turbine"
    LIGHTHOUSE = "lighthouse"
    CRANE = "crane"


_MATERIALS_CACHE = {}


def _get_mat(name, color, roughness=0.7, metallic=0.3):
    if name in _MATERIALS_CACHE:
        return _MATERIALS_CACHE[name]
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    _MATERIALS_CACHE[name] = mat
    return mat


def _box(bm, cx, cy, cz, sx, sy, sz, mat_idx=0):
    hw, hh, hd = sx / 2, sy / 2, sz / 2
    v = [
        bm.verts.new(Vector((cx - hw, cy - hh, cz - hd))),
        bm.verts.new(Vector((cx + hw, cy - hh, cz - hd))),
        bm.verts.new(Vector((cx + hw, cy + hh, cz - hd))),
        bm.verts.new(Vector((cx - hw, cy + hh, cz - hd))),
        bm.verts.new(Vector((cx - hw, cy - hh, cz + hd))),
        bm.verts.new(Vector((cx + hw, cy - hh, cz + hd))),
        bm.verts.new(Vector((cx + hw, cy + hh, cz + hd))),
        bm.verts.new(Vector((cx - hw, cy + hh, cz + hd))),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for fi in faces:
        bm.faces.new([v[i] for i in fi]).material_index = mat_idx


def _cylinder(bm, cx, cy, z_bot, z_top, radius, segs=12, mat_idx=0):
    ring_bot = [bm.verts.new(Vector((cx + math.cos(i/segs*math.pi*2)*radius,
                                      cy + math.sin(i/segs*math.pi*2)*radius,
                                      z_bot))) for i in range(segs)]
    ring_top = [bm.verts.new(Vector((cx + math.cos(i/segs*math.pi*2)*radius,
                                      cy + math.sin(i/segs*math.pi*2)*radius,
                                      z_top))) for i in range(segs)]
    for i in range(segs):
        j = (i + 1) % segs
        bm.faces.new([ring_bot[i], ring_bot[j], ring_top[j], ring_top[i]]).material_index = mat_idx
    return ring_bot, ring_top


def _finalize(name, bm, mat_names):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for mn in mat_names:
        mesh.materials.append(_get_mat(*mn) if isinstance(mn, tuple) else _get_mat(mn, (0.7, 0.7, 0.72, 1)))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@gin.configurable
def generate_water_tower(center=(0, 0, 0), height=25, radius=3):
    bm = bmesh.new()
    cx, cy, cz = center
    leg_h = height * 0.7
    for lx, ly in [(-radius*0.5, -radius*0.5), (radius*0.5, -radius*0.5),
                   (-radius*0.5, radius*0.5), (radius*0.5, radius*0.5)]:
        v = [bm.verts.new(Vector((lx+cx+dx, ly+cy+dy, cz+z))) for dx, dy, z in
             [(0, 0, 0), (0.2, 0, 0), (0.2, 0, leg_h), (0, 0, leg_h)]]
        bm.faces.new(v).material_index = 0
    _cylinder(bm, cx, cy, cz+leg_h, cz+height, radius, mat_idx=1)
    mesh = bpy.data.meshes.new("water_tower")
    bm.to_mesh(mesh); bm.free()
    metal = _get_mat("Urban_Metal", (0.5, 0.45, 0.4, 1), metallic=0.7)
    tank = _get_mat("Urban_Tank", (0.35, 0.5, 0.55, 1), roughness=0.4, metallic=0.3)
    mesh.materials.append(metal); mesh.materials.append(tank)
    obj = bpy.data.objects.new("water_tower", mesh)
    obj.location = Vector(center)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@gin.configurable
def generate_cell_tower(center=(0, 0, 0), height=40):
    bm = bmesh.new()
    cx, cy, cz = center
    segs = 8; base_r, top_r = 1.5, 0.5
    for i in range(segs):
        a1 = (i/segs)*math.pi*2; a2 = ((i+1)/segs)*math.pi*2
        v = [bm.verts.new(Vector((cx+math.cos(a)*r, cy+math.sin(a)*r, cz+z)))
             for a, r, z in [(a1, base_r, 0), (a2, base_r, 0),
                              (a2, top_r, height), (a1, top_r, height)]]
        bm.faces.new(v).material_index = 0
    for i in range(3):
        a = (i/3)*math.pi*2
        px = cx + math.cos(a)*top_r*3; py = cy + math.sin(a)*top_r*3
        v = [bm.verts.new(Vector((px+dx, py+dy, cz+height+z)))
             for dx, dy, z in [(-0.5, -0.1, 0), (0.5, -0.1, 0),
                                (0.5, 0.1, 3), (-0.5, 0.1, 3)]]
        bm.faces.new(v).material_index = 0
    return _finalize("cell_tower", bm, [("Urban_Metal", (0.5, 0.5, 0.55, 1), 0.7, 0.8)])


@gin.configurable
def generate_church(center=(0, 0, 0), height=15):
    bm = bmesh.new()
    cx, cy, cz = center
    _box(bm, cx, cy, cz + height * 0.3, height * 0.5, height * 0.4, height * 0.6, 0)
    steeple_h = height * 0.4
    _box(bm, cx, cy, cz + height * 0.6 + steeple_h * 0.2, height * 0.1, height * 0.1, steeple_h, 1)
    return _finalize("church", bm, [
        ("Urban_Church_Wall", (0.75, 0.7, 0.6, 1), 0.8, 0),
        ("Urban_Church_Roof", (0.4, 0.25, 0.15, 1), 0.9, 0),
    ])


@gin.configurable
def generate_stadium(center=(0, 0, 0), height=20):
    bm = bmesh.new()
    cx, cy, cz = center
    _cylinder(bm, cx, cy, cz, cz + height * 0.3, height * 0.3, segs=16, mat_idx=0)
    _cylinder(bm, cx, cy, cz + height * 0.3, cz + height, height * 0.3, segs=16, mat_idx=1)
    return _finalize("stadium", bm, [
        ("Urban_Stadium_Base", (0.5, 0.5, 0.55, 1), 0.7, 0.2),
        ("Urban_Stadium_Glass", (0.2, 0.3, 0.4, 1), 0.1, 0.3),
    ])


@gin.configurable
def generate_silo(center=(0, 0, 0), height=30):
    bm = bmesh.new()
    cx, cy, cz = center
    _cylinder(bm, cx, cy, cz, cz + height, height * 0.1, segs=14, mat_idx=0)
    dome = bm.verts.new(Vector((cx, cy, cz + height + height * 0.1)))
    ring = [bm.verts.new(Vector((cx + math.cos(i/14*math.pi*2)*height*0.1,
                                  cy + math.sin(i/14*math.pi*2)*height*0.1,
                                  cz + height))) for i in range(14)]
    for i in range(14):
        j = (i + 1) % 14
        bm.faces.new([ring[i], ring[j], dome]).material_index = 0
    return _finalize("silo", bm, [("Urban_Metal", (0.55, 0.5, 0.45, 1), 0.6, 0.7)])


@gin.configurable
def generate_wind_turbine(center=(0, 0, 0), height=60):
    bm = bmesh.new()
    cx, cy, cz = center
    segs = 6; base_r, top_r = 0.8, 0.2
    for i in range(segs):
        a1 = (i/segs)*math.pi*2; a2 = ((i+1)/segs)*math.pi*2
        v = [bm.verts.new(Vector((cx+math.cos(a)*r, cy+math.sin(a)*r, cz+z)))
             for a, r, z in [(a1, base_r, 0), (a2, base_r, 0),
                              (a2, top_r, height), (a1, top_r, height)]]
        bm.faces.new(v).material_index = 0
    blade_len = height * 0.35
    for i in range(3):
        a = i * math.pi * 2 / 3
        v = [bm.verts.new(Vector((cx + dx, cy + dy, cz + height + dz)))
             for dx, dy, dz in
             [(0, -0.05, 0), (0, 0.05, 0),
              (math.cos(a)*blade_len, math.sin(a)*blade_len, 1.5),
              (math.cos(a)*blade_len, math.sin(a)*blade_len, -0.5)]]
        bm.faces.new(v).material_index = 0
    return _finalize("wind_turbine", bm, [("Urban_Metal", (0.6, 0.6, 0.65, 1), 0.5, 0.8)])


@gin.configurable
def generate_lighthouse(center=(0, 0, 0), height=25):
    bm = bmesh.new()
    cx, cy, cz = center
    segs = 10; base_r, top_r = 1.2, 0.6
    for i in range(segs):
        a1 = (i/segs)*math.pi*2; a2 = ((i+1)/segs)*math.pi*2
        v = [bm.verts.new(Vector((cx+math.cos(a)*r, cy+math.sin(a)*r, cz+z)))
             for a, r, z in [(a1, base_r, 0), (a2, base_r, 0),
                              (a2, top_r, height), (a1, top_r, height)]]
        bm.faces.new(v).material_index = 0
    _cylinder(bm, cx, cy, cz + height, cz + height + 1, top_r * 0.5, segs=10, mat_idx=1)
    return _finalize("lighthouse", bm, [
        ("Urban_Lighthouse_Wall", (0.85, 0.82, 0.75, 1), 0.7, 0),
        ("Urban_Lighthouse_Light", (0.95, 0.9, 0.5, 1), 0.2, 0.5),
    ])


@gin.configurable
def generate_crane(center=(0, 0, 0), height=35):
    bm = bmesh.new()
    cx, cy, cz = center
    tower_h = height * 0.7
    _box(bm, cx, cy, cz + tower_h * 0.5, 0.3, 0.3, tower_h, 0)
    jib_len = height * 0.5
    _box(bm, cx + jib_len * 0.5, cy, cz + tower_h, jib_len, 0.1, 0.15, 0)
    _box(bm, cx - jib_len * 0.15, cy, cz + tower_h, jib_len * 0.3, 0.1, 0.1, 0)
    cab = bm.verts.new(Vector((cx, cy, cz + tower_h + 2)))
    for i in range(8):
        a = (i/8)*math.pi*2
        bv = bm.verts.new(Vector((cx + math.cos(a)*0.5, cy + math.sin(a)*0.5, cz + tower_h - 0.5)))
        tv = bm.verts.new(Vector((cx + math.cos(a)*0.5, cy + math.sin(a)*0.5, cz + tower_h + 0.5)))
        bm.faces.new([bv, tv, cab]).material_index = 0
    return _finalize("crane", bm, [("Urban_Metal", (0.7, 0.7, 0.15, 1), 0.5, 0.6)])


@gin.configurable
def place_landmarks(scene_bounds, regional_style, count=5, seed=42):
    random.seed(seed)
    landmarks = []
    if regional_style and regional_style.landmark_type_weights:
        types = list(regional_style.landmark_type_weights.keys())
    else:
        types = ["water_tower", "cell_tower"]
    generators = {
        "water_tower": (generate_water_tower, {"height": (15, 40)}),
        "cell_tower": (generate_cell_tower, {"height": (20, 60)}),
        "church": (generate_church, {"height": (10, 25)}),
        "stadium": (generate_stadium, {"height": (15, 30)}),
        "silo": (generate_silo, {"height": (20, 40)}),
        "wind_turbine": (generate_wind_turbine, {"height": (40, 80)}),
        "lighthouse": (generate_lighthouse, {"height": (15, 35)}),
        "crane": (generate_crane, {"height": (25, 50)}),
    }
    for _ in range(count):
        ltype = random.choice(types)
        if ltype not in generators:
            continue
        gen_fn, params = generators[ltype]
        x = random.uniform(-scene_bounds[0]/2, scene_bounds[0]/2)
        y = random.uniform(-scene_bounds[1]/2, scene_bounds[1]/2)
        h_range = params["height"]
        obj = gen_fn(center=(x, y, 0), height=random.uniform(*h_range))
        obj.name = f"landmark_{ltype}_{_}"
        landmarks.append(obj)
    return landmarks