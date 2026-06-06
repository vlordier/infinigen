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


@gin.configurable
def generate_water_tower(center=(0, 0, 0), height=25, radius=3):
    bm = bmesh.new()
    cx, cy, cz = center
    leg_positions = [(-radius*0.5, -radius*0.5), (radius*0.5, -radius*0.5),
                     (-radius*0.5, radius*0.5), (radius*0.5, radius*0.5)]
    for lx, ly in leg_positions:
        v0 = bm.verts.new(Vector((lx + cx, ly + cy, cz)))
        v1 = bm.verts.new(Vector((lx + cx, ly + cy, cz + height * 0.7)))
        v2 = bm.verts.new(Vector((lx + cx + 0.2, ly + cy, cz + height * 0.7)))
        v3 = bm.verts.new(Vector((lx + cx + 0.2, ly + cy, cz)))
        bm.faces.new([v0, v1, v2, v3])
    tank_base = height * 0.7
    tank_h = height * 0.3
    segs = 12
    cyl_verts = []
    for i in range(segs + 1):
        a = (i / segs) * math.pi * 2
        bv = bm.verts.new(Vector((cx + math.cos(a)*radius, cy + math.sin(a)*radius, cz + tank_base)))
        tv = bm.verts.new(Vector((cx + math.cos(a)*radius, cy + math.sin(a)*radius, cz + tank_base + tank_h)))
        cyl_verts.append((bv, tv))
    for i in range(segs):
        j = (i + 1) % segs
        bm.faces.new([cyl_verts[i][0], cyl_verts[j][0], cyl_verts[j][1], cyl_verts[i][1]])
    mesh = bpy.data.meshes.new("water_tower")
    bm.to_mesh(mesh); bm.free()
    mesh.materials.append(_get_mat("Urban_Metal", (0.5, 0.45, 0.4, 1), metallic=0.7))
    obj = bpy.data.objects.new("water_tower", mesh)
    obj.location = Vector(center)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@gin.configurable
def generate_cell_tower(center=(0, 0, 0), height=40):
    bm = bmesh.new()
    cx, cy, cz = center
    segs = 8
    base_r, top_r = 1.5, 0.5
    for i in range(segs):
        a1 = (i/segs)*math.pi*2; a2 = ((i+1)/segs)*math.pi*2
        b1x = cx+math.cos(a1)*base_r; b1y = cy+math.sin(a1)*base_r
        b2x = cx+math.cos(a2)*base_r; b2y = cy+math.sin(a2)*base_r
        t1x = cx+math.cos(a1)*top_r; t1y = cy+math.sin(a1)*top_r
        t2x = cx+math.cos(a2)*top_r; t2y = cy+math.sin(a2)*top_r
        v = [bm.verts.new(Vector((x, y, z))) for x, y, z in [
            (b1x, b1y, cz), (b2x, b2y, cz), (t2x, t2y, cz+height), (t1x, t1y, cz+height)]]
        bm.faces.new(v)
    for i in range(3):
        a = (i/3)*math.pi*2
        px = cx + math.cos(a)*top_r*3
        py = cy + math.sin(a)*top_r*3
        v = [bm.verts.new(Vector((x, y, z))) for x, y, z in [
            (px-0.5, py-0.1, cz+height), (px+0.5, py-0.1, cz+height),
            (px+0.5, py+0.1, cz+height+3), (px-0.5, py+0.1, cz+height+3)]]
        bm.faces.new(v)
    mesh = bpy.data.meshes.new("cell_tower")
    bm.to_mesh(mesh); bm.free()
    mesh.materials.append(_get_mat("Urban_Metal", (0.5, 0.5, 0.55, 1), metallic=0.8))
    obj = bpy.data.objects.new("cell_tower", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@gin.configurable
def place_landmarks(scene_bounds, regional_style, count=5, seed=42):
    random.seed(seed)
    landmarks = []
    types = list(regional_style.landmark_type_weights.keys()) if regional_style else ["water_tower", "cell_tower"]
    for _ in range(count):
        ltype = random.choice(types)
        x = random.uniform(-scene_bounds[0]/2, scene_bounds[0]/2)
        y = random.uniform(-scene_bounds[1]/2, scene_bounds[1]/2) if len(scene_bounds) > 1 else random.uniform(-500, 500)
        if ltype == "water_tower":
            obj = generate_water_tower(center=(x, y, 0), height=random.uniform(15, 40))
        elif ltype == "cell_tower":
            obj = generate_cell_tower(center=(x, y, 0), height=random.uniform(20, 60))
        obj.name = f"landmark_{ltype}_{_}"
        landmarks.append(obj)
    return landmarks
