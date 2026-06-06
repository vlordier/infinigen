import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector


def _get_material():
    name = "Urban_Building"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.72, 1)
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


@gin.configurable
def generate_building_shell(footprint_polygon, height, name_suffix="", material_name="concrete"):
    bm = bmesh.new()
    n_verts = len(footprint_polygon)
    bottom_verts = [bm.verts.new(Vector((p[0], p[1], 0))) for p in footprint_polygon]
    top_verts = [bm.verts.new(Vector((p[0], p[1], height))) for p in footprint_polygon]
    for i in range(n_verts):
        j = (i + 1) % n_verts
        bm.faces.new([bottom_verts[i], bottom_verts[j], top_verts[j], top_verts[i]])
    bm.faces.new(bottom_verts[::-1])
    bm.faces.new(top_verts)
    mesh_name = f"building_shell_{name_suffix}" if name_suffix else "building_shell"
    mesh = bpy.data.meshes.new(mesh_name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(_get_material())
    obj_name = f"building_shell_{name_suffix}" if name_suffix else "building_shell"
    obj = bpy.data.objects.new(obj_name, mesh)
    return obj


@gin.configurable
def generate_buildings_from_lots(lots, regional_style=None, seed=42):
    random.seed(seed)
    buildings = []
    for idx, lot in enumerate(lots):
        height = random.randint(3, 20) if regional_style is None else random.randint(*regional_style.building_height_range) * 3
        obj = generate_building_shell(lot.boundary, height, name_suffix=str(idx))
        bpy.context.scene.collection.objects.link(obj)
        buildings.append(obj)
    return buildings
