import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector

@gin.configurable
def generate_building_shell(footprint_polygon, height, material_name="concrete"):
    """Generate a simple building shell mesh from a 2D footprint polygon."""
    bm = bmesh.new()
    n_verts = len(footprint_polygon)
    bottom_verts = [bm.verts.new(Vector((p[0], p[1], 0))) for p in footprint_polygon]
    top_verts = [bm.verts.new(Vector((p[0], p[1], height))) for p in footprint_polygon]
    for i in range(n_verts):
        j = (i + 1) % n_verts
        bm.faces.new([bottom_verts[i], bottom_verts[j], top_verts[j], top_verts[i]])
    bm.faces.new(bottom_verts[::-1])
    bm.faces.new(top_verts)
    mesh = bpy.data.meshes.new("building_shell")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("building_shell", mesh)
    return obj

@gin.configurable
def generate_buildings_from_lots(lots, regional_style=None, seed=42):
    random.seed(seed)
    buildings = []
    for lot in lots:
        height = random.randint(3, 20) if regional_style is None else random.randint(*regional_style.building_height_range) * 3
        obj = generate_building_shell(lot.boundary, height)
        buildings.append(obj)
    return buildings
