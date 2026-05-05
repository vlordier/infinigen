import gin
import random
import bpy
import bmesh
from mathutils import Vector


@gin.configurable
def place_streetlights(road_positions, spacing=30, height=8, seed=42):
    random.seed(seed)
    lights = []
    for rid, positions in enumerate(road_positions):
        for i in range(0, len(positions)-1, max(1, int(spacing))):
            pos = positions[i]
            bm = bmesh.new()
            bv = [bm.verts.new(Vector((pos[0], pos[1], h))) for h in (0, height)]
            bm.faces.new(bv + [bv[1], bv[0]])
            top = height
            hv = [bm.verts.new(Vector((pos[0]+dx, pos[1]+dy, top+dz))) for dx, dy, dz in
                   [(-0.3, -0.3, 0), (0.3, -0.3, 0), (0.3, 0.3, 0.5), (-0.3, 0.3, 0.5)]]
            bm.faces.new(hv)
            mesh = bpy.data.meshes.new(f"streetlight_{rid}_{i}")
            bm.to_mesh(mesh); bm.free()
            obj = bpy.data.objects.new(f"streetlight_{rid}_{i}", mesh)
            lights.append(obj)
    return lights
