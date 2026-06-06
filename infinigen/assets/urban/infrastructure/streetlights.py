import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector


def _get_material():
    name = "Urban_Streetlight"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.32, 1)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Metallic"].default_value = 0.6
    return mat


@gin.configurable
def place_streetlights(road_positions, spacing=30, height=8, seed=42):
    random.seed(seed)
    lights = []
    mat = _get_material()
    for rid, pos in enumerate(road_positions):
        bm = bmesh.new()

        # Pole: rectangular column (4 side faces)
        hw = 0.08
        pb = [bm.verts.new(Vector((pos[0]+dx, pos[1]+dy, 0))) for dx, dy in
              [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]]
        pt = [bm.verts.new(Vector((pos[0]+dx, pos[1]+dy, height))) for dx, dy in
              [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]]
        for j in range(4):
            k = (j + 1) % 4
            bm.faces.new([pb[j], pb[k], pt[k], pt[j]])

        # Lamp arm
        arm_len = 0.3
        ax, ay = pos[0] + arm_len, pos[1]
        a0 = bm.verts.new(Vector((pos[0]+hw, pos[1]-hw*0.5, height)))
        a1 = bm.verts.new(Vector((pos[0]+hw, pos[1]+hw*0.5, height)))
        a2 = bm.verts.new(Vector((ax, ay-hw*0.5, height)))
        a3 = bm.verts.new(Vector((ax, ay+hw*0.5, height)))
        bm.faces.new([a0, a1, a3, a2])

        # Lamp housing
        lw, lh, lz = 0.15, 0.1, 0.12
        lb = [bm.verts.new(Vector((ax+dx, ay+dy, height+dz))) for dx, dy, dz in
              [(-lw, -lw, 0), (lw, -lw, 0), (lw, lw, 0), (-lw, lw, 0),
               (-lw, -lw, lz), (lw, -lw, lz), (lw, lw, lz), (-lw, lw, lz)]]
        for j in range(4):
            k = (j + 1) % 4
            bm.faces.new([lb[j], lb[k], lb[4+k], lb[4+j]])
        bm.faces.new([lb[0], lb[3], lb[2], lb[1]])
        bm.faces.new([lb[4], lb[5], lb[6], lb[7]])

        mesh = bpy.data.meshes.new(f"streetlight_{rid}")
        bm.to_mesh(mesh); bm.free()
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(f"streetlight_{rid}", mesh)
        bpy.context.scene.collection.objects.link(obj)

        light_data = bpy.data.lights.new(f"streetlight_light_{rid}", "POINT")
        light_data.energy = 50
        light_data.color = (1.0, 0.95, 0.8)
        light_obj = bpy.data.objects.new(f"streetlight_light_{rid}", light_data)
        light_obj.location = Vector((ax, ay, height + 0.05))
        bpy.context.scene.collection.objects.link(light_obj)

        lights.append(obj)
    return lights
