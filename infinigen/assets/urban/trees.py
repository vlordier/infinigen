import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector


_MATERIALS = {}


def _get_trunk_mat():
    if "Urban_Tree_Trunk" in _MATERIALS:
        return _MATERIALS["Urban_Tree_Trunk"]
    mat = bpy.data.materials.new("Urban_Tree_Trunk")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.35, 0.22, 0.12, 1)
        bsdf.inputs["Roughness"].default_value = 0.95
    _MATERIALS["Urban_Tree_Trunk"] = mat
    return mat


def _get_canopy_mat():
    name = "Urban_Tree_Canopy"
    if name in _MATERIALS:
        return _MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.2, 0.4, 0.15, 1)
        bsdf.inputs["Roughness"].default_value = 0.8
    _MATERIALS[name] = mat
    return mat


@gin.configurable
def make_tree(center=(0, 0, 0), trunk_height=2.5, trunk_radius=0.18, canopy_radius=1.2, species=0):
    bm = bmesh.new()
    cx, cy, cz = center

    trunk_segs = 8
    for i in range(trunk_segs):
        a1 = (i / trunk_segs) * math.pi * 2
        a2 = ((i + 1) / trunk_segs) * math.pi * 2
        v0 = bm.verts.new(Vector((cx + math.cos(a1) * trunk_radius, cy + math.sin(a1) * trunk_radius, cz)))
        v1 = bm.verts.new(Vector((cx + math.cos(a2) * trunk_radius, cy + math.sin(a2) * trunk_radius, cz)))
        v2 = bm.verts.new(Vector((cx + math.cos(a2) * trunk_radius, cy + math.sin(a2) * trunk_radius, cz + trunk_height)))
        v3 = bm.verts.new(Vector((cx + math.cos(a1) * trunk_radius, cy + math.sin(a1) * trunk_radius, cz + trunk_height)))
        bm.faces.new([v0, v1, v2, v3])

    canopy_h = canopy_radius * 1.5
    canopy_segs = 10
    canopy_verts = []
    for i in range(canopy_segs + 1):
        a = (i / canopy_segs) * math.pi * 2
        for z_off, r in [(0, canopy_radius * 0.9), (canopy_h * 0.5, canopy_radius), (canopy_h, canopy_radius * 0.6)]:
            x = cx + math.cos(a) * r
            y = cy + math.sin(a) * r
            z = cz + trunk_height + z_off
            canopy_verts.append(bm.verts.new(Vector((x, y, z))))
    for i in range(canopy_segs):
        for ring in range(2):
            a0 = i * 3 + ring
            a1 = i * 3 + ring + 1
            a2 = (i + 1) * 3 + ring + 1
            a3 = (i + 1) * 3 + ring
            bm.faces.new([canopy_verts[a0], canopy_verts[a1], canopy_verts[a2], canopy_verts[a3]])
    cap_segs = 8
    for i in range(cap_segs):
        a1 = (i / cap_segs) * math.pi * 2
        a2 = ((i + 1) / cap_segs) * math.pi * 2
        v0 = bm.verts.new(Vector((cx + math.cos(a1) * canopy_radius * 0.6, cy + math.sin(a1) * canopy_radius * 0.6, cz + trunk_height + canopy_h)))
        v1 = bm.verts.new(Vector((cx + math.cos(a2) * canopy_radius * 0.6, cy + math.sin(a2) * canopy_radius * 0.6, cz + trunk_height + canopy_h)))
        v2 = bm.verts.new(Vector((cx, cy, cz + trunk_height + canopy_h + canopy_radius * 0.3)))
        bm.faces.new([v0, v1, v2])

    mesh = bpy.data.meshes.new("tree")
    bm.to_mesh(mesh); bm.free()
    mesh.materials.append(_get_trunk_mat())
    mesh.materials.append(_get_canopy_mat())
    poly_list = list(mesh.polygons)
    for i, poly in enumerate(poly_list):
        c = poly.center
        if c.z < trunk_height - 0.1:
            poly.material_index = 0
        else:
            poly.material_index = 1
    obj = bpy.data.objects.new("tree", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@gin.configurable
def place_trees_along_roads(road_segments, spacing=8, offset=1.5, seed=42, city_size=None):
    rng = random.Random(seed)
    placed = []
    for seg in road_segments:
        if not seg.sidewalk:
            continue
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 1:
            continue
        dx, dy = dx / seg_len, dy / seg_len
        px, py = -dy, dx
        half_w = seg.width / 2.0
        for side in (-1, 1):
            n = max(1, int(seg_len / spacing))
            for i in range(n):
                t = (i + 0.5) / n
                if rng.random() < 0.3:
                    continue
                base_x = x1 + dx * seg_len * t
                base_y = y1 + dy * seg_len * t
                x = base_x + px * (half_w + offset) * side + (rng.random() - 0.5) * 0.3
                y = base_y + py * (half_w + offset) * side + (rng.random() - 0.5) * 0.3
                if city_size is not None:
                    if x < -2 or y < -2 or x > city_size + 2 or y > city_size + 2:
                        continue
                tree = make_tree(
                    center=(x, y, 0),
                    trunk_height=rng.uniform(2.0, 3.0),
                    trunk_radius=rng.uniform(0.15, 0.25),
                    canopy_radius=rng.uniform(1.0, 1.5),
                )
                placed.append(tree)
    return placed
