import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector


_MATERIALS = {}


def _wall_mat():
    name = "Urban_Building_Wall"
    if name in _MATERIALS:
        return _MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.72, 1)
        bsdf.inputs["Roughness"].default_value = 0.7
    _MATERIALS[name] = mat
    return mat


def _window_mat():
    name = "Urban_Window"
    if name in _MATERIALS:
        return _MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.15, 0.2, 0.25, 1)
        bsdf.inputs["Roughness"].default_value = 0.15
        bsdf.inputs["Metallic"].default_value = 0.4
    _MATERIALS[name] = mat
    return mat


def _door_mat():
    name = "Urban_Door"
    if name in _MATERIALS:
        return _MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.2, 0.13, 1)
        bsdf.inputs["Roughness"].default_value = 0.4
    _MATERIALS[name] = mat
    return mat


def _roof_mat():
    name = "Urban_Roof"
    if name in _MATERIALS:
        return _MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.35, 0.3, 0.25, 1)
        bsdf.inputs["Roughness"].default_value = 0.85
    _MATERIALS[name] = mat
    return mat


_PALETTES = [
    [(0.85, 0.83, 0.78), (0.7, 0.65, 0.58), (0.6, 0.55, 0.5)],
    [(0.9, 0.85, 0.72), (0.72, 0.6, 0.5), (0.55, 0.45, 0.4)],
    [(0.6, 0.55, 0.5), (0.5, 0.45, 0.4), (0.4, 0.35, 0.3)],
    [(0.8, 0.78, 0.74), (0.65, 0.6, 0.55), (0.5, 0.45, 0.42)],
    [(0.55, 0.55, 0.6), (0.45, 0.45, 0.5), (0.35, 0.35, 0.4)],
    [(0.6, 0.5, 0.45), (0.5, 0.4, 0.35), (0.4, 0.32, 0.28)],
    [(0.7, 0.7, 0.65), (0.6, 0.6, 0.55), (0.5, 0.5, 0.45)],
]


@gin.configurable
def generate_building_shell(footprint_polygon, height, name_suffix="", material_name="concrete"):
    rng = random.Random(hash(tuple(round(p[0], 1) for p in footprint_polygon)) & 0xFFFFFF)
    wall_color = rng.choice(_PALETTES)[0]
    roof_color = (wall_color[0] * 0.7, wall_color[1] * 0.7, wall_color[2] * 0.65)

    n_verts = len(footprint_polygon)
    n_floors = max(1, int(height / 3.5))
    floor_h = height / n_floors

    bm = bmesh.new()
    bv = [bm.verts.new((p[0], p[1], 0)) for p in footprint_polygon]
    tv = [bm.verts.new((p[0], p[1], height)) for p in footprint_polygon]

    for i in range(n_verts):
        j = (i + 1) % n_verts
        bm.faces.new([bv[i], bv[j], tv[j], tv[i]]).material_index = 0

    bm.faces.new(tv).material_index = 3
    bm.faces.new(bv[::-1]).material_index = 0

    win_h = 1.3
    win_w = 1.0
    margin_y = 0.6
    margin_x = 0.5

    for i in range(n_verts):
        p0 = footprint_polygon[i]
        p1 = footprint_polygon[(i + 1) % n_verts]
        ex, ey = p1[0] - p0[0], p1[1] - p0[1]
        edge_len = (ex * ex + ey * ey) ** 0.5
        if edge_len < 0.1:
            continue
        edx, edy = ex / edge_len, ey / edge_len
        nx, ny = -edy, edx

        cx = sum(p[0] for p in footprint_polygon) / n_verts
        cy = sum(p[1] for p in footprint_polygon) / n_verts
        mid_x = (p0[0] + p1[0]) / 2
        mid_y = (p0[1] + p1[1]) / 2
        if (mid_x - cx) * nx + (mid_y - cy) * ny < 0:
            nx, ny = -nx, -ny

        offset = 0.02
        for floor in range(n_floors):
            z_bot = floor * floor_h + margin_y
            z_top = z_bot + win_h
            if z_top > height - margin_y:
                break
            x_cursor = margin_x
            while x_cursor + win_w < edge_len - margin_x:
                v0 = bm.verts.new((p0[0] + edx * x_cursor + nx * offset, p0[1] + edy * x_cursor + ny * offset, z_bot))
                v1 = bm.verts.new((p0[0] + edx * (x_cursor + win_w) + nx * offset, p0[1] + edy * (x_cursor + win_w) + ny * offset, z_bot))
                v2 = bm.verts.new((p0[0] + edx * (x_cursor + win_w) + nx * offset, p0[1] + edy * (x_cursor + win_w) + ny * offset, z_top))
                v3 = bm.verts.new((p0[0] + edx * x_cursor + nx * offset, p0[1] + edy * x_cursor + ny * offset, z_top))
                bm.faces.new([v0, v1, v2, v3]).material_index = 1
                x_cursor += win_w + margin_x

    longest_i = 0
    longest_len = 0
    for i in range(n_verts):
        p0 = footprint_polygon[i]
        p1 = footprint_polygon[(i + 1) % n_verts]
        el = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        if el > longest_len:
            longest_len = el
            longest_i = i
    p0 = footprint_polygon[longest_i]
    p1 = footprint_polygon[(longest_i + 1) % n_verts]
    ex, ey = p1[0] - p0[0], p1[1] - p0[1]
    el = (ex * ex + ey * ey) ** 0.5
    edx, edy = ex / el, ey / el
    nx, ny = -edy, edx
    cx = sum(p[0] for p in footprint_polygon) / n_verts
    cy = sum(p[1] for p in footprint_polygon) / n_verts
    mid_x = (p0[0] + p1[0]) / 2
    mid_y = (p0[1] + p1[1]) / 2
    if (mid_x - cx) * nx + (mid_y - cy) * ny < 0:
        nx, ny = -nx, -ny
    offset = 0.02
    door_w = 1.4
    door_h = min(2.5, height * 0.4)
    door_x = el / 2 - door_w / 2
    v0 = bm.verts.new((p0[0] + edx * door_x + nx * offset, p0[1] + edy * door_x + ny * offset, 0))
    v1 = bm.verts.new((p0[0] + edx * (door_x + door_w) + nx * offset, p0[1] + edy * (door_x + door_w) + ny * offset, 0))
    v2 = bm.verts.new((p0[0] + edx * (door_x + door_w) + nx * offset, p0[1] + edy * (door_x + door_w) + ny * offset, door_h))
    v3 = bm.verts.new((p0[0] + edx * door_x + nx * offset, p0[1] + edy * door_x + ny * offset, door_h))
    bm.faces.new([v0, v1, v2, v3]).material_index = 2

    mesh_name = f"building_shell_{name_suffix}" if name_suffix else "building_shell"
    mesh = bpy.data.meshes.new(mesh_name)
    bm.to_mesh(mesh)
    bm.free()

    wall_mat = _wall_mat().copy()
    wall_bsdf = wall_mat.node_tree.nodes.get("Principled BSDF")
    if wall_bsdf:
        wall_bsdf.inputs["Base Color"].default_value = wall_color + (1.0,)

    roof_mat_inst = _roof_mat().copy()
    roof_bsdf = roof_mat_inst.node_tree.nodes.get("Principled BSDF")
    if roof_bsdf:
        roof_bsdf.inputs["Base Color"].default_value = roof_color + (1.0,)

    mesh.materials.append(wall_mat)
    mesh.materials.append(_window_mat())
    mesh.materials.append(_door_mat())
    mesh.materials.append(roof_mat_inst)

    obj_name = f"building_shell_{name_suffix}" if name_suffix else "building_shell"
    obj = bpy.data.objects.new(obj_name, mesh)
    return obj


@gin.configurable
def generate_buildings_from_lots(lots, regional_style=None, seed=42):
    random.seed(seed)
    buildings = []
    for idx, lot in enumerate(lots):
        if regional_style is not None:
            height = random.randint(*regional_style.building_height_range) * 3
        else:
            height = random.randint(3, 20)
        obj = generate_building_shell(lot.boundary, height, name_suffix=str(idx))
        bpy.context.scene.collection.objects.link(obj)
        buildings.append(obj)
    return buildings
