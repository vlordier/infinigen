import gin
import random
import math
import bpy
import bmesh
from mathutils import Vector


_MATERIALS = {}


_CAR_COLORS = [
    (0.85, 0.1, 0.1),
    (0.1, 0.3, 0.85),
    (0.95, 0.95, 0.9),
    (0.1, 0.1, 0.15),
    (0.4, 0.45, 0.5),
    (0.15, 0.4, 0.2),
    (0.85, 0.75, 0.1),
    (0.6, 0.6, 0.65),
]


def _get_car_paint(color):
    name = f"Car_Paint_{int(color[0]*100)}_{int(color[1]*100)}_{int(color[2]*100)}"
    if name in _MATERIALS:
        return _MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color + (1.0,)
        bsdf.inputs["Roughness"].default_value = 0.3
        bsdf.inputs["Metallic"].default_value = 0.5
    _MATERIALS[name] = mat
    return mat


def _get_window_mat():
    if "Car_Window" in _MATERIALS:
        return _MATERIALS["Car_Window"]
    mat = bpy.data.materials.new("Car_Window")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.1, 0.12, 0.18, 1)
        bsdf.inputs["Roughness"].default_value = 0.05
        bsdf.inputs["Metallic"].default_value = 0.3
    _MATERIALS["Car_Window"] = mat
    return mat


def _get_tire_mat():
    if "Car_Tire" in _MATERIALS:
        return _MATERIALS["Car_Tire"]
    mat = bpy.data.materials.new("Car_Tire")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1)
        bsdf.inputs["Roughness"].default_value = 0.9
    _MATERIALS["Car_Tire"] = mat
    return mat


def _box(bm, x, y, z, sx, sy, sz, mat_idx):
    hw, hh, hd = sx / 2, sy / 2, sz / 2
    v = [
        bm.verts.new(Vector((x - hw, y - hh, z - hd))),
        bm.verts.new(Vector((x + hw, y - hh, z - hd))),
        bm.verts.new(Vector((x + hw, y + hh, z - hd))),
        bm.verts.new(Vector((x - hw, y + hh, z - hd))),
        bm.verts.new(Vector((x - hw, y - hh, z + hd))),
        bm.verts.new(Vector((x + hw, y - hh, z + hd))),
        bm.verts.new(Vector((x + hw, y + hh, z + hd))),
        bm.verts.new(Vector((x - hw, y + hh, z + hd))),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    out = []
    for fi in faces:
        f = bm.faces.new([v[i] for i in fi])
        f.material_index = mat_idx
        out.append(f)
    return out


@gin.configurable
def make_car(center=(0, 0, 0), heading=0.0, length=4.0, width=1.8, height=1.4):
    bm = bmesh.new()
    body_color = random.choice(_CAR_COLORS)

    _box(bm, 0, 0, 0.6, length, width, 0.8, 0)
    _box(bm, 0, 0, 1.4, length * 0.6, width * 0.95, 0.7, 1)

    wheel_offset_x = length * 0.35
    wheel_offset_y = width * 0.45
    wheel_z = 0.35
    for sign in (-1, 1):
        _box(bm, sign * wheel_offset_x, wheel_offset_y, wheel_z, 0.7, 0.3, 0.7, 2)
        _box(bm, sign * wheel_offset_x, -wheel_offset_y, wheel_z, 0.7, 0.3, 0.7, 2)

    mesh = bpy.data.meshes.new("car")
    bm.to_mesh(mesh); bm.free()
    mesh.materials.append(_get_car_paint(body_color))
    mesh.materials.append(_get_window_mat())
    mesh.materials.append(_get_tire_mat())

    obj = bpy.data.objects.new("car", mesh)
    obj.location = Vector(center)
    obj.rotation_euler = (0, 0, heading)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@gin.configurable
def place_parked_cars(road_segments, density=0.4, offset=0.0, seed=42, city_bounds=None):
    rng = random.Random(seed)
    placed = []
    for seg in road_segments:
        if seg.road_type == "highway":
            continue
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 4:
            continue
        dx, dy = dx / seg_len, dy / seg_len
        px, py = -dy, dx
        half_w = seg.width / 2.0
        n = max(1, int(seg_len / 5))
        for i in range(n):
            if rng.random() > density:
                continue
            t = (i + 0.5) / n
            side = rng.choice((-1, 1))
            x = x1 + dx * seg_len * t + px * (half_w + offset) * side
            y = y1 + dy * seg_len * t + py * (half_w + offset) * side
            if city_bounds is not None:
                bx_min, bx_max, by_min, by_max = city_bounds
                if x < bx_min or x > bx_max or y < by_min or y > by_max:
                    continue
            heading = math.atan2(dy, dx) if side == 1 else math.atan2(dy, dx) + math.pi
            car = make_car(
                center=(x, y, 0),
                heading=heading,
                length=rng.uniform(3.8, 4.6),
                width=rng.uniform(1.7, 1.9),
                height=rng.uniform(1.3, 1.5),
            )
            placed.append(car)
    return placed
