import math
import bpy
import bmesh
from mathutils import Vector

def apply_crater(obj, center_xy, radius, depth):
    """Apply spherical depression to terrain mesh at given (x,y) location."""
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        cx, cy = center_xy
        for v in bm.verts:
            dx = v.co.x - cx
            dy = v.co.y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < radius:
                falloff = 1.0 - dist / radius
                displacement = depth * (falloff * falloff) * math.cos(falloff * math.pi * 0.5)
                v.co.z -= displacement
                push = (dist / radius) * radius * 0.15 * falloff
                if dist > 0.01:
                    v.co.x += dx / dist * push
                    v.co.y += dy / dist * push
        bm.to_mesh(obj.data)
        bm.free()
    except Exception:
        pass
