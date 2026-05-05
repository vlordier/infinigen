import random
import math
import bpy
import bmesh
from mathutils import Vector

def fracture_object(obj, intensity=0.5):
    """Apply Voronoi fracture to object mesh."""
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        n_fractures = int(5 + intensity * 30)
        for _ in range(n_fractures):
            if len(bm.verts) < 10:
                break
            center = random.choice([v.co for v in bm.verts])
            radius = random.uniform(0.1, 0.5) * intensity * obj.dimensions.length * 0.2
            for v in bm.verts:
                if (v.co - center).length < radius:
                    v.co += (v.co - center).normalized() * radius * 0.4 * intensity
        bm.to_mesh(obj.data)
        bm.free()
    except Exception:
        pass

def voronoi_shatter(obj, n_cells=20):
    """Simple Voronoi-like shatter — displaces vertices toward random cell centers."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    if len(bm.verts) == 0:
        bm.free()
        return
    centers = [Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))) * obj.dimensions.length * 0.3 for _ in range(n_cells)]
    for v in bm.verts:
        nearest = min(centers, key=lambda c: (v.co - c).length)
        v.co += (nearest - v.co) * 0.3
    bm.to_mesh(obj.data)
    bm.free()
