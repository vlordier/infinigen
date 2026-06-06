import gin


_MATERIALS_CACHE = {}


def _get_mat(name, color, roughness=0.8, metallic=0.0):
    import bpy
    if name in _MATERIALS_CACHE:
        return _MATERIALS_CACHE[name]
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
class IntersectionMesher:
    def __init__(self, crosswalk_width=3.0, curb_radius=0.5):
        self.crosswalk_width = crosswalk_width
        self.curb_radius = curb_radius

    def mesh_intersections(self, dcel, road_segments: list) -> list:
        import bpy
        roads_by_pos: dict[tuple[float, float], list] = {}
        for seg in road_segments:
            for pos, dx, dy in [
                (seg.source, seg.target[0] - seg.source[0], seg.target[1] - seg.source[1]),
                (seg.target, seg.source[0] - seg.target[0], seg.source[1] - seg.target[1]),
            ]:
                length = (dx * dx + dy * dy) ** 0.5
                if length > 0:
                    dx /= length
                    dy /= length
                roads_by_pos.setdefault(pos, []).append(
                    ((dx, dy), seg.width / 2.0, seg.road_type)
                )
        objects = []
        for node in dcel.nodes:
            ends = roads_by_pos.get(node.position, [])
            if len(ends) < 3:
                continue
            cx, cy = node.position
            obj = self._mesh_intersection((cx, cy), ends)
            if obj:
                objects.append(obj)
        return objects

    def _mesh_intersection(self, center: tuple[float, float],
                           road_ends: list):
        import bpy
        cx, cy = center
        all_verts = []
        for end_dir, half_width, _ in road_ends:
            dx, dy = end_dir
            px = -dy
            py = dx
            left = (cx + px * half_width, cy + py * half_width)
            right = (cx - px * half_width, cy - py * half_width)
            all_verts.extend([left, right])
        if len(all_verts) < 3:
            return None
        hull = self._convex_hull(all_verts)
        verts_3d = [(v[0], v[1], 0.0) for v in hull]
        faces = [list(range(len(verts_3d)))]
        mesh = bpy.data.meshes.new("intersection")
        mesh.from_pydata(verts_3d, [], faces)
        mesh.update()
        mesh.materials.append(_get_mat("Urban_Asphalt", (0.15, 0.15, 0.16, 1), roughness=0.9))
        obj = bpy.data.objects.new("intersection", mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    @staticmethod
    def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        def cross(o, a, b):
            return ((a[0] - o[0]) * (b[1] - o[1])
                    - (a[1] - o[1]) * (b[0] - o[0]))
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]
