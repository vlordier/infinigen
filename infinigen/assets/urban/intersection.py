import gin


@gin.configurable
class IntersectionMesher:
    def __init__(self, crosswalk_width=3.0, curb_radius=0.5):
        self.crosswalk_width = crosswalk_width
        self.curb_radius = curb_radius

    def mesh_intersections(self, nodes: dict,
                           road_ends: dict) -> list:
        import bpy
        objects = []
        for node_id, pos in nodes.items():
            ends = road_ends.get(node_id, [])
            if len(ends) < 3:
                continue
            obj = self._mesh_intersection(pos, ends)
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
