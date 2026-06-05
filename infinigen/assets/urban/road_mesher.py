import gin
import math
from infinigen.assets.urban.graph_parser import RoadSegment


@gin.configurable
class RoadMesher:
    def __init__(self, vertex_distance=2.0, max_road_length=50.0,
                 lane_width=3.5, extra_lane_width=1.0,
                 sidewalk_width=1.5, sidewalk_height=0.15,
                 wall_height=0.6):
        self.vertex_distance = vertex_distance
        self.max_road_length = max_road_length
        self.lane_width = lane_width
        self.extra_lane_width = extra_lane_width
        self.sidewalk_width = sidewalk_width
        self.sidewalk_height = sidewalk_height
        self.wall_height = wall_height

    def mesh_roads(self, road_segments: list) -> list:
        import bpy
        objects = []
        for seg in road_segments:
            obj = self._mesh_road_segment(seg)
            if obj:
                objects.append(obj)
        return objects

    def mesh_sidewalks(self, road_segments: list) -> list:
        objects = []
        for seg in road_segments:
            if not seg.sidewalk:
                continue
            obj = self._mesh_sidewalk(seg, side="left")
            if obj:
                objects.append(obj)
            obj = self._mesh_sidewalk(seg, side="right")
            if obj:
                objects.append(obj)
        return objects

    def _mesh_road_segment(self, seg: RoadSegment):
        import bpy
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.01:
            return None
        px = -dy / length
        py = dx / length
        half_width = seg.width / 2.0
        n = max(1, int(length / self.vertex_distance))
        verts = []
        faces = []
        for i in range(n + 1):
            t = i / n
            cx = x1 + dx * t
            cy = y1 + dy * t
            lx = cx + px * half_width
            ly = cy + py * half_width
            rx = cx - px * half_width
            ry = cy - py * half_width
            verts.append((lx, ly, 0.0))
            verts.append((rx, ry, 0.0))
        for i in range(n):
            a = i * 2
            b = i * 2 + 1
            c = (i + 1) * 2 + 1
            d = (i + 1) * 2
            faces.append([a, b, c, d])
        name = f"road_{seg.road_type}"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def _mesh_sidewalk(self, seg: RoadSegment, side: str):
        import bpy
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.01:
            return None
        px = -dy / length
        py = dx / length
        half_width = seg.width / 2.0
        offset = half_width + 0.3
        sign = -1 if side == "left" else 1
        n = max(1, int(length / self.vertex_distance))
        verts = []
        faces = []
        for i in range(n + 1):
            t = i / n
            cx = x1 + dx * t
            cy = y1 + dy * t
            inner_x = cx + px * (offset * sign)
            inner_y = cy + py * (offset * sign)
            outer_x = cx + px * (offset + self.sidewalk_width) * sign
            outer_y = cy + py * (offset + self.sidewalk_width) * sign
            verts.append((inner_x, inner_y, self.sidewalk_height))
            verts.append((outer_x, outer_y, self.sidewalk_height))
        for i in range(n):
            a = i * 2
            b = i * 2 + 1
            c = (i + 1) * 2 + 1
            d = (i + 1) * 2
            faces.append([a, b, c, d])
        name = f"sidewalk_{side}"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj
