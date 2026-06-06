import gin
import math
import bpy
import bmesh
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.dcel import DCEL


_MATERIALS = {}


def _get_mat(name, color, roughness=0.8):
    if name in _MATERIALS:
        return _MATERIALS[name]
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color + (1.0,)
        bsdf.inputs["Roughness"].default_value = roughness
    _MATERIALS[name] = mat
    return mat


def _quad_on_road(seg: RoadSegment, center_offset: float, half_width: float, z: float, length: float):
    """Return 4 corner vertices of a small quad on the road surface."""
    x1, y1 = seg.source
    x2, y2 = seg.target
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 0.01:
        return None, None
    dx, dy = dx / seg_len, dy / seg_len
    px, py = -dy, dx
    cx, cy = (x1 + x2) / 2 + dx * center_offset, (y1 + y2) / 2 + dy * center_offset
    v0 = (cx - px * half_width + dx * length / 2, cy - py * half_width + dy * length / 2, z)
    v1 = (cx + px * half_width + dx * length / 2, cy + py * half_width + dy * length / 2, z)
    v2 = (cx + px * half_width - dx * length / 2, cy + py * half_width - dy * length / 2, z)
    v3 = (cx - px * half_width - dx * length / 2, cy - py * half_width - dy * length / 2, z)
    return (v0, v1, v2, v3), (dx, dy, px, py, seg_len)


@gin.configurable
class RoadMarkingMesher:
    def __init__(self, lane_line_width=0.15, dash_length=2.0, dash_gap=2.0,
                 crosswalk_strip_width=0.6, crosswalk_strip_gap=0.6, marking_z=0.02):
        self.lane_line_width = lane_line_width
        self.dash_length = dash_length
        self.dash_gap = dash_gap
        self.crosswalk_strip_width = crosswalk_strip_width
        self.crosswalk_strip_gap = crosswalk_strip_gap
        self.marking_z = marking_z

    def mesh_markings(self, road_segments: list) -> list:
        objects = []
        yellow = _get_mat("Urban_Marking_Yellow", (0.95, 0.85, 0.1))
        white = _get_mat("Urban_Marking_White", (0.95, 0.95, 0.95))
        for seg in road_segments:
            objs = self._mark_segment(seg, yellow, white)
            objects.extend(objs)
        return objects

    def _mark_segment(self, seg: RoadSegment, yellow, white):
        objects = []
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 0.5:
            return []

        px = -dy / seg_len
        py = dx / seg_len
        half_width = seg.width / 2.0

        if seg.road_type in ("highway", "arterial"):
            lane_centers = []
            for i in range(1, max(2, seg.lane_count)):
                t = (i / max(1, seg.lane_count)) - 0.5
                lane_centers.append(t * seg.width)
            for lc in lane_centers:
                is_center = abs(lc) < 0.1
                if is_center:
                    pass
                else:
                    self._make_dashed_line(seg, lc, self.lane_line_width, yellow if is_center else white, objects)
            self._make_edge_lines(seg, half_width, white, objects)
        else:
            self._make_edge_lines(seg, half_width, white, objects)
        return objects

    def _make_dashed_line(self, seg, lateral_offset, line_width, mat, objects):
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        dx, dy = dx / seg_len, dy / seg_len
        px, py = -dy, dx
        period = self.dash_length + self.dash_gap
        pos = 0.0
        while pos < seg_len:
            start = pos
            end = min(pos + self.dash_length, seg_len)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            t0 = (start - seg_len / 2)
            t1 = (end - seg_len / 2)
            cx0, cy0 = mid_x + dx * t0, mid_y + dy * t0
            cx1, cy1 = mid_x + dx * t1, mid_y + dy * t1
            v0 = (cx0 + px * lateral_offset - px * line_width / 2, cy0 + py * lateral_offset - py * line_width / 2, self.marking_z)
            v1 = (cx0 + px * lateral_offset + px * line_width / 2, cy0 + py * lateral_offset + py * line_width / 2, self.marking_z)
            v2 = (cx1 + px * lateral_offset + px * line_width / 2, cy1 + py * lateral_offset + py * line_width / 2, self.marking_z)
            v3 = (cx1 + px * lateral_offset - px * line_width / 2, cy1 + py * lateral_offset - py * line_width / 2, self.marking_z)
            objects.append(self._make_quad([v0, v1, v2, v3], "marking_dash", mat))
            pos += period

    def _make_edge_lines(self, seg, half_width, mat, objects):
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 0.5:
            return
        dx, dy = dx / seg_len, dy / seg_len
        px, py = -dy, dx
        edge_off = half_width - 0.2
        self._make_solid_line(seg, edge_off, 0.1, mat, objects)
        self._make_solid_line(seg, -edge_off, 0.1, mat, objects)

    def _make_solid_line(self, seg, lateral_offset, line_width, mat, objects):
        x1, y1 = seg.source
        x2, y2 = seg.target
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        dx, dy = dx / seg_len, dy / seg_len
        px, py = -dy, dx
        v0 = (x1 + px * (lateral_offset - line_width / 2), y1 + py * (lateral_offset - line_width / 2), self.marking_z)
        v1 = (x1 + px * (lateral_offset + line_width / 2), y1 + py * (lateral_offset + line_width / 2), self.marking_z)
        v2 = (x2 + px * (lateral_offset + line_width / 2), y2 + py * (lateral_offset + line_width / 2), self.marking_z)
        v3 = (x2 + px * (lateral_offset - line_width / 2), y2 + py * (lateral_offset - line_width / 2), self.marking_z)
        objects.append(self._make_quad([v0, v1, v2, v3], "marking_edge", mat))

    def _make_quad(self, verts, name, mat):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], [[0, 1, 2, 3]])
        mesh.update()
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def mesh_crosswalks(self, dcel, road_segments: list) -> list:
        objects = []
        white = _get_mat("Urban_Marking_White", (0.95, 0.95, 0.95))
        seg_by_pos: dict[tuple[float, float], list] = {}
        for seg in road_segments:
            for pos in (seg.source, seg.target):
                seg_by_pos.setdefault(pos, []).append(seg)
        for node in dcel.nodes:
            segs = seg_by_pos.get(node.position, [])
            if len(segs) < 2:
                continue
            for seg in segs:
                self._make_crosswalk_at_node(seg, node.position, white, objects)
        return objects

    def _make_crosswalk_at_node(self, seg, node_pos, mat, objects):
        x1, y1 = seg.source
        x2, y2 = seg.target
        node_is_source = (abs(x1 - node_pos[0]) < 0.1 and abs(y1 - node_pos[1]) < 0.1)
        if node_is_source:
            base_x, base_y = x1, y1
            dx, dy = x2 - x1, y2 - y1
        else:
            base_x, base_y = x2, y2
            dx, dy = x1 - x2, y1 - y2
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 0.1:
            return
        dx, dy = dx / seg_len, dy / seg_len
        px, py = -dy, dx
        half_width = seg.width / 2.0
        offset = 0.5
        x_cursor = -offset - 0.5
        n_strips = max(1, int(seg.width / (self.crosswalk_strip_width + self.crosswalk_strip_gap)))
        strip_length = 1.5
        start_x = -half_width
        for i in range(n_strips):
            sx = start_x + i * (self.crosswalk_strip_width + self.crosswalk_strip_gap)
            v0 = (base_x + dx * offset + px * sx, base_y + dy * offset + py * sx, self.marking_z)
            v1 = (base_x + dx * offset + px * (sx + self.crosswalk_strip_width), base_y + dy * offset + py * (sx + self.crosswalk_strip_width), self.marking_z)
            v2 = (base_x + dx * (offset + strip_length) + px * (sx + self.crosswalk_strip_width), base_y + dy * (offset + strip_length) + py * (sx + self.crosswalk_strip_width), self.marking_z)
            v3 = (base_x + dx * (offset + strip_length) + px * sx, base_y + dy * (offset + strip_length) + py * sx, self.marking_z)
            objects.append(self._make_quad([v0, v1, v2, v3], "crosswalk_strip", mat))
