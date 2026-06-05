from dataclasses import dataclass, field
from infinigen.assets.urban.dcel import DCEL


@dataclass
class RoadSegment:
    source: tuple[float, float]
    target: tuple[float, float]
    road_type: str = "local"
    lane_count: int = 2
    width: float = 12.0
    sidewalk: bool = True
    length: float = 0.0


@dataclass
class CityArea:
    boundary: list[tuple[float, float]]
    area: float = 0.0


class GraphParser:
    def __init__(self, dcel: DCEL):
        self.dcel = dcel
        self.road_segments: list[RoadSegment] = []
        self.city_areas: list[CityArea] = []
        self._parse()

    def _parse(self):
        self._extract_road_segments()
        self._extract_city_areas()

    def _extract_road_segments(self):
        visited = set()
        edge_lengths = []
        for he in self.dcel.half_edges:
            he_id = id(he)
            if he_id in visited or (he.twin is not None and id(he.twin) in visited):
                continue
            if he.twin is None:
                continue
            visited.add(he_id)
            visited.add(id(he.twin))
            src = he.origin.position
            dst = he.twin.origin.position
            dx = dst[0] - src[0]
            dy = dst[1] - src[1]
            length = (dx * dx + dy * dy) ** 0.5
            edge_lengths.append(length)
        avg_length = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 10
        visited.clear()
        for he in self.dcel.half_edges:
            he_id = id(he)
            if he_id in visited or (he.twin is not None and id(he.twin) in visited):
                continue
            if he.twin is None:
                continue
            visited.add(he_id)
            visited.add(id(he.twin))
            src = he.origin.position
            dst = he.twin.origin.position
            dx = dst[0] - src[0]
            dy = dst[1] - src[1]
            length = (dx * dx + dy * dy) ** 0.5
            node0_degree = self._node_degree(he.origin)
            node1_degree = self._node_degree(he.twin.origin)
            avg_degree = (node0_degree + node1_degree) / 2
            road_type, lane_count, width, sidewalk = self._assign_road_type(
                length, avg_length, avg_degree
            )
            seg = RoadSegment(
                source=src,
                target=dst,
                road_type=road_type,
                lane_count=lane_count,
                width=width,
                sidewalk=sidewalk,
                length=length,
            )
            self.road_segments.append(seg)

    def _node_degree(self, node) -> int:
        start_he = node.half_edge
        if start_he is None:
            return 0
        count = 0
        he = start_he
        while True:
            count += 1
            he = he.twin.next if he.twin is not None else None
            if he is None or he is start_he:
                break
        return count

    def _assign_road_type(self, length, avg_length, avg_degree):
        if length > avg_length * 2.0 and avg_degree >= 3:
            return ("highway", 4, 24.0, False)
        elif length > avg_length * 1.2:
            return ("arterial", 2, 16.0, True)
        elif avg_degree <= 2 and length < avg_length * 0.5:
            return ("alley", 1, 5.0, False)
        else:
            return ("local", 2, 12.0, True)

    def _extract_city_areas(self):
        for face in self.dcel.faces:
            if face.is_boundary:
                continue
            he = face.half_edge
            if he is None:
                continue
            verts = []
            start = he
            while True:
                verts.append(he.origin.position)
                he = he.next
                if he is start or he is None:
                    break
            area = self._polygon_area(verts)
            self.city_areas.append(CityArea(boundary=verts, area=area))

    @staticmethod
    def _polygon_area(verts):
        area = 0.0
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
