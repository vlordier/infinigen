import math
import random
from dataclasses import dataclass, field
from infinigen.assets.urban.graph_parser import RoadSegment


@dataclass
class BlockFace:
    boundary: list[tuple[float, float]]
    zone_id: str
    connection_nodes: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class CitySkeleton:
    road_segments: list[RoadSegment]
    blocks: list[BlockFace]


class RadialGenerator:
    @staticmethod
    def generate(size: float, n_radials: int = 8, n_rings: int = 4,
                 irregularity: float = 0.15, seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        cx = cy = size / 2
        max_r = size * 0.45
        road_segments = []
        blocks = []
        ring_distances = [max_r * (i + 1) / n_rings for i in range(n_rings)]
        base_angles = [2 * math.pi * i / n_radials for i in range(n_radials)]
        angles = [a + rng.uniform(-irregularity, irregularity) for a in base_angles]
        radials = []
        for ring_i in range(n_rings + 1):
            r = 0 if ring_i == 0 else ring_distances[ring_i - 1]
            ring = []
            for a in angles:
                jitter = r * rng.uniform(-irregularity * 0.3, irregularity * 0.3) if r > 0 else 0
                eff_r = r + jitter
                x = cx + eff_r * math.cos(a)
                y = cy + eff_r * math.sin(a)
                ring.append((x, y))
            radials.append(ring)
        for ring_i in range(1, n_rings + 1):
            for radial_i in range(n_radials):
                next_i = (radial_i + 1) % n_radials
                segment = RoadSegment(
                    source=radials[ring_i - 1][radial_i],
                    target=radials[ring_i][radial_i],
                    road_type="arterial" if ring_i <= 2 else "local",
                    lane_count=4 if ring_i <= 1 else 2,
                    width=24.0 if ring_i <= 1 else 12.0,
                    sidewalk=True,
                )
                road_segments.append(segment)
        for ring_i in range(n_rings):
            for radial_i in range(n_radials):
                next_i = (radial_i + 1) % n_radials
                a = radials[ring_i][radial_i]
                b = radials[ring_i][next_i]
                c = radials[ring_i + 1][next_i]
                d = radials[ring_i + 1][radial_i]
                segment = RoadSegment(
                    source=a, target=b,
                    road_type="ring",
                    lane_count=2, width=12.0, sidewalk=True,
                )
                road_segments.append(segment)
                if ring_i <= 2:
                    zone_id = "core" if ring_i == 0 else "inner" if ring_i == 1 else "outer"
                else:
                    zone_id = "outer"
                blocks.append(BlockFace(
                    boundary=[a, b, c, d],
                    zone_id=zone_id,
                ))
        boundary_ring = radials[-1]
        boundary_segments = []
        for i in range(n_radials):
            s = boundary_ring[i]
            t = boundary_ring[(i + 1) % n_radials]
            boundary_segments.append(RoadSegment(
                source=s, target=t, road_type="boundary",
                lane_count=2, width=12.0, sidewalk=True,
            ))
        road_segments.extend(boundary_segments)
        return CitySkeleton(road_segments=road_segments, blocks=blocks)


class GridGenerator:
    @staticmethod
    def generate(size: float, rows: int = 5, cols: int = 5,
                 irregularity: float = 0.0, seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        spacing_x = size / cols
        spacing_y = size / rows
        nodes = {}
        segs = []
        blocks = []
        for r in range(rows + 1):
            for c in range(cols + 1):
                jx = rng.uniform(-irregularity * spacing_x, irregularity * spacing_x) if irregularity > 0 else 0
                jy = rng.uniform(-irregularity * spacing_y, irregularity * spacing_y) if irregularity > 0 else 0
                x = c * spacing_x + jx
                y = r * spacing_y + jy
                nodes[(c, r)] = (x, y)
        for r in range(rows + 1):
            for c in range(cols):
                segs.append(RoadSegment(
                    source=nodes[(c, r)], target=nodes[(c + 1, r)],
                    road_type="local", lane_count=2, width=12.0, sidewalk=True,
                ))
        for c in range(cols + 1):
            for r in range(rows):
                segs.append(RoadSegment(
                    source=nodes[(c, r)], target=nodes[(c, r + 1)],
                    road_type="local", lane_count=2, width=12.0, sidewalk=True,
                ))
        for r in range(rows):
            for c in range(cols):
                blocks.append(BlockFace(
                    boundary=[nodes[(c, r)], nodes[(c+1, r)], nodes[(c+1, r+1)], nodes[(c, r+1)]],
                    zone_id="inner",
                ))
        return CitySkeleton(road_segments=segs, blocks=blocks)


class OrganicSpineGenerator:
    @staticmethod
    def generate(size: float, n_branches: int = 8, irregularity: float = 0.4,
                 seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        segs = []
        blocks = []
        cx = size * 0.5
        cy = size * 0.5
        spine_pts = []
        x, y = cx - size * 0.3, cy
        for i in range(6):
            x += size * 0.12 + rng.uniform(-size * 0.03, size * 0.03)
            y += rng.uniform(-size * 0.05, size * 0.05)
            y = max(size * 0.1, min(size * 0.9, y))
            spine_pts.append((x, y))
        for i in range(len(spine_pts) - 1):
            segs.append(RoadSegment(
                source=spine_pts[i], target=spine_pts[i+1],
                road_type="arterial", lane_count=2, width=16.0, sidewalk=True,
            ))
        for i in range(0, len(spine_pts), max(1, len(spine_pts) // n_branches)):
            bx, by = spine_pts[i]
            angle = rng.uniform(-math.pi * 0.4, math.pi * 0.4)
            if rng.random() < 0.3:
                angle += math.pi
            length = rng.uniform(size * 0.1, size * 0.3)
            ex = bx + length * math.cos(angle)
            ey = by + length * math.sin(angle)
            segs.append(RoadSegment(
                source=(bx, by), target=(ex, ey),
                road_type="local", lane_count=2, width=12.0, sidewalk=True,
            ))
            if rng.random() < 0.4:
                lx = ex + rng.uniform(-20, 20)
                ly = ey + rng.uniform(-20, 20)
                segs.append(RoadSegment(
                    source=(ex, ey), target=(lx, ly),
                    road_type="alley", lane_count=1, width=5.0, sidewalk=False,
                ))
        blocks.append(BlockFace(
            boundary=[(0, 0), (size, 0), (size, size), (0, size)],
            zone_id="inner",
        ))
        return CitySkeleton(road_segments=segs, blocks=blocks)


class SingleSpineGenerator:
    @staticmethod
    def generate(size: float, n_lanes: int = 6, seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        segs = []
        blocks = []
        spine_y = size * 0.5
        jitter = size * 0.02
        pts = []
        for i in range(4):
            x = size * (i + 0.5) / 4
            y = spine_y + rng.uniform(-jitter, jitter)
            pts.append((x, y))
        spine_pts = [(0, spine_y)] + pts + [(size, spine_y)]
        for i in range(len(spine_pts) - 1):
            segs.append(RoadSegment(
                source=spine_pts[i], target=spine_pts[i+1],
                road_type="local", lane_count=2, width=12.0, sidewalk=True,
            ))
        lane_spacing = size * 0.7 / n_lanes
        for i in range(1, len(spine_pts) - 1):
            sx, sy = spine_pts[i]
            for side in [-1, 1]:
                for li in range(n_lanes // 2):
                    ly = sy + side * (li + 1) * lane_spacing
                    if ly < 0 or ly > size:
                        continue
                    segs.append(RoadSegment(
                        source=(sx, sy), target=(sx, ly),
                        road_type="alley", lane_count=1, width=5.0, sidewalk=False,
                    ))
        blocks.append(BlockFace(
            boundary=[(0, 0), (size, 0), (size, size), (0, size)],
            zone_id="outer",
        ))
        return CitySkeleton(road_segments=segs, blocks=blocks)
