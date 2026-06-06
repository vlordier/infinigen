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
