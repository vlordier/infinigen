import math
from dataclasses import dataclass, field
import random
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.block_subdivision import BuildingLot


@dataclass
class DistrictTemplateConfig:
    internal_road_width: float = 8.0
    internal_sidewalk: bool = False
    lot_depth: float = 30.0
    lot_width: float = 20.0
    lot_min_area: float = 20.0
    irregularity: float = 0.0
    dead_end_chance: float = 0.0
    density: float = 0.5


@dataclass
class DistrictFill:
    road_segments: list[RoadSegment] = field(default_factory=list)
    building_lots: list[BuildingLot] = field(default_factory=list)


class BaseTemplate:
    name = "base"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        raise NotImplementedError


def register_template(cls):
    _TEMPLATE_REGISTRY[cls.name] = cls
    return cls


_TEMPLATE_REGISTRY = {}


def get_template(name: str):
    return _TEMPLATE_REGISTRY.get(name)


from infinigen.assets.urban.template_utils import make_grid_segments, bbox_lots, clip_segments_to_boundary


@register_template
class RectangularGridTemplate(BaseTemplate):
    name = "rectangular_grid"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = make_grid_segments(
            (x0, y0), (x1, y1),
            spacing=max(config.lot_depth, config.lot_width) * 2,
            road_type="local", width=config.internal_road_width,
            sidewalk=config.internal_sidewalk, rng=rng,
            irregularity=config.irregularity,
        )
        lots = bbox_lots(
            (x0 + 2, y0 + 2), (x1 - 2, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class OrganicGridTemplate(BaseTemplate):
    name = "organic_grid"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        spacing = max(config.lot_depth, config.lot_width) * 2
        segs = make_grid_segments(
            (x0, y0), (x1, y1), spacing=spacing,
            road_type="local", width=config.internal_road_width,
            sidewalk=config.internal_sidewalk,
            irregularity=config.irregularity, rng=rng,
        )
        segs = clip_segments_to_boundary(segs, boundary)
        dither = spacing * config.irregularity * 0.5 if config.irregularity else 0
        lots = bbox_lots(
            (x0 + dither + 2, y0 + dither + 2),
            (x1 - dither - 2, y1 - dither - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class MedievalOrganicTemplate(BaseTemplate):
    name = "medieval_organic"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = []
        n_nodes = rng.randint(3, 6)
        nodes = []
        for _ in range(n_nodes):
            nx = rng.uniform(x0 + 5, x1 - 5)
            ny = rng.uniform(y0 + 5, y1 - 5)
            nodes.append((nx, ny))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if rng.random() < 0.5:
                    segs.append(RoadSegment(
                        source=nodes[i], target=nodes[j],
                        road_type="local", lane_count=1, width=config.internal_road_width,
                        sidewalk=False,
                    ))
        if rng.random() < config.dead_end_chance:
            from_node = rng.choice(nodes)
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(5, 20)
            dead_end = (from_node[0] + dist * math.cos(angle),
                        from_node[1] + dist * math.sin(angle))
            segs.append(RoadSegment(
                source=from_node, target=dead_end,
                road_type="alley", lane_count=1, width=4.0, sidewalk=False,
            ))
        lots = _voronoi_lots(nodes, boundary, rng)
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


def _voronoi_lots(points, boundary, rng, n_samples=500):
    from infinigen.assets.urban.template_utils import _point_in_polygon
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    assignments = {}
    for _ in range(n_samples):
        px = rng.uniform(x0, x1)
        py = rng.uniform(y0, y1)
        if not _point_in_polygon((px, py), boundary):
            continue
        best = min(range(len(points)), key=lambda i: (px-points[i][0])**2 + (py-points[i][1])**2)
        assignments.setdefault(best, []).append((px, py))
    lots = []
    for region_id, pts in assignments.items():
        if len(pts) < 3:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
        hull = _convex_hull(sorted_pts)
        if len(hull) < 3:
            continue
        area = _polygon_area(hull)
        lots.append(BuildingLot(boundary=hull, area=area))
    return lots


def _convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def _polygon_area(verts):
    area = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % n]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0


@register_template
class SovietBlockTemplate(BaseTemplate):
    name = "soviet_block"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = x1 - x0
        bh = y1 - y0
        n_cols = max(1, int(bw / config.lot_width))
        n_rows = max(1, int(bh / config.lot_depth))
        segs = []
        for i in range(1, n_cols):
            x = x0 + i * bw / n_cols
            segs.append(RoadSegment(
                source=(x, y0), target=(x, y1),
                road_type="local", lane_count=2, width=config.internal_road_width,
                sidewalk=False,
            ))
        lots = []
        for r in range(n_rows):
            for c in range(n_cols):
                lx = x0 + c * bw / n_cols + 2
                ly = y0 + r * bh / n_rows + 2
                rx = x0 + (c + 1) * bw / n_cols - 2
                ry = y0 + (r + 1) * bh / n_rows - 2
                area = (rx - lx) * (ry - ly)
                if area >= config.lot_min_area:
                    lots.append(BuildingLot(
                        boundary=[(lx, ly), (rx, ly), (rx, ry), (lx, ry)],
                        area=area, building_type="industrial",
                    ))
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class SuburbanCulDeSacTemplate(BaseTemplate):
    name = "suburban_cul_de_sac"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = []
        cx = (x0 + x1) / 2
        spine = [(cx, y0 + 5), (cx, y1 - 5)]
        segs.append(RoadSegment(
            source=spine[0], target=spine[1],
            road_type="local", lane_count=2, width=config.internal_road_width,
            sidewalk=config.internal_sidewalk,
        ))
        for side in [-1, 1]:
            for i in range(rng.randint(2, 4)):
                t = rng.uniform(0.2, 0.8)
                sy = y0 + t * (y1 - y0)
                length = rng.uniform(config.lot_depth, config.lot_depth * 2)
                ex = cx + side * length
                segs.append(RoadSegment(
                    source=(cx, sy), target=(ex, sy),
                    road_type="local", lane_count=2, width=config.internal_road_width,
                    sidewalk=False,
                ))
        lots = bbox_lots(
            (x0 + 2, y0 + 2), (cx - 5, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        ) + bbox_lots(
            (cx + 5, y0 + 2), (x1 - 2, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class GardenPlotsTemplate(BaseTemplate):
    name = "garden_plots"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = []
        cx = (x0 + x1) / 2
        segs.append(RoadSegment(
            source=(cx, y0), target=(cx, y1),
            road_type="local", lane_count=1, width=config.internal_road_width,
            sidewalk=False,
        ))
        plot_width = config.lot_width
        n_plots = max(1, int((x1 - x0) / 2 / plot_width))
        for side in [-1, 1]:
            for i in range(n_plots):
                px = cx + side * (i * plot_width + 2)
                if px < x0 + 2 or px > x1 - 2:
                    continue
                segs.append(RoadSegment(
                    source=(px, y0), target=(px, y1),
                    road_type="alley", lane_count=1, width=3.0, sidewalk=False,
                ))
        lots = []
        for side in [-1, 1]:
            for i in range(n_plots):
                px = cx + side * (i * plot_width + 2)
                if px < x0 + 2 or px > x1 - 2:
                    continue
                lx = min(px, cx) if side == -1 else px
                rx = max(px, cx) if side == -1 else px + plot_width
                lots.append(BuildingLot(
                    boundary=[(lx, y0 + 2), (rx, y0 + 2), (rx, y1 - 2), (lx, y1 - 2)],
                    area=(rx - lx) * (y1 - y0 - 4),
                    building_type="residential",
                ))
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class SparseOrganicTemplate(BaseTemplate):
    name = "sparse_organic"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        lots = []
        n_lots = rng.randint(1, 3)
        for _ in range(n_lots):
            lx = rng.uniform(x0 + 5, x1 - config.lot_width - 5)
            ly = rng.uniform(y0 + 5, y1 - config.lot_depth - 5)
            w = config.lot_width * rng.uniform(0.8, 1.2)
            h = config.lot_depth * rng.uniform(0.8, 1.2)
            lots.append(BuildingLot(
                boundary=[(lx, ly), (lx + w, ly), (lx + w, ly + h), (lx, ly + h)],
                area=w * h,
                building_type="residential",
            ))
        return DistrictFill(road_segments=[], building_lots=lots)
