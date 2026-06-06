import math
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.block_subdivision import BuildingLot


def make_grid_segments(bottom_left, top_right, spacing, road_type="local",
                       width=8.0, sidewalk=False, irregularity=0.0, rng=None):
    x0, y0 = bottom_left
    x1, y1 = top_right
    segs = []
    cols = max(1, int((x1 - x0) / spacing))
    rows = max(1, int((y1 - y0) / spacing))
    for c in range(cols + 1):
        x = x0 + c * spacing
        if rng and irregularity:
            x += rng.uniform(-irregularity, irregularity)
        segs.append(RoadSegment(
            source=(x, y0), target=(x, y1),
            road_type=road_type, lane_count=2, width=width, sidewalk=sidewalk,
        ))
    for r in range(rows + 1):
        y = y0 + r * spacing
        if rng and irregularity:
            y += rng.uniform(-irregularity, irregularity)
        segs.append(RoadSegment(
            source=(x0, y), target=(x1, y),
            road_type=road_type, lane_count=2, width=width, sidewalk=sidewalk,
        ))
    return segs


def clip_segments_to_boundary(segments, boundary):
    clipped = []
    for seg in segments:
        segs = _clip_line_to_polygon(seg.source, seg.target, boundary)
        for s, t in segs:
            clipped.append(RoadSegment(
                source=s, target=t,
                road_type=seg.road_type, lane_count=seg.lane_count,
                width=seg.width, sidewalk=seg.sidewalk,
            ))
    return clipped


def _clip_line_to_polygon(a, b, polygon):
    result = []
    inside_a = _point_in_polygon(a, polygon)
    inside_b = _point_in_polygon(b, polygon)
    if inside_a and inside_b:
        result.append((a, b))
        return result
    intersections = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        pt = _segment_intersection(a, b, p1, p2)
        if pt:
            intersections.append(pt)
    intersections.sort(key=lambda p: ((p[0]-a[0])**2 + (p[1]-a[1])**2))
    if inside_a and not inside_b and intersections:
        result.append((a, intersections[0]))
    elif not inside_a and inside_b and intersections:
        result.append((intersections[0], b))
    elif not inside_a and not inside_b and len(intersections) >= 2:
        result.append((intersections[0], intersections[1]))
    return result


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _segment_intersection(a, b, c, d):
    denom = ((b[0]-a[0])*(d[1]-c[1]) - (b[1]-a[1])*(d[0]-c[0]))
    if abs(denom) < 1e-10:
        return None
    t = ((c[0]-a[0])*(d[1]-c[1]) - (c[1]-a[1])*(d[0]-c[0])) / denom
    u = ((c[0]-a[0])*(b[1]-a[1]) - (c[1]-a[1])*(b[0]-a[0])) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
    return None


def bbox_lots(bottom_left, top_right, lot_width, lot_depth):
    x0, y0 = bottom_left
    x1, y1 = top_right
    lots = []
    y = y0
    while y < y1 - lot_depth * 0.5:
        x = x0
        while x < x1 - lot_width * 0.5:
            bx, by = x, y
            tx, ty = min(x + lot_width, x1), min(y + lot_depth, y1)
            lots.append(BuildingLot(
                boundary=[(bx, by), (tx, by), (tx, ty), (bx, ty)],
                area=(tx - bx) * (ty - by),
            ))
            x += lot_width
        y += lot_depth
    return lots
