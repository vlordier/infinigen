from dataclasses import dataclass
import random
import math


@dataclass
class BuildingLot:
    boundary: list[tuple[float, float]]
    area: float = 0.0
    building_type: str = "residential"


def _polygon_area(pts):
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _point_left_of_edge(point, e0, e1):
    return (e1[0] - e0[0]) * (point[1] - e0[1]) - (e1[1] - e0[1]) * (point[0] - e0[0]) >= 0


def _line_intersection(a0, a1, b0, b1):
    da = (a1[0] - a0[0], a1[1] - a0[1])
    db = (b1[0] - b0[0], b1[1] - b0[1])
    denom = da[0] * db[1] - da[1] * db[0]
    if abs(denom) < 1e-12:
        return None
    t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / denom
    return (a0[0] + t * da[0], a0[1] + t * da[1])


def _signed_area(pts):
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return area


def _ensure_ccw(poly):
    if _signed_area(poly) < 0:
        return list(reversed(poly))
    return list(poly)


def _clip_polygon(subject, clip):
    if len(subject) < 3 or len(clip) < 3:
        return []
    clip = _ensure_ccw(clip)
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if len(output) < 3:
            return []
        e0 = clip[i]
        e1 = clip[(i + 1) % n]
        input_list = output
        output = []
        m = len(input_list)
        for j in range(m):
            curr = input_list[j]
            prev = input_list[(j - 1) % m]
            curr_inside = _point_left_of_edge(curr, e0, e1)
            prev_inside = _point_left_of_edge(prev, e0, e1)
            if curr_inside:
                if not prev_inside:
                    t = _line_intersection(e0, e1, prev, curr)
                    if t:
                        output.append(t)
                output.append(curr)
            elif prev_inside:
                t = _line_intersection(e0, e1, prev, curr)
                if t:
                    output.append(t)
    return output


def subdivide_lots(city_areas: list, seed: int = 0,
                   front_setback: float = 5.0, side_setback: float = 3.0,
                   back_setback: float = 3.0) -> list[BuildingLot]:
    rng = random.Random(seed)
    lots = []
    for area in city_areas:
        block_lots = _subdivide_block(
            area.boundary, area.area, rng,
            front_setback, side_setback, back_setback,
        )
        lots.extend(block_lots)
    return lots


def _subdivide_block(boundary, area, rng, front_setback, side_setback, back_setback):
    boundary = list(boundary)
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max_x - min_x
    bh = max_y - min_y
    if bw <= 0 or bh <= 0:
        return []

    # Determine subdivision axis (longer side)
    flipped = bw < bh
    if flipped:
        bw, bh = bh, bw
        min_x, min_y = min_y, min_x
        max_x, max_y = max_y, max_x

    inner_x = min_x + side_setback
    inner_y = min_y + front_setback
    inner_w = bw - 2 * side_setback
    inner_h = bh - front_setback - back_setback
    if inner_w <= 0 or inner_h <= 0:
        return []

    lot_depth = max(15.0, inner_h)
    n_lots = max(1, int(inner_h / lot_depth))
    actual_depth = inner_h / n_lots

    lots = []
    for i in range(n_lots):
        depth_jitter = rng.uniform(-actual_depth * 0.1, actual_depth * 0.1) if n_lots > 1 else 0.0
        offset_jitter = rng.uniform(-side_setback * 0.2, side_setback * 0.2)
        ld = actual_depth + depth_jitter
        lo = offset_jitter
        ly = inner_y + i * actual_depth

        if flipped:
            rect = [
                (min_x + ly - min_y - lo, min_y + inner_x - min_x),
                (min_x + ly - min_y - lo, min_y + inner_x + inner_w - min_x),
                (min_x + ly + ld - min_y - lo, min_y + inner_x + inner_w - min_x),
                (min_x + ly + ld - min_y - lo, min_y + inner_x - min_x),
            ]
        else:
            rect = [
                (inner_x + lo, ly),
                (inner_x + inner_w + lo, ly),
                (inner_x + inner_w + lo, ly + ld),
                (inner_x + lo, ly + ld),
            ]

        clipped = _clip_polygon(rect, boundary)
        if len(clipped) < 3:
            continue
        lot_area = _polygon_area(clipped)
        if lot_area < 20:
            continue
        btype = _infer_building_type(lot_area, rng)
        lots.append(BuildingLot(
            boundary=clipped, area=lot_area, building_type=btype,
        ))
    return lots


def _infer_building_type(area: float, rng: random.Random = None) -> str:
    if rng is None:
        rng = random.Random()
    if area > 2000:
        if rng.random() < 0.15:
            return "commercial"
        return "industrial"
    elif area > 500:
        if rng.random() < 0.2:
            return "residential"
        elif rng.random() < 0.1:
            return "industrial"
        return "commercial"
    else:
        if rng.random() < 0.1:
            return "commercial"
        return "residential"
