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


def subdivide_block_fill(boundary, rng=None, max_building_width=40, gap=3, setback=3):
    """CARLA-style: fill a block face with rectangular buildings along its longest axis."""
    if rng is None:
        rng = random.Random()
    pts = list(boundary)
    if len(pts) < 3:
        return []

    # Find longest span direction (principal axis of block)
    n_pts = len(pts)
    max_d2 = 0
    best_i, best_j = 0, 0
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            d2 = (pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2
            if d2 > max_d2:
                max_d2 = d2
                best_i, best_j = i, j
    p0 = pts[best_i]; p1 = pts[best_j]
    ax_len = max_d2 ** 0.5
    if ax_len < 5:
        return []
    ax = (p1[0]-p0[0]) / ax_len; ay = (p1[1]-p0[1]) / ax_len
    px, py = -ay, ax  # perpendicular

    # Project all vertices onto both axes
    proj_along = [(p[0]-p0[0])*ax + (p[1]-p0[1])*ay for p in pts]
    proj_perp = [(p[0]-p0[0])*px + (p[1]-p0[1])*py for p in pts]
    min_a, max_a = min(proj_along), max(proj_along)
    min_p, max_p = min(proj_perp), max(proj_perp)

    a_len = max_a - min_a
    p_len = max_p - min_p

    # Number of buildings along the long axis
    # CARLA fills the space - divide the long dimension
    if a_len > max_building_width:
        n_bldg = max(2, int(a_len / max_building_width))
    else:
        n_bldg = 1

    # Width per building (along long axis), evenly divided with gaps
    total_gap = gap * (n_bldg - 1)
    bldg_width = (a_len - total_gap) / n_bldg
    if bldg_width < 5:
        bldg_width = a_len / n_bldg

    # Depth (perpendicular axis) with setback on both sides
    bldg_depth = p_len - setback * 2
    if bldg_depth < 3:
        bldg_depth = p_len

    lots = []
    for i in range(n_bldg):
        start_a = min_a + setback + i * (bldg_width + gap)
        end_a = start_a + bldg_width
        if end_a > max_a - setback and n_bldg > 1:
            end_a = max_a - setback

        # Compute 4 corners of this strip in the block
        corners = []
        for ca, cp in [(start_a, min_p + setback), (end_a, min_p + setback),
                       (end_a, min_p + setback + bldg_depth), (start_a, min_p + setback + bldg_depth)]:
            x = p0[0] + ca*ax + cp*px
            y = p0[1] + ca*ay + cp*py
            corners.append((x, y))

        w = ((corners[1][0]-corners[0][0])**2 + (corners[1][1]-corners[0][1])**2)**0.5
        d = ((corners[2][0]-corners[1][0])**2 + (corners[2][1]-corners[1][1])**2)**0.5
        area = w * d

        # Jitter
        if rng and n_bldg > 1:
            j = rng.uniform(-0.5, 0.5)
            corners = [(x+j*ax, y+j*ay) for x, y in corners]
            area = abs(w * d + j * d * rng.uniform(-0.3, 0.3))

        if area < 30:
            continue

        btype = _infer_building_type(area, rng) if rng else "residential"
        lots.append(BuildingLot(
            boundary=corners, area=area, building_type=btype,
        ))

    return lots
