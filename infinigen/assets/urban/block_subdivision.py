from dataclasses import dataclass
import random


@dataclass
class BuildingLot:
    boundary: list[tuple[float, float]]
    area: float = 0.0
    building_type: str = "residential"


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
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max_x - min_x
    bh = max_y - min_y
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
    jitter = rng.uniform(-1.5, 1.5) if n_lots > 1 else 0.0
    lots = []
    for i in range(n_lots):
        depth_jitter = rng.uniform(-actual_depth * 0.1, actual_depth * 0.1) if n_lots > 1 else 0.0
        offset_jitter = rng.uniform(-side_setback * 0.2, side_setback * 0.2)
        ld = actual_depth + depth_jitter
        lo = offset_jitter
        ly = inner_y + i * actual_depth
        if flipped:
            lot_boundary = [
                (min_x + ly - min_y - lo, min_y + inner_x - min_x),
                (min_x + ly - min_y - lo, min_y + inner_x + inner_w - min_x),
                (min_x + ly + ld - min_y - lo, min_y + inner_x + inner_w - min_x),
                (min_x + ly + ld - min_y - lo, min_y + inner_x - min_x),
            ]
        else:
            lot_boundary = [
                (inner_x + lo, ly),
                (inner_x + inner_w + lo, ly),
                (inner_x + inner_w + lo, ly + ld),
                (inner_x + lo, ly + ld),
            ]
        lot_area = inner_w * ld
        if lot_area < 20:
            continue
        btype = _infer_building_type(lot_area, rng)
        lots.append(BuildingLot(
            boundary=lot_boundary, area=lot_area, building_type=btype,
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
