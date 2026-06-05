import pytest
from infinigen.assets.urban.block_subdivision import subdivide_lots, BuildingLot


def make_area(boundary, area=None):
    class FakeArea:
        pass
    a = FakeArea()
    a.boundary = boundary
    a.area = area or _polygon_area(boundary)
    return a


def _polygon_area(verts):
    area = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def test_subdivide_square_block():
    block = [(0, 0), (0, 50), (50, 50), (50, 0)]
    lots = subdivide_lots([make_area(block, 2500)], seed=42)
    assert len(lots) > 0
    for lot in lots:
        assert len(lot.boundary) >= 4


def test_lots_have_positive_area():
    block = [(0, 0), (0, 50), (50, 50), (50, 0)]
    lots = subdivide_lots([make_area(block, 2500)], seed=42)
    for lot in lots:
        assert lot.area > 0


def test_setback_respected():
    block = [(0, 0), (0, 50), (50, 50), (50, 0)]
    lots = subdivide_lots([make_area(block, 2500)], seed=42,
                          front_setback=5.0, side_setback=3.0)
    for lot in lots:
        xs = [p[0] for p in lot.boundary]
        ys = [p[1] for p in lot.boundary]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        assert w <= 50 - 2 * 3.0 + 0.01
        assert h <= 50 - 5.0 - 3.0 + 0.01


def test_building_type_assigned():
    block = [(0, 0), (0, 50), (50, 50), (50, 0)]
    lots = subdivide_lots([make_area(block, 2500)], seed=42)
    for lot in lots:
        assert lot.building_type in ("residential", "commercial", "industrial")


def test_multiple_blocks():
    block1 = [(0, 0), (0, 50), (50, 50), (50, 0)]
    block2 = [(60, 0), (60, 30), (100, 30), (100, 0)]
    lots = subdivide_lots([make_area(block1, 2500), make_area(block2, 1200)], seed=42)
    assert len(lots) >= 2
