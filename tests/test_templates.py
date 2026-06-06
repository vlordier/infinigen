from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.template_utils import (
    make_grid_segments, clip_segments_to_boundary, bbox_lots,
)


def test_make_grid_segments():
    segs, xp, yp = make_grid_segments((0, 0), (100, 100), spacing=50)
    assert len(segs) >= 4
    assert len(xp) >= 2
    assert len(yp) >= 2


def test_clip_segments():
    segs = [RoadSegment(source=(0, 0), target=(100, 0))]
    boundary = [(10, -10), (90, -10), (90, 10), (10, 10)]
    clipped = clip_segments_to_boundary(segs, boundary)
    assert len(clipped) >= 1


def test_bbox_lots():
    lots = bbox_lots((0, 0), (100, 100), lot_width=50, lot_depth=50)
    assert len(lots) == 4


import random

from infinigen.assets.urban.templates import (
    RectangularGridTemplate, OrganicGridTemplate, DistrictTemplateConfig,
)


def test_rectangular_grid_fill():
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    config = DistrictTemplateConfig(lot_depth=25, lot_width=25)
    rng = random.Random(42)
    result = RectangularGridTemplate.fill(boundary, config, rng)
    assert len(result.road_segments) >= 2
    assert len(result.building_lots) >= 4


def test_organic_grid_fill():
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    config = DistrictTemplateConfig(lot_depth=25, lot_width=25, irregularity=0.2)
    rng = random.Random(42)
    result = OrganicGridTemplate.fill(boundary, config, rng)
    assert len(result.road_segments) >= 2


def test_medieval_organic_fill():
    from infinigen.assets.urban.templates import MedievalOrganicTemplate
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    config = DistrictTemplateConfig(lot_depth=10, lot_width=8, density=0.8)
    result = MedievalOrganicTemplate.fill(boundary, config, random.Random(42))
    assert len(result.building_lots) >= 3


def test_soviet_block_fill():
    from infinigen.assets.urban.templates import SovietBlockTemplate
    boundary = [(0, 0), (200, 0), (200, 200), (0, 200)]
    config = DistrictTemplateConfig(lot_depth=100, lot_width=80, density=0.3)
    result = SovietBlockTemplate.fill(boundary, config, random.Random(42))
    assert len(result.building_lots) >= 1
