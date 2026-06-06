from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.template_utils import (
    make_grid_segments, clip_segments_to_boundary, bbox_lots,
)


def test_make_grid_segments():
    segs = make_grid_segments((0, 0), (100, 100), spacing=50)
    assert len(segs) >= 4


def test_clip_segments():
    segs = [RoadSegment(source=(0, 0), target=(100, 0))]
    boundary = [(10, -10), (90, -10), (90, 10), (10, 10)]
    clipped = clip_segments_to_boundary(segs, boundary)
    assert len(clipped) >= 1


def test_bbox_lots():
    lots = bbox_lots((0, 0), (100, 100), lot_width=50, lot_depth=50)
    assert len(lots) == 4
