from infinigen.assets.urban.graph_parser import GraphParser, RoadSegment, CityArea
from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.dcel import DCEL


def test_parse_road_segments():
    dcel = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(dcel)
    segments = parser.road_segments
    assert len(segments) > 0
    for seg in segments:
        assert len(seg.source) == 2
        assert len(seg.target) == 2
        assert seg.road_type in ("highway", "arterial", "local", "alley")
        assert seg.lane_count >= 1
        assert seg.width > 0


def test_parse_city_areas():
    dcel = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(dcel)
    areas = parser.city_areas
    assert len(areas) > 0
    for area in areas:
        assert len(area.boundary) >= 3
        assert area.area > 0


def test_road_segment_lengths_positive():
    dcel = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(dcel)
    for seg in parser.road_segments:
        assert seg.length > 0


def test_each_edge_visited_once():
    """Every unique edge in the DCEL produces exactly one RoadSegment."""
    dcel = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(dcel)
    expected = len(dcel.half_edges) // 2
    assert len(parser.road_segments) == expected


def test_city_areas_cover_interior():
    """City areas should correspond to non-boundary faces."""
    dcel = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(dcel)
    interior_count = sum(1 for f in dcel.faces if not f.is_boundary)
    assert len(parser.city_areas) == interior_count
