from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.graph_parser import GraphParser
from infinigen.assets.urban.road_to_dcel import RoadToDCEL
from infinigen.assets.urban.dcel import DCEL
from infinigen.assets.urban.graph_parser import RoadSegment


def test_roundtrip_preserves_segments():
    original = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(original)
    segments = parser.road_segments
    rebuilt = RoadToDCEL.build(segments)
    assert isinstance(rebuilt, DCEL)
    assert len(rebuilt.nodes) == len(original.nodes)
    assert len(rebuilt.half_edges) == len(original.half_edges)


def test_dcel_from_simple_cycle():
    segments = [
        RoadSegment(source=(0, 0), target=(100, 0), road_type="local"),
        RoadSegment(source=(100, 0), target=(100, 100), road_type="local"),
        RoadSegment(source=(100, 100), target=(0, 100), road_type="local"),
        RoadSegment(source=(0, 100), target=(0, 0), road_type="local"),
    ]
    dcel = RoadToDCEL.build(segments)
    assert len(dcel.nodes) == 4
    assert len(dcel.half_edges) == 8
    assert len(dcel.faces) >= 1


def test_dcel_from_t_junction():
    segments = [
        RoadSegment(source=(0, 0), target=(100, 0), road_type="local"),
        RoadSegment(source=(100, 0), target=(100, 100), road_type="local"),
        RoadSegment(source=(100, 100), target=(0, 100), road_type="local"),
        RoadSegment(source=(0, 100), target=(0, 0), road_type="local"),
        RoadSegment(source=(50, 0), target=(50, 50), road_type="local"),
    ]
    dcel = RoadToDCEL.build(segments)
    assert len(dcel.nodes) == 6
    assert len(dcel.faces) >= 2
