import networkx as nx
from infinigen.assets.urban.osmnx_skeleton import OsmnxSkeleton
from infinigen.assets.urban.skeleton import CitySkeleton, BlockFace


def _make_square_graph():
    G = nx.MultiDiGraph()
    nodes = {
        0: {"x": 0.0, "y": 0.0},
        1: {"x": 100.0, "y": 0.0},
        2: {"x": 100.0, "y": 100.0},
        3: {"x": 0.0, "y": 100.0},
    }
    for nid, pos in nodes.items():
        G.add_node(nid, **pos)
    edges = [
        (0, 1, {"highway": "primary", "lanes": 4}),
        (1, 2, {"highway": "primary", "lanes": 4}),
        (2, 3, {"highway": "primary", "lanes": 4}),
        (3, 0, {"highway": "primary", "lanes": 4}),
    ]
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    return G


def _make_t_junction_graph():
    G = nx.MultiDiGraph()
    nodes = {
        0: {"x": 0.0, "y": 0.0},
        1: {"x": 100.0, "y": 0.0},
        2: {"x": 100.0, "y": 100.0},
        3: {"x": 0.0, "y": 100.0},
        4: {"x": 50.0, "y": 0.0},
        5: {"x": 50.0, "y": 50.0},
    }
    for nid, pos in nodes.items():
        G.add_node(nid, **pos)
    edges = [
        (0, 1, {"highway": "primary", "lanes": 4}),
        (1, 2, {"highway": "primary", "lanes": 4}),
        (2, 3, {"highway": "primary", "lanes": 4}),
        (3, 0, {"highway": "primary", "lanes": 4}),
        (4, 5, {"highway": "secondary", "lanes": 2}),
    ]
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    return G


def test_from_graph_returns_city_skeleton():
    G = _make_square_graph()
    result = OsmnxSkeleton.from_graph(G)
    assert isinstance(result, CitySkeleton)
    assert len(result.road_segments) >= 4
    assert len(result.blocks) >= 1
    for block in result.blocks:
        assert isinstance(block, BlockFace)
        assert len(block.boundary) >= 3
        assert block.zone_id == "inner"


def test_from_graph_road_segments():
    G = _make_square_graph()
    result = OsmnxSkeleton.from_graph(G)
    segs = result.road_segments
    assert len(segs) == 4
    types = {s.road_type for s in segs}
    assert types == {"arterial"}
    assert all(s.lane_count == 4 for s in segs)


def test_from_graph_t_junction():
    G = _make_t_junction_graph()
    result = OsmnxSkeleton.from_graph(G)
    assert len(result.road_segments) == 5
    assert len(result.blocks) >= 1
    assert result.blocks[0].zone_id == "inner"


def test_highway_mapping():
    G = nx.MultiDiGraph()
    G.add_node(0, x=0, y=0)
    G.add_node(1, x=100, y=0)
    G.add_edge(0, 1, highway="motorway")
    result = OsmnxSkeleton.from_graph(G)
    assert result.road_segments[0].road_type == "arterial"

    G2 = nx.MultiDiGraph()
    G2.add_node(0, x=0, y=0)
    G2.add_node(1, x=100, y=0)
    G2.add_edge(0, 1, highway="residential")
    result2 = OsmnxSkeleton.from_graph(G2)
    assert result2.road_segments[0].road_type == "local"


def test_lane_parsing():
    G = nx.MultiDiGraph()
    G.add_node(0, x=0, y=0)
    G.add_node(1, x=100, y=0)
    G.add_edge(0, 1, highway="primary", lanes="3")
    result = OsmnxSkeleton.from_graph(G)
    assert result.road_segments[0].lane_count == 3


def test_from_graph_with_edge_geometry():
    from shapely.geometry import LineString
    G = nx.MultiDiGraph()
    G.add_node(0, x=0, y=0)
    G.add_node(1, x=100, y=0)
    geom = LineString([(0, 0), (50, 10), (100, 0)])
    G.add_edge(0, 1, highway="primary", geometry=geom)
    result = OsmnxSkeleton.from_graph(G)
    assert len(result.road_segments) == 2
    seg_a, seg_b = result.road_segments
    assert seg_a.source == (0, 0)
    assert seg_a.target == (50, 10)
    assert seg_b.source == (50, 10)
    assert seg_b.target == (100, 0)
