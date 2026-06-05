from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.dcel import DCEL


def test_generate_returns_dcel():
    dcel = GraphGenerator.generate(100, 100, seed=42)
    assert isinstance(dcel, DCEL)
    assert len(dcel.nodes) >= 4
    assert len(dcel.faces) >= 2


def test_generate_deterministic():
    dcel1 = GraphGenerator.generate(100, 100, seed=42)
    dcel2 = GraphGenerator.generate(100, 100, seed=42)
    pos1 = sorted([n.position for n in dcel1.nodes])
    pos2 = sorted([n.position for n in dcel2.nodes])
    assert pos1 == pos2


def test_generate_different_seed():
    dcel1 = GraphGenerator.generate(100, 100, seed=42)
    dcel2 = GraphGenerator.generate(100, 100, seed=99)
    pos1 = sorted([n.position for n in dcel1.nodes])
    pos2 = sorted([n.position for n in dcel2.nodes])
    assert pos1 != pos2


def test_generate_expected_node_count():
    dcel = GraphGenerator.generate(100, 100, seed=42)
    assert len(dcel.nodes) >= 10


def test_all_faces_are_simple_polygons():
    dcel = GraphGenerator.generate(100, 100, seed=42)
    for face in dcel.faces:
        if face.is_boundary:
            continue
        vertices = []
        he = face.half_edge
        start = he
        while True:
            vertices.append(he.origin.position)
            he = he.next
            if he is start:
                break
        assert len(vertices) >= 3
