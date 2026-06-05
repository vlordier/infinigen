import pytest

from infinigen.assets.urban.dcel import DCELNode, DCEHalfEdge, DCEFace, DCEL


def test_empty_dcel():
    dcel = DCEL()
    assert len(dcel.nodes) == 0
    assert len(dcel.half_edges) == 0
    assert len(dcel.faces) == 0


def test_dcel_from_cycle():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert len(dcel.nodes) == 4
    assert len(dcel.half_edges) == 8
    assert len(dcel.faces) == 2


def test_half_edge_invariants():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    for he in dcel.half_edges:
        assert he.twin is not None
        assert he.twin.twin is he
        assert he.next is not None
        assert he.prev is not None
        assert he.face is not None
        assert he.origin is not None


def test_face_cycle_closed():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    for face in dcel.faces:
        edges = []
        he = face.half_edge
        start = he
        while True:
            edges.append(he)
            he = he.next
            if he is start:
                break
        assert len(edges) >= 3


def test_split_edge():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    he = dcel.half_edges[0]  # (0,0) -> (0,10)
    new_node = dcel.split_edge(he, (0, 5))
    assert new_node.position == (0, 5)
    assert len(dcel.nodes) == 5
    assert len(dcel.half_edges) == 10  # was 8, +2 for split


def test_connect_nodes_in_same_face():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    n_face = dcel.faces[0]  # interior face
    new_node = dcel.add_node((5, 5), n_face)
    origin = dcel.nodes[0]
    new_face = dcel.connect_nodes(origin, new_node)
    assert new_face is not None
    assert not new_face.is_boundary
    assert len(dcel.faces) == 4  # 2 original + add_node split + connect_nodes split


def test_add_node_splits_face():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    n_face = dcel.faces[0]
    new_node = dcel.add_node((5, 5), n_face)
    assert len(dcel.faces) == 3  # interior split into 2
    assert len(dcel.nodes) == 5


def test_split_edge_preserves_invariants():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    dcel.split_edge(dcel.half_edges[0], (0, 5))
    for he in dcel.half_edges:
        if he.next is not None:
            assert he.twin is not None
            assert he.twin.twin is he
            assert he.next.prev is he


def test_connect_nodes_rejects_self_connection():
    dcel = DCEL.from_cycle([(0, 0), (0, 10), (10, 10), (10, 0)])
    with pytest.raises(ValueError):
        dcel.connect_nodes(dcel.nodes[0], dcel.nodes[0])
