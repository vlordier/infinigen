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
