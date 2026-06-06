import math
from infinigen.assets.urban.dcel import DCEL, DCELNode, DCEHalfEdge, DCEFace
from infinigen.assets.urban.graph_parser import RoadSegment


class RoadToDCEL:
    @staticmethod
    def build(segments: list[RoadSegment]) -> DCEL:
        dcel = DCEL()
        pos_to_node = {}
        node_to_outgoing: dict[int, list[tuple[float, float]]] = {}
        for seg in segments:
            for pos in (seg.source, seg.target):
                if pos not in pos_to_node:
                    node = DCELNode(position=pos)
                    pos_to_node[pos] = node
                    dcel.nodes.append(node)
        for seg in segments:
            src_node = pos_to_node[seg.source]
            tgt_node = pos_to_node[seg.target]
            dx = tgt_node.position[0] - src_node.position[0]
            dy = tgt_node.position[1] - src_node.position[1]
            angle = (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
            node_to_outgoing.setdefault(id(src_node), []).append(
                (angle, tgt_node.position)
            )
            dx = src_node.position[0] - tgt_node.position[0]
            dy = src_node.position[1] - tgt_node.position[1]
            angle = (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
            node_to_outgoing.setdefault(id(tgt_node), []).append(
                (angle, src_node.position)
            )
        he_map = {}
        for seg in segments:
            src_node = pos_to_node[seg.source]
            tgt_node = pos_to_node[seg.target]
            he_a = DCEHalfEdge(origin=src_node)
            he_b = DCEHalfEdge(origin=tgt_node)
            he_a.twin = he_b
            he_b.twin = he_a
            dcel.half_edges.append(he_a)
            dcel.half_edges.append(he_b)
            he_map[(src_node.position, tgt_node.position)] = he_a
            he_map[(tgt_node.position, src_node.position)] = he_b
            if src_node.half_edge is None:
                src_node.half_edge = he_a
            if tgt_node.half_edge is None:
                tgt_node.half_edge = he_b
        for node in dcel.nodes:
            outgoing = node_to_outgoing.get(id(node), [])
            outgoing.sort(key=lambda x: x[0])
            n = len(outgoing)
            for i in range(n):
                _, tgt_pos = outgoing[i]
                _, next_tgt_pos = outgoing[(i + 1) % n]
                he = he_map[(node.position, tgt_pos)]
                next_he = he_map[(node.position, next_tgt_pos)]
                he.twin.next = next_he
        for he in dcel.half_edges:
            if he.next is not None and he.next.prev is None:
                he.next.prev = he
        for he in dcel.half_edges:
            if he.twin is not None and he.twin.next is not None:
                he.prev = he.twin.next.twin
        dcel.faces = RoadToDCEL._extract_faces(dcel)
        _set_face_half_edges(dcel)
        return dcel

    @staticmethod
    def _extract_faces(dcel: DCEL) -> list[DCEFace]:
        visited = set()
        faces = []
        boundary_face = DCEFace(is_boundary=True)
        for start_he in dcel.half_edges:
            if id(start_he) in visited:
                continue
            he = start_he
            cycle = []
            while id(he) not in visited:
                visited.add(id(he))
                cycle.append(he)
                if he.next is None:
                    break
                he = he.next
                if he is start_he:
                    break
            if len(cycle) < 3:
                continue
            is_boundary = RoadToDCEL._is_boundary_cycle(cycle)
            face = boundary_face if is_boundary else DCEFace()
            for h in cycle:
                h.face = face
            if not is_boundary:
                faces.append(face)
        faces.append(boundary_face)
        return faces

    @staticmethod
    def _is_boundary_cycle(cycle) -> bool:
        return any(
            he.twin is None or he.twin.face is None
            for he in cycle
        )


def _set_face_half_edges(dcel: DCEL):
    for face in dcel.faces:
        for he in dcel.half_edges:
            if he.face is face:
                face.half_edge = he
                break
