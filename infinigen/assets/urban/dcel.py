from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DCELNode:
    position: tuple[float, float]
    half_edge: Optional[DCEHalfEdge] = None


@dataclass
class DCEHalfEdge:
    origin: DCELNode
    twin: Optional[DCEHalfEdge] = None
    next: Optional[DCEHalfEdge] = None
    prev: Optional[DCEHalfEdge] = None
    face: Optional[DCEFace] = None
    road_type: str = "local"


@dataclass
class DCEFace:
    half_edge: Optional[DCEHalfEdge] = None
    is_boundary: bool = False


@dataclass
class DCEL:
    nodes: list[DCELNode] = field(default_factory=list)
    half_edges: list[DCEHalfEdge] = field(default_factory=list)
    faces: list[DCEFace] = field(default_factory=list)

    @staticmethod
    def from_cycle(positions: list[tuple[float, float]]) -> DCEL:
        dcel = DCEL()
        nodes = [DCELNode(position=p) for p in positions]
        dcel.nodes.extend(nodes)
        n = len(nodes)
        half_edges = []
        for i in range(n):
            he = DCEHalfEdge(origin=nodes[i], twin=None,
                             next=None, prev=None, face=None)
            twin = DCEHalfEdge(origin=nodes[(i + 1) % n], twin=he,
                               next=None, prev=None, face=None)
            he.twin = twin
            half_edges.append(he)
            half_edges.append(twin)
        dcel.half_edges.extend(half_edges)
        for i in range(n):
            he = half_edges[i * 2]
            he.next = half_edges[((i + 1) % n) * 2]
            he.prev = half_edges[((i - 1 + n) % n) * 2]
            twin = he.twin
            twin.next = half_edges[((i - 1 + n) % n) * 2 + 1]
            twin.prev = half_edges[((i + 1) % n) * 2 + 1]
        interior = DCEFace(half_edge=half_edges[0])
        dcel.faces.append(interior)
        for i in range(n):
            half_edges[i * 2].face = interior
        exterior = DCEFace(half_edge=half_edges[1], is_boundary=True)
        dcel.faces.append(exterior)
        for i in range(n):
            half_edges[i * 2 + 1].face = exterior
        for i, node in enumerate(nodes):
            node.half_edge = half_edges[i * 2]
        return dcel

    def add_node(self, position: tuple[float, float],
                 target_face: DCEFace) -> DCELNode:
        edges = []
        he = target_face.half_edge
        if he is None:
            raise ValueError("Target face has no half-edge")
        start = he
        while True:
            edges.append(he)
            he = he.next
            if he is start or he is None:
                break
        target_edge = random.choice(edges)
        ax, ay = target_edge.origin.position
        bx, by = target_edge.twin.origin.position if target_edge.twin else (ax, ay)
        # Project position onto the edge's line to find the split point
        dx = bx - ax
        dy = by - ay
        edge_len_sq = dx * dx + dy * dy
        if edge_len_sq > 0:
            t = ((position[0] - ax) * dx + (position[1] - ay) * dy) / edge_len_sq
            t = max(0.01, min(0.99, t))
            split_pos = (ax + dx * t, ay + dy * t)
        else:
            split_pos = (ax, ay)
        new_node = self.split_edge(target_edge, split_pos)
        candidates = []
        he = target_face.half_edge
        if he is not None:
            start = he
            while True:
                n = he.origin
                is_new = n is new_node
                adjacent = False
                adj = new_node.half_edge
                if adj is not None:
                    adj_start = adj
                    while True:
                        if adj.twin is not None and adj.twin.origin is n:
                            adjacent = True
                            break
                        adj = adj.twin.next if adj.twin is not None else None
                        if adj is None or adj is adj_start:
                            break
                if not is_new and not adjacent:
                    candidates.append(n)
                he = he.next
                if he is start or he is None:
                    break
        if candidates:
            chosen = random.choice(candidates)
            self.connect_nodes(new_node, chosen)
        return new_node

    def _find_shared_face(self, node0: DCELNode, node1: DCELNode) -> Optional[DCEFace]:
        visited = set()
        start_he = node0.half_edge
        if start_he is None:
            return None
        he = start_he
        while True:
            face = he.face
            if face is not None and not face.is_boundary and id(face) not in visited:
                f_he = face.half_edge
                if f_he is not None:
                    f_start = f_he
                    while True:
                        if f_he.origin is node1:
                            return face
                        f_he = f_he.next
                        if f_he is f_start or f_he is None:
                            break
                visited.add(id(face))
            he = he.twin.next if he.twin is not None else None
            if he is None or he is start_he:
                break
        return None

    def _find_incident_half_edge(self, node: DCELNode, face: DCEFace) -> Optional[DCEHalfEdge]:
        start_he = node.half_edge
        if start_he is None:
            return None
        he = start_he
        while True:
            if he.face is face:
                return he
            he = he.twin.next if he.twin is not None else None
            if he is None or he is start_he:
                break
        return None

    def connect_nodes(self, node0: DCELNode,
                      node1: DCELNode) -> DCEFace:
        if node0 is node1:
            raise ValueError("Cannot connect a node to itself")
        face = self._find_shared_face(node0, node1)
        if face is None:
            raise ValueError("Nodes not on same interior face or cannot find shared face")
        he_a_start = self._find_incident_half_edge(node0, face)
        if he_a_start is None:
            raise ValueError("Cannot find incident half-edge")
        path_a = []
        he = he_a_start
        while he.origin is not node1:
            path_a.append(he)
            he = he.next
            if he is he_a_start or he is None:
                raise ValueError("Cannot find path from node0 to node1")
        he_at_node1 = he
        path_b = []
        he = he_at_node1
        while he is not None and he.origin is not node0:
            path_b.append(he)
            he = he.next
            if he is he_at_node1:
                break
        if not path_b:
            raise ValueError("Nodes are adjacent, cannot connect")
        he_new = DCEHalfEdge(origin=node0)
        he_new_twin = DCEHalfEdge(origin=node1, twin=he_new)
        he_new.twin = he_new_twin
        self.half_edges.append(he_new)
        self.half_edges.append(he_new_twin)
        new_face = DCEFace(half_edge=path_b[0])
        self.faces.append(new_face)
        for phe in path_b:
            phe.face = new_face
        he_new_twin.face = face
        he_new.face = new_face
        path_a[-1].next = he_new_twin
        he_new_twin.prev = path_a[-1]
        he_new_twin.next = he_a_start
        he_a_start.prev = he_new_twin
        he_new.next = path_b[0]
        path_b[0].prev = he_new
        path_b[-1].next = he_new
        he_new.prev = path_b[-1]
        if face.half_edge is not None and face.half_edge.face is not face:
            face.half_edge = path_a[0] if path_a else he_new_twin
        return new_face

    def split_edge(self, he: DCEHalfEdge,
                   position: tuple[float, float]) -> DCELNode:
        if he.twin is None:
            raise ValueError("Cannot split a half-edge with no twin")
        twin = he.twin
        he_next = he.next
        he_prev = he.prev
        twin_next = twin.next
        twin_prev = twin.prev
        face = he.face
        twin_face = twin.face
        new_node = DCELNode(position=position)
        self.nodes.append(new_node)
        he_a = DCEHalfEdge(origin=he.origin, face=face)
        he_a_twin = DCEHalfEdge(origin=new_node, twin=he_a, face=twin_face)
        he_a.twin = he_a_twin
        he_b = DCEHalfEdge(origin=new_node, face=face)
        he_b_twin = DCEHalfEdge(origin=twin.origin, twin=he_b, face=twin_face)
        he_b.twin = he_b_twin
        self.half_edges.extend([he_a, he_a_twin, he_b, he_b_twin])
        self.half_edges.remove(he)
        self.half_edges.remove(twin)
        he_a.next = he_b
        he_a.prev = he_prev
        he_b.next = he_next
        he_b.prev = he_a
        if he_prev is not None:
            he_prev.next = he_a
        if he_next is not None:
            he_next.prev = he_b
        he_b_twin.next = he_a_twin
        he_b_twin.prev = twin_prev
        he_a_twin.next = twin_next
        he_a_twin.prev = he_b_twin
        if twin_prev is not None:
            twin_prev.next = he_b_twin
        if twin_next is not None:
            twin_next.prev = he_a_twin
        if face is not None and face.half_edge is he:
            face.half_edge = he_a
        if twin_face is not None and twin_face.half_edge is twin:
            twin_face.half_edge = he_a_twin
        if he.origin.half_edge is he:
            he.origin.half_edge = he_a
        if twin.origin.half_edge is twin:
            twin.origin.half_edge = he_b_twin
        new_node.half_edge = he_b
        return new_node
