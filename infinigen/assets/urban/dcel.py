from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DCELNode:
    position: tuple[float, float]
    half_edge: Optional[DCEHalfEdge] = None


@dataclass
class DCEHalfEdge:
    origin: DCELNode
    twin: DCEHalfEdge
    next: DCEHalfEdge
    prev: DCEHalfEdge
    face: DCEFace
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
        raise NotImplementedError("add_node — full impl with GraphGenerator")

    def connect_nodes(self, node0: DCELNode,
                      node1: DCELNode) -> DCEFace:
        raise NotImplementedError(
            "connect_nodes — full impl with GraphGenerator")

    def split_edge(self, he: DCEHalfEdge,
                   position: tuple[float, float]) -> DCELNode:
        raise NotImplementedError(
            "split_edge — full impl with GraphGenerator")
