import gin
import random
import math
from dataclasses import dataclass, field
from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.graph_parser import GraphParser

@dataclass
class RoadNode:
    position: tuple = (0, 0, 0)
    node_type: str = "intersection"

@dataclass
class RoadEdge:
    node_a: str = ""
    node_b: str = ""
    road_type: str = "local"
    lane_count: int = 2
    width: float = 12.0
    sidewalk: bool = True

@gin.configurable
class RoadGraph:
    def __init__(self, bounds=(1000, 1000), seed=42):
        self.bounds = bounds
        self.seed = seed
        self.nodes = {}
        self.edges = []
        random.seed(seed)
    
    def generate_grid(self, spacing=100, jitter=20):
        w, d = self.bounds
        n_x = int(w / spacing)
        n_y = int(d / spacing)
        for i in range(n_x + 1):
            for j in range(n_y + 1):
                x = i * spacing - w/2 + random.uniform(-jitter, jitter)
                y = j * spacing - d/2 + random.uniform(-jitter, jitter)
                node_id = f"n_{i}_{j}"
                self.nodes[node_id] = RoadNode(position=(x, y, 0), node_type="intersection")
        for i in range(n_x + 1):
            for j in range(n_y):
                a = f"n_{i}_{j}"
                b = f"n_{i}_{j+1}"
                self.edges.append(RoadEdge(node_a=a, node_b=b, road_type="local"))
        for i in range(n_x):
            for j in range(n_y + 1):
                a = f"n_{i}_{j}"
                b = f"n_{i+1}_{j}"
                self.edges.append(RoadEdge(node_a=a, node_b=b, road_type="local"))
    
    def add_arterial(self, start_node, end_node, road_type="arterial"):
        self.edges.append(RoadEdge(node_a=start_node, node_b=end_node, road_type=road_type, lane_count=4, width=20.0))
    
    def get_all_positions(self):
        return {nid: nd.position for nid, nd in self.nodes.items()}

    def generate_dcel(self, size_x=None, size_y=None, seed=None):
        size_x = size_x or (self.bounds[0] if not isinstance(self.bounds, (int, float)) else self.bounds)
        size_y = size_y or (self.bounds[1] if not isinstance(self.bounds, (int, float)) else self.bounds)
        seed = seed or self.seed
        dcel = GraphGenerator.generate(float(size_x), float(size_y), seed)
        parser = GraphParser(dcel)
        self.nodes = {}
        self.edges = []
        for i, seg in enumerate(parser.road_segments):
            a_id = f"dcel_{i}_a"
            b_id = f"dcel_{i}_b"
            self.nodes[a_id] = RoadNode(position=(*seg.source, 0), node_type="intersection")
            self.nodes[b_id] = RoadNode(position=(*seg.target, 0), node_type="intersection")
            self.edges.append(RoadEdge(
                node_a=a_id, node_b=b_id,
                road_type=seg.road_type,
                lane_count=seg.lane_count,
                width=seg.width,
                sidewalk=seg.sidewalk,
            ))
        return parser
