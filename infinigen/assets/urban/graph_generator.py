import random
from infinigen.assets.urban.dcel import DCEL


class GraphGenerator:
    @staticmethod
    def generate(size_x: float, size_y: float, seed: int = 0,
                 density: float = 1.0) -> DCEL:
        state = random.getstate()
        random.seed(seed)
        rng = random.Random(seed)
        dcel = DCEL.from_cycle([(0, 0), (0, size_y), (size_x, size_y), (size_x, 0)])

        iterations = max(5, int(size_x * size_y * density * 0.001))

        for _ in range(iterations):
            interior_faces = [f for f in dcel.faces
                              if not f.is_boundary]
            if not interior_faces:
                break

            areas = []
            for face in interior_faces:
                verts = []
                he = face.half_edge
                start = he
                while True:
                    verts.append(he.origin.position)
                    he = he.next
                    if he is start:
                        break
                area = GraphGenerator._polygon_area(verts)
                areas.append(max(area, 0.01))

            total = sum(areas)
            weights = [a / total for a in areas]
            face = rng.choices(interior_faces, weights=weights, k=1)[0]

            verts = []
            he = face.half_edge
            start = he
            while True:
                verts.append(he.origin.position)
                he = he.next
                if he is start:
                    break

            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)

            edge_lengths = []
            for i in range(len(verts)):
                v0, v1 = verts[i], verts[(i + 1) % len(verts)]
                d = ((v0[0] - v1[0]) ** 2 + (v0[1] - v1[1]) ** 2) ** 0.5
                edge_lengths.append(d)
            avg_edge = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 10

            jx = rng.uniform(-avg_edge * 0.3, avg_edge * 0.3)
            jy = rng.uniform(-avg_edge * 0.3, avg_edge * 0.3)
            pos = (cx + jx, cy + jy)

            min_x = min(v[0] for v in verts)
            max_x = max(v[0] for v in verts)
            min_y = min(v[1] for v in verts)
            max_y = max(v[1] for v in verts)
            pos = (
                max(min_x + 1, min(pos[0], max_x - 1)),
                max(min_y + 1, min(pos[1], max_y - 1)),
            )

            dcel.add_node(pos, face)

        random.setstate(state)
        return dcel

    @staticmethod
    def _polygon_area(verts: list[tuple[float, float]]) -> float:
        area = 0.0
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
