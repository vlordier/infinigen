# Urban Road Skeleton + District Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the random planar subdivision with a skeleton + district template architecture that produces realistic road networks matching historical/cultural city presets.

**Architecture:** CitySkeleton (major roads + superblocks) + DistrictTemplate.fill() (internal streets + lots per block) → combined RoadSegments → RoadToDCEL → existing pipeline.

**Tech Stack:** Python 3.11, pure math (no bpy needed), existing DCEL/RoadSegment/BuildingLot datatypes.

---

### File Structure

```
New:
  infinigen/assets/urban/road_to_dcel.py     # list[RoadSegment] → DCEL
  infinigen/assets/urban/skeleton.py          # CitySkeleton, BlockFace, all skeleton generators
  infinigen/assets/urban/template_utils.py    # Shared helpers (grid construction, clipping)
  infinigen/assets/urban/templates.py         # DistrictTemplateConfig, DistrictFill, all templates
  infinigen/assets/urban/city_presets.py      # CITY_PRESETS dict, load_preset()
  tests/test_road_to_dcel.py                  # RoadToDCEL + round-trip tests
  tests/test_skeleton.py                      # Skeleton generator tests
  tests/test_templates.py                     # Template tests

Modified:
  infinigen/assets/urban/compose_urban.py     # Use skeleton + templates
  infinigen/assets/urban/__init__.py          # Export new modules
  infinigen/assets/urban/graph_parser.py      # Minor: RoadSegment.source_id/target_id fields
  tests/render_urban.py                       # Accept --preset flag
```

---

### Task 1: RoadToDCEL

Converts a flat list of `RoadSegment` objects into a `DCEL` with proper half-edge topology and face extraction.

**Files:**
- Create: `infinigen/assets/urban/road_to_dcel.py`
- Create: `tests/test_road_to_dcel.py`

- [ ] **Step 1: Write failing round-trip test**

`tests/test_road_to_dcel.py`:
```python
from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.graph_parser import GraphParser
from infinigen.assets.urban.road_to_dcel import RoadToDCEL
from infinigen.assets.urban.dcel import DCEL


def test_roundtrip_preserves_segments():
    original = GraphGenerator.generate(100, 100, seed=42)
    parser = GraphParser(original)
    segments = parser.road_segments
    rebuilt = RoadToDCEL.build(segments)
    assert isinstance(rebuilt, DCEL)
    assert len(rebuilt.nodes) == len(original.nodes)
    assert len(rebuilt.half_edges) == len(original.half_edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/vincent/Work/infinigen:$PYTHONPATH pytest tests/test_road_to_dcel.py::test_roundtrip_preserves_segments -v --noconftest`
Expected: ImportError for RoadToDCEL

- [ ] **Step 3: Implement RoadToDCEL.build()**

`infinigen/assets/urban/road_to_dcel.py`:
```python
from infinigen.assets.urban.dcel import DCEL, DCELNode, DCEHalfEdge, DCEFace
from infinigen.assets.urban.graph_parser import RoadSegment


class RoadToDCEL:
    @staticmethod
    def build(segments: list[RoadSegment]) -> DCEL:
        dcel = DCEL()
        pos_to_node = {}
        node_to_outgoing: dict[int, list[tuple[float, float, RoadSegment]]] = {}
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
                (angle, tgt_node.position, seg)
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
                _, tgt_pos, _ = outgoing[i]
                _, next_tgt_pos, _ = outgoing[(i + 1) % n]
                he = he_map[(node.position, tgt_pos)]
                next_he = he_map[(next_tgt_pos, node.position)]
                he.next = next_he
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
```

Note: Need `import math` at top of file.

Also add a helper function `_set_face_half_edges(dcel)` that sets each face's `half_edge` to the first half-edge in its cycle.

```python
def _set_face_half_edges(dcel: DCEL):
    for face in dcel.faces:
        for he in dcel.half_edges:
            if he.face is face:
                face.half_edge = he
                break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/vincent/Work/infinigen:$PYTHONPATH pytest tests/test_road_to_dcel.py::test_roundtrip_preserves_segments -v --noconftest`
Expected: PASS

- [ ] **Step 5: Add additional RoadToDCEL tests**

In `tests/test_road_to_dcel.py`:

```python
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
```

- [ ] **Step 6: Run all RoadToDCEL tests**

Run: `PYTHONPATH=/Users/vincent/Work/infinigen:$PYTHONPATH pytest tests/test_road_to_dcel.py -v --noconftest`
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add infinigen/assets/urban/road_to_dcel.py tests/test_road_to_dcel.py
git commit -m "feat(urban): add RoadToDCEL - build DCEL from road segments"
```

---

### Task 2: Skeleton Data Types + Radial Generator

**Files:**
- Create: `infinigen/assets/urban/skeleton.py`
- Create: `tests/test_skeleton.py`
- Modify: `infinigen/assets/urban/__init__.py`

- [ ] **Step 1: Write failing radial skeleton test**

`tests/test_skeleton.py`:
```python
from infinigen.assets.urban.skeleton import RadialGenerator


def test_radial_skeleton_basic():
    result = RadialGenerator.generate(size=200, n_radials=8, n_rings=3, seed=42)
    assert len(result.road_segments) >= 8 * 3
    assert len(result.blocks) >= 8 * 3
    for block in result.blocks:
        assert len(block.boundary) >= 3
        assert isinstance(block.zone_id, str)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: ImportError for RadialGenerator

- [ ] **Step 3: Implement CitySkeleton + BlockFace + RadialGenerator**

`infinigen/assets/urban/skeleton.py`:

```python
import math
import random
from dataclasses import dataclass, field
from infinigen.assets.urban.graph_parser import RoadSegment


@dataclass
class BlockFace:
    boundary: list[tuple[float, float]]
    zone_id: str
    connection_nodes: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class CitySkeleton:
    road_segments: list[RoadSegment]
    blocks: list[BlockFace]


class RadialGenerator:
    @staticmethod
    def generate(size: float, n_radials: int = 8, n_rings: int = 4,
                 irregularity: float = 0.15, seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        cx = cy = size / 2
        max_r = size * 0.45
        road_segments = []
        blocks = []
        ring_distances = [max_r * (i + 1) / n_rings for i in range(n_rings)]
        base_angles = [2 * math.pi * i / n_radials for i in range(n_radials)]
        angles = [a + rng.uniform(-irregularity, irregularity) for a in base_angles]
        radials = []
        for ring_i in range(n_rings + 1):
            r = 0 if ring_i == 0 else ring_distances[ring_i - 1]
            ring = []
            for a in angles:
                jitter = r * rng.uniform(-irregularity * 0.3, irregularity * 0.3) if r > 0 else 0
                eff_r = r + jitter
                x = cx + eff_r * math.cos(a)
                y = cy + eff_r * math.sin(a)
                ring.append((x, y))
            radials.append(ring)
        for ring_i in range(1, n_rings + 1):
            for radial_i in range(n_radials):
                next_i = (radial_i + 1) % n_radials
                segment = RoadSegment(
                    source=radials[ring_i - 1][radial_i],
                    target=radials[ring_i][radial_i],
                    road_type="arterial" if ring_i <= 2 else "local",
                    lane_count=4 if ring_i <= 1 else 2,
                    width=24.0 if ring_i <= 1 else 12.0,
                    sidewalk=True,
                )
                road_segments.append(segment)
        for ring_i in range(n_rings):
            for radial_i in range(n_radials):
                next_i = (radial_i + 1) % n_radials
                a = radials[ring_i][radial_i]
                b = radials[ring_i][next_i]
                c = radials[ring_i + 1][next_i]
                d = radials[ring_i + 1][radial_i]
                segment = RoadSegment(
                    source=a, target=b,
                    road_type="ring",
                    lane_count=2, width=12.0, sidewalk=True,
                )
                road_segments.append(segment)
                if ring_i <= 2:
                    zone_id = "core" if ring_i == 0 else "inner" if ring_i == 1 else "outer"
                else:
                    zone_id = "outer"
                blocks.append(BlockFace(
                    boundary=[a, b, c, d],
                    zone_id=zone_id,
                ))
        boundary_ring = radials[-1]
        boundary_segments = []
        for i in range(n_radials):
            s = boundary_ring[i]
            t = boundary_ring[(i + 1) % n_radials]
            boundary_segments.append(RoadSegment(
                source=s, target=t, road_type="boundary",
                lane_count=2, width=12.0, sidewalk=True,
            ))
        road_segments.extend(boundary_segments)
        return CitySkeleton(road_segments=road_segments, blocks=blocks)
```

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS

- [ ] **Step 5: Add more radial skeleton tests**

In `tests/test_skeleton.py`:

```python
def test_radial_deterministic():
    a = RadialGenerator.generate(200, seed=42)
    b = RadialGenerator.generate(200, seed=42)
    assert len(a.road_segments) == len(b.road_segments)
    assert len(a.blocks) == len(b.blocks)

def test_radial_different_seed():
    a = RadialGenerator.generate(200, seed=42)
    b = RadialGenerator.generate(200, seed=99)
    assert len(a.road_segments) == len(b.road_segments)
```

- [ ] **Step 6: Run all skeleton tests**

Expected: 3 PASS

- [ ] **Step 7: Add skeleton to __init__.py**

In `infinigen/assets/urban/__init__.py`, add:
```python
from . import skeleton
```

- [ ] **Step 8: Commit**

```bash
git add infinigen/assets/urban/skeleton.py tests/test_skeleton.py infinigen/assets/urban/__init__.py
git commit -m "feat(urban): CitySkeleton + BlockFace + RadialGenerator"
```

---

### Task 3: Grid + OrganicSpine + SingleSpine Generators

**Files:**
- Modify: `infinigen/assets/urban/skeleton.py`
- Modify: `tests/test_skeleton.py`

- [ ] **Step 1: Write failing grid skeleton test**

In `tests/test_skeleton.py`:
```python
from infinigen.assets.urban.skeleton import GridGenerator, OrganicSpineGenerator, SingleSpineGenerator


def test_grid_skeleton():
    result = GridGenerator.generate(size=200, rows=5, cols=5, seed=42)
    assert len(result.blocks) == 5 * 5

def test_organic_spine():
    result = OrganicSpineGenerator.generate(size=200, seed=42)
    assert len(result.road_segments) >= 3
    assert len(result.blocks) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL with "can't import GridGenerator"

- [ ] **Step 3: Implement GridGenerator**

In `infinigen/assets/urban/skeleton.py`:

```python
class GridGenerator:
    @staticmethod
    def generate(size: float, rows: int = 5, cols: int = 5,
                 irregularity: float = 0.0, seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        spacing_x = size / cols
        spacing_y = size / rows
        nodes = {}
        segs = []
        blocks = []
        for r in range(rows + 1):
            for c in range(cols + 1):
                jx = rng.uniform(-irregularity * spacing_x, irregularity * spacing_x) if irregularity > 0 else 0
                jy = rng.uniform(-irregularity * spacing_y, irregularity * spacing_y) if irregularity > 0 else 0
                x = c * spacing_x + jx
                y = r * spacing_y + jy
                nodes[(c, r)] = (x, y)
        for r in range(rows + 1):
            for c in range(cols):
                segs.append(RoadSegment(
                    source=nodes[(c, r)], target=nodes[(c + 1, r)],
                    road_type="local", lane_count=2, width=12.0, sidewalk=True,
                ))
        for c in range(cols + 1):
            for r in range(rows):
                segs.append(RoadSegment(
                    source=nodes[(c, r)], target=nodes[(c, r + 1)],
                    road_type="local", lane_count=2, width=12.0, sidewalk=True,
                ))
        for r in range(rows):
            for c in range(cols):
                blocks.append(BlockFace(
                    boundary=[nodes[(c, r)], nodes[(c+1, r)], nodes[(c+1, r+1)], nodes[(c, r+1)]],
                    zone_id="inner",
                ))
        return CitySkeleton(road_segments=segs, blocks=blocks)
```

- [ ] **Step 4: Implement OrganicSpineGenerator**

```python
class OrganicSpineGenerator:
    @staticmethod
    def generate(size: float, n_branches: int = 8, irregularity: float = 0.4,
                 seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        segs = []
        blocks = []
        cx = size * 0.5
        cy = size * 0.5
        spine_pts = []
        x, y = cx - size * 0.3, cy
        for i in range(6):
            x += size * 0.12 + rng.uniform(-size * 0.03, size * 0.03)
            y += rng.uniform(-size * 0.05, size * 0.05)
            y = max(size * 0.1, min(size * 0.9, y))
            spine_pts.append((x, y))
        for i in range(len(spine_pts) - 1):
            segs.append(RoadSegment(
                source=spine_pts[i], target=spine_pts[i+1],
                road_type="arterial", lane_count=2, width=16.0, sidewalk=True,
            ))
        for i in range(0, len(spine_pts), max(1, len(spine_pts) // n_branches)):
            bx, by = spine_pts[i]
            angle = rng.uniform(-math.pi * 0.4, math.pi * 0.4)
            if rng.random() < 0.3:
                angle += math.pi
            length = rng.uniform(size * 0.1, size * 0.3)
            ex = bx + length * math.cos(angle)
            ey = by + length * math.sin(angle)
            segs.append(RoadSegment(
                source=(bx, by), target=(ex, ey),
                road_type="local", lane_count=2, width=12.0, sidewalk=True,
            ))
            if rng.random() < 0.4:
                lx = ex + rng.uniform(-20, 20)
                ly = ey + rng.uniform(-20, 20)
                segs.append(RoadSegment(
                    source=(ex, ey), target=(lx, ly),
                    road_type="alley", lane_count=1, width=5.0, sidewalk=False,
                ))
        blocks.append(BlockFace(
            boundary=[(0, 0), (size, 0), (size, size), (0, size)],
            zone_id="inner",
        ))
        return CitySkeleton(road_segments=segs, blocks=blocks)
```

- [ ] **Step 5: Implement SingleSpineGenerator**

```python
class SingleSpineGenerator:
    @staticmethod
    def generate(size: float, n_lanes: int = 6, seed: int = 0) -> CitySkeleton:
        rng = random.Random(seed)
        segs = []
        blocks = []
        spine_y = size * 0.5
        jitter = size * 0.02
        pts = []
        for i in range(4):
            x = size * (i + 0.5) / 4
            y = spine_y + rng.uniform(-jitter, jitter)
            pts.append((x, y))
        spine_pts = [(0, spine_y)] + pts + [(size, spine_y)]
        for i in range(len(spine_pts) - 1):
            segs.append(RoadSegment(
                source=spine_pts[i], target=spine_pts[i+1],
                road_type="local", lane_count=2, width=12.0, sidewalk=True,
            ))
        lane_spacing = size * 0.7 / n_lanes
        for i in range(1, len(spine_pts) - 1):
            sx, sy = spine_pts[i]
            for side in [-1, 1]:
                for li in range(n_lanes // 2):
                    ly = sy + side * (li + 1) * lane_spacing
                    if ly < 0 or ly > size:
                        continue
                    segs.append(RoadSegment(
                        source=(sx, sy), target=(sx, ly),
                        road_type="alley", lane_count=1, width=5.0, sidewalk=False,
                    ))
        blocks.append(BlockFace(
            boundary=[(0, 0), (size, 0), (size, size), (0, size)],
            zone_id="outer",
        ))
        return CitySkeleton(road_segments=segs, blocks=blocks)
```

- [ ] **Step 6: Run tests**

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add infinigen/assets/urban/skeleton.py tests/test_skeleton.py
git commit -m "feat(urban): GridGenerator, OrganicSpineGenerator, SingleSpineGenerator"
```

---

### Task 4: District Template Infrastructure + Utility Helpers

**Files:**
- Create: `infinigen/assets/urban/template_utils.py`
- Create: `infinigen/assets/urban/templates.py`
- Create: `tests/test_templates.py`

- [ ] **Step 1: Write template_utils tests**

`tests/test_templates.py`:
```python
from infinigen.assets.urban.template_utils import (
    make_grid_segments, clip_segments_to_boundary, bbox_lots,
)


def test_make_grid_segments():
    segs = make_grid_segments((0, 0), (100, 100), spacing=50)
    assert len(segs) >= 4

def test_clip_segments():
    segs = [((0, 0), (100, 0))]
    boundary = [(10, -10), (90, -10), (90, 10), (10, 10)]
    clipped = clip_segments_to_boundary(segs, boundary)
    assert len(clipped) >= 1

def test_bbox_lots():
    lots = bbox_lots((0, 0), (100, 100), lot_width=50, lot_depth=50)
    assert len(lots) == 4
```

- [ ] **Step 2: Implement template_utils**

`infinigen/assets/urban/template_utils.py`:
```python
import math
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.block_subdivision import BuildingLot


def make_grid_segments(bottom_left, top_right, spacing, road_type="local",
                       width=8.0, sidewalk=False, irregularity=0.0, rng=None):
    x0, y0 = bottom_left
    x1, y1 = top_right
    segs = []
    cols = max(1, int((x1 - x0) / spacing))
    rows = max(1, int((y1 - y0) / spacing))
    for c in range(cols + 1):
        x = x0 + c * spacing
        if rng and irregularity:
            x += rng.uniform(-irregularity, irregularity)
        segs.append(RoadSegment(
            source=(x, y0), target=(x, y1),
            road_type=road_type, lane_count=2, width=width, sidewalk=sidewalk,
        ))
    for r in range(rows + 1):
        y = y0 + r * spacing
        if rng and irregularity:
            y += rng.uniform(-irregularity, irregularity)
        segs.append(RoadSegment(
            source=(x0, y), target=(x1, y),
            road_type=road_type, lane_count=2, width=width, sidewalk=sidewalk,
        ))
    return segs


def clip_segments_to_boundary(segments, boundary):
    clipped = []
    for seg in segments:
        segs = _clip_line_to_polygon(seg.source, seg.target, boundary)
        for s, t in segs:
            clipped.append(RoadSegment(
                source=s, target=t,
                road_type=seg.road_type, lane_count=seg.lane_count,
                width=seg.width, sidewalk=seg.sidewalk,
            ))
    return clipped


def _clip_line_to_polygon(a, b, polygon):
    result = []
    inside_a = _point_in_polygon(a, polygon)
    inside_b = _point_in_polygon(b, polygon)
    if inside_a and inside_b:
        result.append((a, b))
        return result
    intersections = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        pt = _segment_intersection(a, b, p1, p2)
        if pt:
            intersections.append(pt)
    intersections.sort(key=lambda p: ((p[0]-a[0])**2 + (p[1]-a[1])**2))
    if inside_a and not inside_b and intersections:
        result.append((a, intersections[0]))
    elif not inside_a and inside_b and intersections:
        result.append((intersections[0], b))
    elif not inside_a and not inside_b and len(intersections) >= 2:
        result.append((intersections[0], intersections[1]))
    return result


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _segment_intersection(a, b, c, d):
    denom = ((b[0]-a[0])*(d[1]-c[1]) - (b[1]-a[1])*(d[0]-c[0]))
    if abs(denom) < 1e-10:
        return None
    t = ((c[0]-a[0])*(d[1]-c[1]) - (c[1]-a[1])*(d[0]-c[0])) / denom
    u = ((c[0]-a[0])*(b[1]-a[1]) - (c[1]-a[1])*(b[0]-a[0])) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
    return None


def bbox_lots(bottom_left, top_right, lot_width, lot_depth):
    x0, y0 = bottom_left
    x1, y1 = top_right
    lots = []
    y = y0
    while y < y1 - lot_depth * 0.5:
        x = x0
        while x < x1 - lot_width * 0.5:
            bx, by = x, y
            tx, ty = min(x + lot_width, x1), min(y + lot_depth, y1)
            lots.append(BuildingLot(
                boundary=[(bx, by), (tx, by), (tx, ty), (bx, ty)],
                area=(tx - bx) * (ty - by),
            ))
            x += lot_width
        y += lot_depth
    return lots
```

- [ ] **Step 3: Write the base template class + DistrictTemplateConfig + DistrictFill**

`infinigen/assets/urban/templates.py`:
```python
from dataclasses import dataclass, field
import random
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.block_subdivision import BuildingLot


@dataclass
class DistrictTemplateConfig:
    internal_road_width: float = 8.0
    internal_sidewalk: bool = False
    lot_depth: float = 30.0
    lot_width: float = 20.0
    lot_min_area: float = 20.0
    irregularity: float = 0.0
    dead_end_chance: float = 0.0
    density: float = 0.5


@dataclass
class DistrictFill:
    road_segments: list[RoadSegment] = field(default_factory=list)
    building_lots: list[BuildingLot] = field(default_factory=list)


class BaseTemplate:
    name = "base"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        raise NotImplementedError


def register_template(cls):
    _TEMPLATE_REGISTRY[cls.name] = cls
    return cls


_TEMPLATE_REGISTRY = {}


def get_template(name: str):
    return _TEMPLATE_REGISTRY.get(name)
```

- [ ] **Step 4: Run tests**

Run all template tests: `PYTHONPATH=/Users/vincent/Work/infinigen:$PYTHONPATH pytest tests/test_templates.py -v --noconftest`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add infinigen/assets/urban/template_utils.py infinigen/assets/urban/templates.py tests/test_templates.py
git commit -m "feat(urban): template infrastructure + utility helpers"
```

---

### Task 5: RectangularGrid + OrganicGrid Templates

- [ ] **Step 1: Write failing template tests**

In `tests/test_templates.py`:
```python
from infinigen.assets.urban.templates import (
    RectangularGridTemplate, OrganicGridTemplate, DistrictTemplateConfig,
)


def test_rectangular_grid_fill():
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    config = DistrictTemplateConfig(lot_depth=25, lot_width=25)
    rng = random.Random(42)
    result = RectangularGridTemplate.fill(boundary, config, rng)
    assert len(result.road_segments) >= 2
    assert len(result.building_lots) >= 4

def test_organic_grid_fill():
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    config = DistrictTemplateConfig(lot_depth=25, lot_width=25, irregularity=0.2)
    rng = random.Random(42)
    result = OrganicGridTemplate.fill(boundary, config, rng)
    assert len(result.road_segments) >= 2
```

- [ ] **Step 2: Implement RectangularGridTemplate**

In `infinigen/assets/urban/templates.py`:
```python
from infinigen.assets.urban.template_utils import make_grid_segments, bbox_lots


@register_template
class RectangularGridTemplate(BaseTemplate):
    name = "rectangular_grid"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = make_grid_segments(
            (x0, y0), (x1, y1),
            spacing=max(config.lot_depth, config.lot_width) * 2,
            road_type="local", width=config.internal_road_width,
            sidewalk=config.internal_sidewalk, rng=rng,
            irregularity=config.irregularity,
        )
        lots = bbox_lots(
            (x0 + 2, y0 + 2), (x1 - 2, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)
```

- [ ] **Step 3: Implement OrganicGridTemplate**

```python
@register_template
class OrganicGridTemplate(BaseTemplate):
    name = "organic_grid"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        spacing = max(config.lot_depth, config.lot_width) * 2
        segs = make_grid_segments(
            (x0, y0), (x1, y1), spacing=spacing,
            road_type="local", width=config.internal_road_width,
            sidewalk=config.internal_sidewalk,
            irregularity=config.irregularity, rng=rng,
        )
        from infinigen.assets.urban.template_utils import clip_segments_to_boundary
        segs = clip_segments_to_boundary(segs, boundary)
        dither = spacing * config.irregularity * 0.5 if config.irregularity else 0
        lots = bbox_lots(
            (x0 + dither + 2, y0 + dither + 2),
            (x1 - dither - 2, y1 - dither - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)
```

- [ ] **Step 4: Run tests**

Expected: 5 PASS (3 utility + 2 template)

- [ ] **Step 5: Commit**

```bash
git add infinigen/assets/urban/templates.py tests/test_templates.py
git commit -m "feat(urban): RectangularGrid + OrganicGrid templates"
```

---

### Task 6: Remaining Templates (MedievalOrganic, SuburbanCulDeSac, SovietBlock, GardenPlots, SparseOrganic)

- [ ] **Step 1: Write failing tests**

In `tests/test_templates.py`:
```python
def test_medieval_organic_fill():
    from infinigen.assets.urban.templates import MedievalOrganicTemplate
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    config = DistrictTemplateConfig(lot_depth=10, lot_width=8, density=0.8)
    result = MedievalOrganicTemplate.fill(boundary, config, random.Random(42))
    assert len(result.building_lots) >= 5

def test_soviet_block_fill():
    from infinigen.assets.urban.templates import SovietBlockTemplate
    boundary = [(0, 0), (200, 0), (200, 200), (0, 200)]
    config = DistrictTemplateConfig(lot_depth=100, lot_width=80, density=0.3)
    result = SovietBlockTemplate.fill(boundary, config, random.Random(42))
    assert len(result.building_lots) >= 1
```

- [ ] **Step 2: Implement MedievalOrganicTemplate**

```python
@register_template
class MedievalOrganicTemplate(BaseTemplate):
    name = "medieval_organic"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = []
        n_nodes = rng.randint(3, 6)
        nodes = []
        for _ in range(n_nodes):
            nx = rng.uniform(x0 + 5, x1 - 5)
            ny = rng.uniform(y0 + 5, y1 - 5)
            nodes.append((nx, ny))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if rng.random() < 0.5:
                    segs.append(RoadSegment(
                        source=nodes[i], target=nodes[j],
                        road_type="local", lane_count=1, width=config.internal_road_width,
                        sidewalk=False,
                    ))
        if rng.random() < config.dead_end_chance:
            from_node = rng.choice(nodes)
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(5, 20)
            dead_end = (from_node[0] + dist * math.cos(angle),
                        from_node[1] + dist * math.sin(angle))
            segs.append(RoadSegment(
                source=from_node, target=dead_end,
                road_type="alley", lane_count=1, width=4.0, sidewalk=False,
            ))
        lots = _voronoi_lots(nodes, boundary, rng)
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


def _voronoi_lots(points, boundary, rng, n_samples=500):
    from infinigen.assets.urban.template_utils import _point_in_polygon
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    assignments = {}
    for _ in range(n_samples):
        px = rng.uniform(x0, x1)
        py = rng.uniform(y0, y1)
        if not _point_in_polygon((px, py), boundary):
            continue
        best = min(range(len(points)), key=lambda i: (px-points[i][0])**2 + (py-points[i][1])**2)
        assignments.setdefault(best, []).append((px, py))
    lots = []
    for region_id, pts in assignments.items():
        if len(pts) < 3:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
        hull = _convex_hull(sorted_pts)
        if len(hull) < 3:
            continue
        area = _polygon_area(hull)
        lots.append(BuildingLot(boundary=hull, area=area))
    return lots


def _convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def _polygon_area(verts):
    area = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % n]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0
```

Note: Need `import math` in templates.py.

- [ ] **Step 3: Implement SovietBlockTemplate**

```python
@register_template
class SovietBlockTemplate(BaseTemplate):
    name = "soviet_block"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = x1 - x0
        bh = y1 - y0
        n_cols = max(1, int(bw / config.lot_width))
        n_rows = max(1, int(bh / config.lot_depth))
        segs = []
        for i in range(1, n_cols):
            x = x0 + i * bw / n_cols
            segs.append(RoadSegment(
                source=(x, y0), target=(x, y1),
                road_type="local", lane_count=2, width=config.internal_road_width,
                sidewalk=False,
            ))
        lots = []
        for r in range(n_rows):
            for c in range(n_cols):
                lx = x0 + c * bw / n_cols + 2
                ly = y0 + r * bh / n_rows + 2
                rx = x0 + (c + 1) * bw / n_cols - 2
                ry = y0 + (r + 1) * bh / n_rows - 2
                area = (rx - lx) * (ry - ly)
                if area >= config.lot_min_area:
                    lots.append(BuildingLot(
                        boundary=[(lx, ly), (rx, ly), (rx, ry), (lx, ry)],
                        area=area, building_type="industrial",
                    ))
        return DistrictFill(road_segments=segs, building_lots=lots)
```

- [ ] **Step 4: Implement SuburbanCulDeSacTemplate**

```python
@register_template
class SuburbanCulDeSacTemplate(BaseTemplate):
    name = "suburban_cul_de_sac"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = []
        cx = (x0 + x1) / 2
        spine = [(cx, y0 + 5), (cx, y1 - 5)]
        segs.append(RoadSegment(
            source=spine[0], target=spine[1],
            road_type="local", lane_count=2, width=config.internal_road_width,
            sidewalk=config.internal_sidewalk,
        ))
        for side in [-1, 1]:
            for i in range(rng.randint(2, 4)):
                t = rng.uniform(0.2, 0.8)
                sy = y0 + t * (y1 - y0)
                length = rng.uniform(config.lot_depth, config.lot_depth * 2)
                ex = cx + side * length
                segs.append(RoadSegment(
                    source=(cx, sy), target=(ex, sy),
                    road_type="local", lane_count=2, width=config.internal_road_width,
                    sidewalk=False,
                ))
        lots = bbox_lots(
            (x0 + 2, y0 + 2), (cx - 5, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        ) + bbox_lots(
            (cx + 5, y0 + 2), (x1 - 2, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)
```

- [ ] **Step 5: Implement GardenPlotsTemplate + SparseOrganicTemplate**

```python
@register_template
class GardenPlotsTemplate(BaseTemplate):
    name = "garden_plots"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = []
        cx = (x0 + x1) / 2
        segs.append(RoadSegment(
            source=(cx, y0), target=(cx, y1),
            road_type="local", lane_count=1, width=config.internal_road_width,
            sidewalk=False,
        ))
        plot_width = config.lot_width
        n_plots = max(1, int((x1 - x0) / 2 / plot_width))
        for side in [-1, 1]:
            for i in range(n_plots):
                px = cx + side * (i * plot_width + 2)
                if px < x0 + 2 or px > x1 - 2:
                    continue
                segs.append(RoadSegment(
                    source=(px, y0), target=(px, y1),
                    road_type="alley", lane_count=1, width=3.0, sidewalk=False,
                ))
        lots = []
        for side in [-1, 1]:
            for i in range(n_plots):
                px = cx + side * (i * plot_width + 2)
                if px < x0 + 2 or px > x1 - 2:
                    continue
                lx = min(px, cx) if side == -1 else px
                rx = max(px, cx) if side == -1 else px + plot_width
                lots.append(BuildingLot(
                    boundary=[(lx, y0 + 2), (rx, y0 + 2), (rx, y1 - 2), (lx, y1 - 2)],
                    area=(rx - lx) * (y1 - y0 - 4),
                    building_type="residential",
                ))
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class SparseOrganicTemplate(BaseTemplate):
    name = "sparse_organic"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        lots = []
        n_lots = rng.randint(1, 3)
        for _ in range(n_lots):
            lx = rng.uniform(x0 + 5, x1 - config.lot_width - 5)
            ly = rng.uniform(y0 + 5, y1 - config.lot_depth - 5)
            w = config.lot_width * rng.uniform(0.8, 1.2)
            h = config.lot_depth * rng.uniform(0.8, 1.2)
            lots.append(BuildingLot(
                boundary=[(lx, ly), (lx + w, ly), (lx + w, ly + h), (lx, ly + h)],
                area=w * h,
                building_type="residential",
            ))
        return DistrictFill(road_segments=[], building_lots=lots)
```

- [ ] **Step 6: Run all template tests**

Expected: 7 PASS (3 utility + 4 template)

- [ ] **Step 7: Commit**

```bash
git add infinigen/assets/urban/templates.py tests/test_templates.py
git commit -m "feat(urban): MedievalOrganic, SovietBlock, SuburbanCulDeSac, GardenPlots, SparseOrganic templates"
```

---

### Task 7: City Presets + Integration

**Files:**
- Create: `infinigen/assets/urban/city_presets.py`
- Modify: `infinigen/assets/urban/compose_urban.py`
- Modify: `infinigen/assets/urban/__init__.py`
- Modify: `tests/render_urban.py`

- [ ] **Step 1: Implement CITY_PRESETS + load_preset**

`infinigen/assets/urban/city_presets.py`:
```python
from infinigen.assets.urban.templates import DistrictTemplateConfig


CITY_PRESETS = {
    "european_old": {
        "skeleton_type": "radial",
        "skeleton_params": {"n_radials": 10, "n_rings": 5, "irregularity": 0.2},
        "zone_templates": {
            "core":  {"template": "organic_grid",  "config": DistrictTemplateConfig(lot_width=15, lot_depth=20, irregularity=0.15, internal_road_width=6.0)},
            "inner": {"template": "organic_grid",  "config": DistrictTemplateConfig(lot_width=20, lot_depth=25, irregularity=0.1, internal_road_width=8.0)},
            "outer": {"template": "rectangular_grid", "config": DistrictTemplateConfig(lot_width=25, lot_depth=30, irregularity=0.05, internal_road_width=10.0)},
        },
        "regional_style": "mediterranean",
    },
    "medieval_village": {
        "skeleton_type": "organic_spine",
        "skeleton_params": {"n_branches": 6, "irregularity": 0.5},
        "zone_templates": {
            "inner": {"template": "medieval_organic", "config": DistrictTemplateConfig(lot_width=8, lot_depth=10, lot_min_area=20, dead_end_chance=0.3, density=0.9, internal_road_width=4.0)},
        },
        "regional_style": "mediterranean",
    },
    "suburban_estonia": {
        "skeleton_type": "grid",
        "skeleton_params": {"rows": 4, "cols": 4, "irregularity": 0.1},
        "zone_templates": {
            "inner": {"template": "suburban_cul_de_sac", "config": DistrictTemplateConfig(lot_width=30, lot_depth=40, lot_min_area=500, internal_road_width=8.0, density=0.3)},
        },
        "regional_style": "baltic",
    },
    "ukrainian_city": {
        "skeleton_type": "grid",
        "skeleton_params": {"rows": 6, "cols": 6, "irregularity": 0.05},
        "zone_templates": {
            "inner": {"template": "rectangular_grid", "config": DistrictTemplateConfig(lot_width=40, lot_depth=50, lot_min_area=500, internal_road_width=16.0)},
        },
        "regional_style": "soviet",
    },
    "ukrainian_village": {
        "skeleton_type": "single_spine",
        "skeleton_params": {"n_lanes": 8},
        "zone_templates": {
            "outer": {"template": "garden_plots", "config": DistrictTemplateConfig(lot_width=15, lot_depth=40, lot_min_area=300, internal_road_width=5.0)},
        },
        "regional_style": "soviet",
    },
    "soviet_microdistrict": {
        "skeleton_type": "radial",
        "skeleton_params": {"n_radials": 6, "n_rings": 3, "irregularity": 0.05},
        "zone_templates": {
            "core":  {"template": "soviet_block", "config": DistrictTemplateConfig(lot_width=80, lot_depth=100, internal_road_width=24.0, density=0.3)},
            "inner": {"template": "soviet_block", "config": DistrictTemplateConfig(lot_width=60, lot_depth=80, internal_road_width=18.0, density=0.4)},
            "outer": {"template": "sparse_organic", "config": DistrictTemplateConfig(lot_width=50, lot_depth=50, lot_min_area=1000)},
        },
        "regional_style": "soviet",
    },
}


def load_preset(name: str) -> dict:
    if name not in CITY_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(CITY_PRESETS.keys())}")
    return dict(CITY_PRESETS[name])
```

- [ ] **Step 2: Update __init__.py**

In `infinigen/assets/urban/__init__.py`, add:
```python
from . import road_to_dcel
from . import template_utils
from . import templates
from . import city_presets
```

- [ ] **Step 3: Update compose_urban to accept presets**

Replace the body of `compose_urban` in `infinigen/assets/urban/compose_urban.py` to:
1. Accept a `preset_name` param (default: "european_old")
2. Load preset via `city_presets.load_preset()`
3. Instantiate the right skeleton generator based on `preset["skeleton_type"]`
4. For each block, look up zone template and call fill()
5. Combine all segments → RoadToDCEL.build() → existing pipeline
6. Combine all lots → BuildingGenerator (skip subdivide_lots for filled blocks)

Note: Since `compose_urban` imports from `infinigen.core.util.pipeline` which requires Blender, keep the bpy-only constraint. The skeleton generators and templates are already testable without Blender.

- [ ] **Step 4: Update render_urban.py**

Add `--preset` argument and pass it to compose_urban. Render a city for each preset.

- [ ] **Step 5: Commit**

```bash
git add infinigen/assets/urban/city_presets.py infinigen/assets/urban/__init__.py infinigen/assets/urban/compose_urban.py tests/render_urban.py
git commit -m "feat(urban): city presets + compose_urban integration"
```

---

### Self-Review Checklist

1. **Spec coverage:** Does every section in the spec correspond to a task in the plan?
   - Skeleton types ✓ (Task 2-3)
   - District templates ✓ (Task 4-6)
   - RoadToDCEL ✓ (Task 1)
   - CityPresets ✓ (Task 7)
   - Integration ✓ (Task 7)
   - Testing ✓ (all tasks)

2. **Placeholder scan:** No TBD, TODO, or incomplete steps.

3. **Type consistency:** RoadSegment, BuildingLot, DCEL, BlockFace, CitySkeleton all match across tasks.
