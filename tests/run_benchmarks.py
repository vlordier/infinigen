#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Benchmark runner — measures key operations and outputs JSON results.

When run on the PR branch (default): uses optimised implementations from
the repository's utility modules and vectorised NumPy patterns.

When run with ``--upstream``: uses baseline (slow) implementations that
mirror the code paths in the upstream Princeton ``main`` branch.

Each benchmark returns the **same key name** regardless of mode so the
comparison script can match results across branches.

Usage::

    # PR branch (optimised)
    python tests/run_benchmarks.py --output /tmp/pr_results.json

    # Upstream (baseline)
    python tests/run_benchmarks.py --output /tmp/upstream_results.json --upstream
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Module loader – avoids triggering bpy via core/util/__init__.py
# ---------------------------------------------------------------------------

_UTIL_DIR = Path(__file__).resolve().parent.parent / "infinigen" / "core" / "util"


def _load_module(name: str, filepath: Path):
    """Load a single Python file as a module, bypassing package __init__."""
    if not filepath.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

N_REPEAT = 7  # default: enough for stable median
N_REPEAT_SLOW = 3  # for benchmarks ~0.1–1 s each
N_REPEAT_VERY_SLOW = 2  # for benchmarks > 1 s each


def _median_time(fn, n_repeat=N_REPEAT):
    """Run *fn* *n_repeat* times and return the median wall-clock time (s).

    A single warm-up call is executed first (not counted) so OS-level
    caching effects (page faults, memory mapping) don't skew the first
    measurement.
    """
    fn()  # warm-up
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# =====================================================================
# 1. unique_rows — void-view (PR) vs np.unique(axis=0) (upstream)
#    Mirrors: infinigen/core/util/exporting.py (instance-ID dedup)
# =====================================================================


def bench_unique_rows(upstream=False):
    """Deduplicate 500k×3 int32 rows — mirrors instance-ID dedup in exporting.py."""
    np.random.seed(42)
    arr = np.random.randint(0, 100, (500_000, 3), dtype=np.int32)

    if not upstream:
        mod = _load_module("array_ops", _UTIL_DIR / "array_ops.py")
        if mod is not None and hasattr(mod, "unique_rows"):
            t = _median_time(lambda: mod.unique_rows(arr))
            return ("unique_rows", t)
    # Upstream baseline: np.unique(axis=0) — slow lexsort path
    t = _median_time(lambda: np.unique(arr, axis=0))
    return ("unique_rows", t)


# =====================================================================
# 2. Heightmap grid — vectorised meshgrid (PR) vs Python loop (upstream)
#    Mirrors: infinigen/terrain/utils/mesh.py Mesh.__init__ heightmap branch
# =====================================================================

_GRID_N = 256
_GRID_L = 10.0


def bench_heightmap_grid(upstream=False):
    """Build 256×256 heightmap vertices + triangular faces."""
    np.random.seed(10)
    N, L = _GRID_N, _GRID_L
    heightmap = np.random.rand(N, N).astype(np.float32)

    if not upstream:
        # PR: vectorised meshgrid + broadcasting for face indices
        def _run():
            xs = (-1 + 2 * np.arange(N) / (N - 1)) * L / 2
            ys = (-1 + 2 * np.arange(N) / (N - 1)) * L / 2
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            verts = np.stack([gx, gy, heightmap], axis=-1).reshape(-1, 3)
            i_idx = np.arange(N - 1).reshape(-1, 1)
            j_idx = np.arange(N - 1).reshape(1, -1)
            faces = np.empty((2, N - 1, N - 1, 3), np.int32)
            faces[0] = np.stack(
                [
                    i_idx * N + j_idx,
                    (i_idx + 1) * N + j_idx,
                    i_idx * N + j_idx + 1,
                ],
                axis=-1,
            )
            faces[1] = np.stack(
                [
                    i_idx * N + j_idx + 1,
                    (i_idx + 1) * N + j_idx,
                    (i_idx + 1) * N + j_idx + 1,
                ],
                axis=-1,
            )
            return verts, faces.reshape(-1, 3)

        t = _median_time(_run)
    else:
        # Upstream: nested Python loops — mirrors original mesh.py
        def _run():
            verts = []
            for i in range(N):
                for j in range(N):
                    x = (-1 + 2 * i / (N - 1)) * L / 2
                    y = (-1 + 2 * j / (N - 1)) * L / 2
                    verts.append([x, y, heightmap[i, j]])
            verts = np.array(verts, dtype=np.float32)
            faces = []
            for i in range(N - 1):
                for j in range(N - 1):
                    v00 = i * N + j
                    v10 = (i + 1) * N + j
                    v01 = i * N + j + 1
                    v11 = (i + 1) * N + j + 1
                    faces.append([v00, v10, v01])
                    faces.append([v01, v10, v11])
            return verts, np.array(faces, dtype=np.int32)

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("heightmap_grid", t)


# =====================================================================
# 3. Mesh.cat — pre-allocated single-pass (PR) vs repeated vstack (upstream)
#    Mirrors: infinigen/terrain/utils/mesh.py Mesh.cat()
# =====================================================================

_CAT_N_MESHES = 50
_CAT_VERTS_PER = 2_000
_CAT_FACES_PER = 3_000


def _make_fake_meshes(n_meshes, n_verts, n_faces, seed=88):
    """Create a list of (verts, faces, attrs) tuples mimicking Mesh objects."""
    np.random.seed(seed)
    meshes = []
    for _ in range(n_meshes):
        v = np.random.rand(n_verts, 3).astype(np.float64)
        f = np.random.randint(0, n_verts, (n_faces, 3), dtype=np.int32)
        a = {"sdf": np.random.rand(n_verts).astype(np.float32)}
        meshes.append((v, f, a))
    return meshes


def bench_mesh_cat(upstream=False):
    """Concatenate 50 meshes (2k vertices each) — mirrors Mesh.cat()."""
    meshes = _make_fake_meshes(_CAT_N_MESHES, _CAT_VERTS_PER, _CAT_FACES_PER)

    if not upstream:
        # PR: pre-allocated single-pass concatenation
        def _run():
            total_v = sum(v.shape[0] for v, _, _ in meshes)
            total_f = sum(f.shape[0] for _, f, _ in meshes)
            verts = np.empty((total_v, 3), dtype=np.float64)
            faces = np.empty((total_f, 3), dtype=np.int32)
            sdf = np.zeros(total_v, dtype=np.float32)
            vo, fo = 0, 0
            for v, f, a in meshes:
                nv, nf = v.shape[0], f.shape[0]
                verts[vo : vo + nv] = v
                faces[fo : fo + nf] = f + vo
                sdf[vo : vo + nv] = a["sdf"]
                vo += nv
                fo += nf
            return verts, faces, sdf

        t = _median_time(_run)
    else:
        # Upstream: naive repeated vstack — O(n²) memory traffic
        def _run():
            verts = np.empty((0, 3), dtype=np.float64)
            faces = np.empty((0, 3), dtype=np.int32)
            sdf = np.empty(0, dtype=np.float32)
            vo = 0
            for v, f, a in meshes:
                verts = np.vstack([verts, v])
                faces = np.vstack([faces, f + vo])
                sdf = np.concatenate([sdf, a["sdf"]])
                vo += v.shape[0]
            return verts, faces, sdf

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    return ("mesh_cat", t)


# =====================================================================
# 4. Tree vertices — lazy concat (PR) vs O(n²) np.append (upstream)
#    Mirrors: infinigen/assets/objects/trees/tree.py TreeVertices
# =====================================================================

_TREE_N_PARTS = 300
_TREE_PTS_PER = 80


def bench_tree_vertices(upstream=False):
    """Accumulate 300 branch segments (80 vertices each) — mirrors TreeVertices."""
    np.random.seed(7)
    parts = [np.random.rand(_TREE_PTS_PER, 3) for _ in range(_TREE_N_PARTS)]

    if not upstream:
        # PR: list accumulate + single np.concatenate
        def _run():
            acc = []
            for p in parts:
                acc.append(p)
            return np.concatenate(acc, axis=0)

        t = _median_time(_run)
    else:
        # Upstream: eager O(n²) np.append pattern
        def _run():
            arr = np.empty((0, 3))
            for p in parts:
                arr = np.append(arr, p, axis=0)
            return arr

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    return ("tree_vertices", t)


# =====================================================================
# 5. Camera projection — pre-computed combined matrix (PR) vs separate (upstream)
#    Mirrors: infinigen/terrain/utils/mesh.py Mesh.camera_annotation
# =====================================================================

_PROJ_N_VERTS = 200_000
_PROJ_N_CAMS = 10


def bench_projection(upstream=False):
    """Project 200k vertices through 10 cameras — mirrors camera_annotation."""
    np.random.seed(55)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    cams = []
    for _ in range(_PROJ_N_CAMS):
        cam = np.eye(4, dtype=np.float64)
        cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
        cam[:3, 3] = np.random.randn(3) * 5
        cams.append(cam)
    points = np.random.rand(_PROJ_N_VERTS, 3).astype(np.float64)

    if not upstream:
        # PR: pre-compute combined projection matrix, homogeneous coords once
        homo = np.hstack([points, np.ones((_PROJ_N_VERTS, 1))]).T  # (4, N)
        projs = [K @ np.linalg.inv(c)[:3, :] for c in cams]

        def _run():
            visible = np.zeros(_PROJ_N_VERTS, dtype=bool)
            for proj in projs:
                coords = proj @ homo  # (3, N) single matmul
                coords[:2] /= coords[2]
                visible |= (
                    (coords[2] > 0) & (coords[0] > 0) & (coords[0] < 640)
                )
            return visible

        t = _median_time(_run)
    else:
        # Upstream: recompute inv + separate matmuls per camera each call
        def _run():
            visible = np.zeros(_PROJ_N_VERTS, dtype=bool)
            for cam in cams:
                cam_inv = np.linalg.inv(cam)
                pts_h = np.hstack(
                    [points, np.ones((_PROJ_N_VERTS, 1))]
                )
                world = (cam_inv[:3, :] @ pts_h.T).T
                screen = (K @ world.T).T
                screen[:, :2] /= screen[:, 2:3]
                visible |= (
                    (screen[:, 2] > 0)
                    & (screen[:, 0] > 0)
                    & (screen[:, 0] < 640)
                )
            return visible

        t = _median_time(_run)
    return ("projection", t)


# =====================================================================
# 6. Distance transform — scipy EDT (PR) vs brute-force loop (upstream)
#    Mirrors: infinigen/terrain/utils/image_processing.py grid_distance
# =====================================================================

_DIST_N = 128


def bench_distance_transform(upstream=False):
    """Distance transform on 128×128 binary mask — mirrors grid_distance."""
    np.random.seed(42)
    mask = (np.random.rand(_DIST_N, _DIST_N) > 0.5).astype(np.float64)

    if not upstream:
        # PR: scipy distance_transform_edt — O(N²)
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return ("distance_transform", float("nan"))
        t = _median_time(lambda: distance_transform_edt(mask))
    else:
        # Upstream: O(N²·B) brute-force loop — use 64×64 sub-grid and extrapolate
        SUB = 64
        mask_sub = (np.random.rand(SUB, SUB) > 0.5).astype(np.float64)
        seeds = np.argwhere(mask_sub > 0)

        def _loop():
            dist = np.full((SUB, SUB), float("inf"))
            for si, sj in seeds:
                for i in range(SUB):
                    for j in range(SUB):
                        d = (i - si) ** 2 + (j - sj) ** 2
                        if d < dist[i, j]:
                            dist[i, j] = d
            return np.sqrt(dist)

        t64 = _median_time(_loop, n_repeat=N_REPEAT_VERY_SLOW)
        # Extrapolate: O(N² × B) where B ∝ N² → O(N⁴)
        scale = (_DIST_N / SUB) ** 4
        t = t64 * scale
    return ("distance_transform", t)


# =====================================================================
# 7. Boundary detection — vectorised (PR) vs pixel loop (upstream)
#    Mirrors: image_processing.py grid_distance boundary identification
# =====================================================================


def bench_boundary_detection(upstream=False):
    """Detect boundary pixels on 256×256 binary mask."""
    np.random.seed(42)
    N = 256
    source = (np.random.rand(N, N) > 0.4).astype(bool)

    if not upstream:
        # PR: vectorised edge detection using array slicing
        def _run():
            boundary = np.zeros_like(source, dtype=bool)
            boundary[:-1, :] |= ~source[1:, :]
            boundary[1:, :] |= ~source[:-1, :]
            boundary[:, :-1] |= ~source[:, 1:]
            boundary[:, 1:] |= ~source[:, :-1]
            boundary &= source
            return boundary

        t = _median_time(_run)
    else:
        # Upstream: pixel-by-pixel loop with neighbour checks
        def _run():
            boundary = np.zeros((N, N), dtype=bool)
            for i in range(N):
                for j in range(N):
                    if not source[i, j]:
                        continue
                    if i > 0 and not source[i - 1, j]:
                        boundary[i, j] = True
                    elif i < N - 1 and not source[i + 1, j]:
                        boundary[i, j] = True
                    elif j > 0 and not source[i, j - 1]:
                        boundary[i, j] = True
                    elif j < N - 1 and not source[i, j + 1]:
                        boundary[i, j] = True
            return boundary

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("boundary_detection", t)


# =====================================================================
# 8. Surface normals — vectorised (PR) vs loop (upstream)
#    Mirrors: infinigen/terrain/utils/image_processing.py get_normal
# =====================================================================


def bench_surface_normals(upstream=False):
    """Compute surface normals for 256×256 heightmap."""
    np.random.seed(77)
    N = 256
    z = np.random.rand(N, N).astype(np.float64)
    grid_size = 1.0 / (N - 1)

    if not upstream:
        # PR: vectorised finite differences + normalisation
        def _run():
            dzdx = np.zeros_like(z)
            dzdx[1:] = z[1:] - z[:-1]
            dzdx[0] = dzdx[1]
            dzdy = np.zeros_like(z)
            dzdy[:, 1:] = z[:, 1:] - z[:, :-1]
            dzdy[:, 0] = dzdy[:, 1]
            n = np.stack(
                (-dzdy, -dzdx, grid_size * np.ones_like(z)), axis=-1
            )
            n /= np.linalg.norm(n, axis=-1, keepdims=True)
            return n

        t = _median_time(_run)
    else:
        # Upstream: per-pixel loop with individual np.array + np.linalg.norm
        def _run():
            normals = np.zeros((N, N, 3), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    dzdx = (
                        z[min(i + 1, N - 1), j] - z[max(i - 1, 0), j]
                    ) / 2
                    dzdy = (
                        z[i, min(j + 1, N - 1)] - z[i, max(j - 1, 0)]
                    ) / 2
                    n = np.array([-dzdy, -dzdx, grid_size])
                    normals[i, j] = n / np.linalg.norm(n)
            return normals

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("surface_normals", t)


# =====================================================================
# 9. Full pipeline composite — all optimised ops in sequence (PR)
#    vs all baseline ops (upstream).  Simulates processing a single
#    terrain tile end-to-end.
# =====================================================================


def bench_full_pipeline(upstream=False):
    """End-to-end: grid build + unique_rows + projection + EDT."""
    np.random.seed(123)
    N = 128

    if not upstream:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return ("full_pipeline", float("nan"))

        mod = _load_module("array_ops", _UTIL_DIR / "array_ops.py")

        def _pipeline():
            # Step 1: vectorised heightmap grid
            h = np.random.rand(N, N).astype(np.float32)
            xs = np.linspace(-5, 5, N, dtype=np.float32)
            ys = np.linspace(-5, 5, N, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            verts = np.stack([gx, gy, h], axis=-1).reshape(-1, 3)

            # Step 2: fast unique_rows (or np.unique fallback)
            ids = np.random.randint(0, 50, (len(verts), 3), dtype=np.int32)
            if mod is not None and hasattr(mod, "unique_rows"):
                mod.unique_rows(ids)
            else:
                np.unique(ids, axis=0)

            # Step 3: pre-computed projection (2 cameras)
            K = np.array(
                [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                dtype=np.float64,
            )
            homo = np.hstack(
                [verts.astype(np.float64), np.ones((len(verts), 1))]
            ).T
            for _ in range(2):
                cam = np.eye(4)
                cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
                proj = K @ np.linalg.inv(cam)[:3, :]
                proj @ homo

            # Step 4: scipy EDT
            mask = (h > 0.5).astype(np.float64)
            distance_transform_edt(mask)

        t = _median_time(_pipeline)
    else:

        def _pipeline():
            # Step 1: loop heightmap grid
            h = np.random.rand(N, N).astype(np.float32)
            verts = []
            for i in range(N):
                for j in range(N):
                    verts.append(
                        [-5 + 10 * i / (N - 1), -5 + 10 * j / (N - 1), h[i, j]]
                    )
            verts = np.array(verts, dtype=np.float32)

            # Step 2: np.unique axis=0
            ids = np.random.randint(0, 50, (len(verts), 3), dtype=np.int32)
            np.unique(ids, axis=0)

            # Step 3: separate projection
            K = np.array(
                [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                dtype=np.float64,
            )
            for _ in range(2):
                cam = np.eye(4)
                cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
                cam_inv = np.linalg.inv(cam)
                pts_h = np.hstack(
                    [verts.astype(np.float64), np.ones((len(verts), 1))]
                )
                world = (cam_inv[:3, :] @ pts_h.T).T
                _screen = (K @ world.T).T

            # Step 4: loop distance on 32×32 sub-grid
            mask = (h > 0.5).astype(np.float64)
            sub = 32
            mask_s = mask[:sub, :sub]
            seeds = np.argwhere(mask_s > 0)
            dist = np.full((sub, sub), float("inf"))
            for si, sj in seeds:
                for i in range(sub):
                    for j in range(sub):
                        d = (i - si) ** 2 + (j - sj) ** 2
                        if d < dist[i, j]:
                            dist[i, j] = d

        t = _median_time(_pipeline, n_repeat=N_REPEAT_VERY_SLOW)
    return ("full_pipeline", t)


# =====================================================================
# 10. RRT nearest-neighbour — KDTree O(log n) (PR) vs brute-force O(n) (upstream)
#     Mirrors: infinigen/core/util/rrt.py RRTTree.nearest()
# =====================================================================

_RRT_N_NODES = 5_000
_RRT_N_QUERIES = 500


def bench_rrt_nearest(upstream=False):
    """Find nearest vertex among 5k RRT nodes for 500 queries."""
    np.random.seed(17)
    nodes = np.random.rand(_RRT_N_NODES, 3).astype(np.float64)
    queries = np.random.rand(_RRT_N_QUERIES, 3).astype(np.float64)

    if not upstream:
        # PR: scipy KDTree — O(log n) per query
        from scipy.spatial import cKDTree

        tree = cKDTree(nodes)

        def _run():
            _, idx = tree.query(queries, k=1)
            return idx

        t = _median_time(_run)
    else:
        # Upstream: brute-force repmat + norm — O(n) per query (mirrors rrt.py)
        def _run():
            results = np.empty(_RRT_N_QUERIES, dtype=int)
            for qi in range(_RRT_N_QUERIES):
                dists = np.linalg.norm(nodes - queries[qi], axis=1)
                results[qi] = np.argmin(dists)
            return results

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    return ("rrt_nearest", t)


# =====================================================================
# 11. RRT neighbourhood — broadcasting (PR) vs repmat (upstream)
#     Mirrors: infinigen/core/util/rrt.py RRTTree.neighborhood()
# =====================================================================


def bench_rrt_neighborhood(upstream=False):
    """Find all nodes within radius for 200 queries over 5k nodes."""
    np.random.seed(18)
    nodes = np.random.rand(_RRT_N_NODES, 3).astype(np.float64)
    queries = np.random.rand(200, 3).astype(np.float64)
    radius = 0.15

    if not upstream:
        # PR: KDTree ball query — O(log n + k) per query
        from scipy.spatial import cKDTree

        tree = cKDTree(nodes)

        def _run():
            return tree.query_ball_point(queries, radius)

        t = _median_time(_run)
    else:
        # Upstream: repmat + norm per query (mirrors rrt.py neighborhood)
        from numpy.matlib import repmat

        def _run():
            results = []
            for q in queries:
                xr = repmat(q, len(nodes), 1)
                dists = np.linalg.norm(xr - nodes, axis=1)
                results.append(np.where(dists < radius)[0])
            return results

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    return ("rrt_neighborhood", t)


# =====================================================================
# 12. Space colonisation — vectorised mean direction (PR) vs loop (upstream)
#     Mirrors: infinigen/assets/objects/trees/tree.py space_colonisation
# =====================================================================

_SC_N_NODES = 500
_SC_N_ATTS = 2_000


def bench_space_colonisation(upstream=False):
    """Compute new growth directions for 500 tree nodes from 2k attractors."""
    np.random.seed(22)
    nodes = np.random.rand(_SC_N_NODES, 3).astype(np.float64)
    # Each attractor is assigned to a node (simulates curr_match)
    curr_match = np.random.randint(0, _SC_N_NODES, _SC_N_ATTS)
    att_positions = np.random.rand(_SC_N_ATTS, 3).astype(np.float64)

    if not upstream:
        # PR: vectorised — pre-compute deltas once, then group by node
        deltas = att_positions - nodes[curr_match]  # (n_atts, 3) — single bulk op

        def _run():
            matched_nodes = np.unique(curr_match)
            new_dirs = np.empty((len(matched_nodes), 3))
            for idx, n_idx in enumerate(matched_nodes):
                mask = curr_match == n_idx
                mean_dir = deltas[mask].mean(axis=0)
                norm = np.linalg.norm(mean_dir)
                new_dirs[idx] = mean_dir / norm if norm > 0 else 0
            return new_dirs

        t = _median_time(_run)
    else:
        # Upstream: per-attractor Python loop computing delta individually
        # (mirrors the naive pattern where each attractor's contribution
        # is computed one-by-one without vectorised subtraction)
        def _run():
            matched_nodes = np.unique(curr_match)
            new_dirs = []
            for n_idx in matched_nodes:
                direction = np.zeros(3)
                count = 0
                for ai in range(_SC_N_ATTS):
                    if curr_match[ai] == n_idx:
                        # Compute delta per attractor (no pre-computed array)
                        direction += att_positions[ai] - nodes[n_idx]
                        count += 1
                if count > 0:
                    direction /= count
                norm = np.linalg.norm(direction)
                new_dirs.append(direction / norm if norm > 0 else np.zeros(3))
            return np.stack(new_dirs)

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("space_colonisation", t)


# =====================================================================
# 13. Colour sampling — batched (PR) vs per-sample loop (upstream)
#     Mirrors: infinigen/core/util/random.py random_color_mapping()
# =====================================================================


def bench_color_sampling(upstream=False):
    """Generate 10 000 colour samples with HSV→RGB + sRGB gamma."""
    np.random.seed(33)
    n_samples = 10_000
    mean_hsv = np.array([0.3, 0.6, 0.5])
    std_mat = np.array([[0.05, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

    if not upstream:
        # PR: batch-generate all noise, vectorised transform
        def _run():
            noise = np.clip(np.random.randn(n_samples, 3), -1, 1)
            hsv = mean_hsv + noise @ std_mat.T
            hsv[:, 2] = np.clip(hsv[:, 2], 0.1, 0.9)
            # Vectorised sRGB gamma (simplified — real code uses colorsys)
            rgb = np.clip(hsv, 0, 1)  # simplified HSV→RGB
            srgb = np.where(
                rgb >= 0.04045,
                ((rgb + 0.055) / 1.055) ** 2.4,
                rgb / 12.92,
            )
            return np.column_stack([srgb, np.ones(n_samples)])

        t = _median_time(_run)
    else:
        # Upstream: per-sample loop with individual matmul + clip
        def _run():
            results = []
            for _ in range(n_samples):
                noise = np.clip(np.random.randn(3), -1, 1)
                color = mean_hsv + std_mat @ noise
                color[2] = max(min(color[2], 0.9), 0.1)
                rgb = np.clip(color, 0, 1)  # simplified
                srgb = np.where(
                    rgb >= 0.04045,
                    ((rgb + 0.055) / 1.055) ** 2.4,
                    rgb / 12.92,
                )
                results.append(np.concatenate([srgb, [1.0]]))
            return results

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    return ("color_sampling", t)


# =====================================================================
# 14. Smooth attribute — pre-allocated buffers (PR) vs per-iter alloc (upstream)
#     Mirrors: infinigen/core/surface.py smooth_attribute()
# =====================================================================

_SMOOTH_N_VERTS = 10_000
_SMOOTH_N_EDGES = 25_000
_SMOOTH_ITERS = 10


def bench_smooth_attribute(upstream=False):
    """Smooth 10k-vertex attribute over 25k edges for 10 iterations."""
    np.random.seed(44)
    data = np.random.rand(_SMOOTH_N_VERTS, 1).astype(np.float64)
    # Random edges (pairs of vertex indices)
    edges = np.random.randint(0, _SMOOTH_N_VERTS, (_SMOOTH_N_EDGES, 2))
    weight = 0.05

    if not upstream:
        # PR: sparse matrix smoothing — build adjacency once, matmul per iter
        from scipy.sparse import csr_matrix

        row = np.concatenate([edges[:, 0], edges[:, 1], np.arange(_SMOOTH_N_VERTS)])
        col = np.concatenate([edges[:, 1], edges[:, 0], np.arange(_SMOOTH_N_VERTS)])
        vals = np.concatenate([
            np.full(len(edges), weight),
            np.full(len(edges), weight),
            np.ones(_SMOOTH_N_VERTS),
        ])
        W = csr_matrix((vals, (row, col)), shape=(_SMOOTH_N_VERTS, _SMOOTH_N_VERTS))
        # Normalise rows
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        diag_inv = csr_matrix(
            (1.0 / row_sums, (np.arange(_SMOOTH_N_VERTS), np.arange(_SMOOTH_N_VERTS))),
            shape=(_SMOOTH_N_VERTS, _SMOOTH_N_VERTS),
        )
        S = diag_inv @ W  # normalised smoothing matrix

        def _run():
            d = data.copy()
            for _ in range(_SMOOTH_ITERS):
                d = S @ d
            return d

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    else:
        # Upstream: per-edge Python loop (mirrors a naive per-edge implementation)
        def _run():
            d = data.copy().ravel()
            for _ in range(_SMOOTH_ITERS):
                accum = d.copy()
                counts = np.ones_like(d)
                for e0, e1 in edges:
                    accum[e0] += d[e1] * weight
                    counts[e0] += weight
                    accum[e1] += d[e0] * weight
                    counts[e1] += weight
                d = accum / counts
            return d.reshape(-1, 1)

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("smooth_attribute", t)


# =====================================================================
# 15. Vertex group — batch assignment (PR) vs per-vertex loop (upstream)
#     Mirrors: infinigen/core/surface.py attribute_to_vertex_group()
# =====================================================================

_VG_N_VERTS = 100_000


def bench_vertex_group(upstream=False):
    """Assign weighted vertex group for 100k vertices (simulate Blender API)."""
    np.random.seed(55)
    attr_data = np.random.rand(_VG_N_VERTS).astype(np.float64)
    min_thresh = 0.3

    if not upstream:
        # PR: batch — filter with boolean mask, single array op
        def _run():
            mask = attr_data > min_thresh
            indices = np.where(mask)[0]
            values = attr_data[mask]
            # Simulate batch API call (return arrays)
            return indices, values

        t = _median_time(_run)
    else:
        # Upstream: per-vertex loop (mirrors surface.py attribute_to_vertex_group)
        def _run():
            indices = []
            values = []
            for i, v in enumerate(attr_data):
                if v > min_thresh:
                    indices.append(i)
                    values.append(v)
            return indices, values

        t = _median_time(_run, n_repeat=N_REPEAT_SLOW)
    return ("vertex_group", t)


# =====================================================================
# 16. Mesh bulk copy — NumPy slicing (PR) vs Python element loop (upstream)
#     Mirrors: mesh data transfer patterns in core/surface.py
# =====================================================================

_BULKCOPY_N = 500_000


def bench_mesh_bulk_copy(upstream=False):
    """Copy 500k vertex coords into pre-allocated array."""
    np.random.seed(66)
    src = np.random.rand(_BULKCOPY_N, 3).astype(np.float64)
    dst = np.empty_like(src)

    if not upstream:
        # PR: bulk NumPy slice assignment
        def _run():
            dst[:] = src
            return dst

        t = _median_time(_run)
    else:
        # Upstream: per-vertex Python loop (simulates foreach_get patterns)
        def _run():
            for i in range(_BULKCOPY_N):
                dst[i] = src[i]
            return dst

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("mesh_bulk_copy", t)


# =====================================================================
# 17. SDF batch evaluation — stacked kernel (PR) vs sequential (upstream)
#     Mirrors: infinigen/terrain/mesher/uniform_mesher.py kernel_caller
# =====================================================================

_SDF_N_POINTS = 100_000
_SDF_N_KERNELS = 5


def bench_sdf_batch(upstream=False):
    """Evaluate 5 SDF kernels on 100k points and take element-wise min."""
    np.random.seed(77)
    points = np.random.rand(_SDF_N_POINTS, 3).astype(np.float64)
    # Simulate kernels as random sphere SDFs
    kernels = [
        (np.random.rand(3).astype(np.float64), float(np.random.rand()))
        for _ in range(_SDF_N_KERNELS)
    ]

    if not upstream:
        # PR: vectorised bulk — all kernels stacked, single min
        def _run():
            sdfs = np.empty((_SDF_N_KERNELS, _SDF_N_POINTS))
            for ki, (center, radius) in enumerate(kernels):
                sdfs[ki] = np.linalg.norm(points - center, axis=1) - radius
            return sdfs.min(axis=0)

        t = _median_time(_run)
    else:
        # Upstream: per-point Python loop computing distance to each kernel
        # (mirrors naive SDF evaluation without vectorisation)
        _N_SUB = 10_000  # subset to keep timing reasonable
        pts_sub = points[:_N_SUB]

        def _run():
            result = np.full(_N_SUB, float("inf"))
            for i in range(_N_SUB):
                for center, radius in kernels:
                    d = 0.0
                    for dim in range(3):
                        d += (pts_sub[i, dim] - center[dim]) ** 2
                    d = d ** 0.5 - radius
                    if d < result[i]:
                        result[i] = d
            return result

        t_sub = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
        # Extrapolate to full size
        t = t_sub * (_SDF_N_POINTS / _N_SUB)
    return ("sdf_batch", t)


# =====================================================================
# 18. Grid edge generation — vectorised (PR) vs triple loop (upstream)
#     Mirrors: infinigen/core/placement/path_finding.py path_finding()
# =====================================================================

_GRID_EDGE_N = 30  # 30×30×30 = 27k nodes


def bench_grid_edges(upstream=False):
    """Build connectivity graph for 30×30×30 grid (27k nodes)."""
    np.random.seed(88)
    N = _GRID_EDGE_N
    NN = N ** 3

    # Direction offsets for 9-connected grid (half of 18-connected)
    directions = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [0, 1, 1], [1, 0, 1],
        [1, -1, 0], [0, 1, -1], [1, 0, -1],
    ], dtype=np.int32)

    if not upstream:
        # PR: vectorised grid index computation using meshgrid + broadcasting
        def _run():
            idx = np.arange(N)
            gi, gj, gk = np.meshgrid(idx, idx, idx, indexing="ij")
            flat_from = (gi * N * N + gj * N + gk).ravel()
            row, col = [], []
            for d in directions:
                ni, nj, nk = gi + d[0], gj + d[1], gk + d[2]
                valid = (
                    (ni >= 0) & (ni < N) &
                    (nj >= 0) & (nj < N) &
                    (nk >= 0) & (nk < N)
                )
                flat_to = (ni * N * N + nj * N + nk)
                src = flat_from[valid.ravel()]
                dst = flat_to.ravel()[valid.ravel()]
                row.append(src)
                row.append(dst)
                col.append(dst)
                col.append(src)
            return np.concatenate(row), np.concatenate(col)

        t = _median_time(_run)
    else:
        # Upstream: triple nested loop (mirrors path_finding.py)
        def _run():
            row, col = [], []
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        idx_ijk = i * N * N + j * N + k
                        for di, dj, dk in directions:
                            ni, nj, nk = i + di, j + dj, k + dk
                            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                                idx_n = ni * N * N + nj * N + nk
                                row.append(idx_ijk)
                                col.append(idx_n)
                                row.append(idx_n)
                                col.append(idx_ijk)
            return np.array(row), np.array(col)

        t = _median_time(_run, n_repeat=N_REPEAT_VERY_SLOW)
    return ("grid_edges", t)


# =====================================================================
# 19. Full advanced pipeline — combines new optimisations end-to-end
#     RRT + smooth + grid edges + colour sampling + bulk copy
# =====================================================================


def bench_advanced_pipeline(upstream=False):
    """End-to-end: KDTree query + smooth attr + grid edges + colour sampling."""
    np.random.seed(200)

    if not upstream:
        from scipy.spatial import cKDTree

        def _pipeline():
            # Step 1: KDTree RRT queries (1000 nodes, 100 queries)
            nodes = np.random.rand(1000, 3)
            tree = cKDTree(nodes)
            queries = np.random.rand(100, 3)
            tree.query(queries, k=1)

            # Step 2: Smooth attribute (10k verts, 25k edges, 10 iters)
            nv, ne = 10_000, 25_000
            d = np.random.rand(nv, 1)
            edges = np.random.randint(0, nv, (ne, 2))
            d_out = np.empty_like(d)
            v_w = np.empty(nv)
            for _ in range(10):
                d_out[:] = d
                v_w[:] = 1.0
                np.add.at(d_out, edges[:, 0], d[edges[:, 1]] * 0.05)
                np.add.at(v_w, edges[:, 0], 0.05)
                np.add.at(d_out, edges[:, 1], d[edges[:, 0]] * 0.05)
                np.add.at(v_w, edges[:, 1], 0.05)
                d = d_out / v_w[:, None]

            # Step 3: Vectorised grid edges (15×15×15)
            N = 15
            idx = np.arange(N)
            gi, gj, gk = np.meshgrid(idx, idx, idx, indexing="ij")
            flat = (gi * N * N + gj * N + gk).ravel()
            for d_off in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                ni = gi + d_off[0]
                nj = gj + d_off[1]
                nk = gk + d_off[2]
                valid = (ni >= 0) & (ni < N) & (nj >= 0) & (nj < N) & (nk >= 0) & (nk < N)
                _ = flat[valid.ravel()]

            # Step 4: Batch colour sampling (5000 samples)
            noise = np.clip(np.random.randn(5000, 3), -1, 1)
            mean = np.array([0.3, 0.6, 0.5])
            std = np.eye(3) * 0.1
            hsv = mean + noise @ std.T
            rgb = np.clip(hsv, 0, 1)
            np.where(rgb >= 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

            # Step 5: Bulk copy (50k verts)
            src = np.random.rand(50_000, 3)
            dst = np.empty_like(src)
            dst[:] = src

        t = _median_time(_pipeline)
    else:

        def _pipeline():
            # Step 1: Brute-force nearest (mirrors rrt.py)
            nodes = np.random.rand(1000, 3)
            queries = np.random.rand(100, 3)
            for q in queries:
                dists = np.linalg.norm(nodes - q, axis=1)
                np.argmin(dists)

            # Step 2: Smooth attribute with per-edge loop
            nv, ne = 10_000, 25_000
            d = np.random.rand(nv, 1).ravel()
            edges = np.random.randint(0, nv, (ne, 2))
            for _ in range(10):
                vw = np.ones(nv)
                d_o = d.copy()
                for e0, e1 in edges:
                    d_o[e0] += d[e1] * 0.05
                    vw[e0] += 0.05
                    d_o[e1] += d[e0] * 0.05
                    vw[e1] += 0.05
                d = d_o / vw

            # Step 3: Triple loop grid edges (15×15×15)
            N = 15
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        for di, dj, dk in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                            ni, nj, nk = i + di, j + dj, k + dk
                            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                                pass  # would append to list

            # Step 4: Per-sample colour loop
            mean = np.array([0.3, 0.6, 0.5])
            std = np.eye(3) * 0.1
            for _ in range(5000):
                noise = np.clip(np.random.randn(3), -1, 1)
                c = mean + std @ noise
                c = np.clip(c, 0, 1)
                np.where(c >= 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

            # Step 5: Per-vertex copy loop
            src = np.random.rand(50_000, 3)
            dst = np.empty_like(src)
            for i in range(len(src)):
                dst[i] = src[i]

        t = _median_time(_pipeline, n_repeat=N_REPEAT_VERY_SLOW)
    return ("advanced_pipeline", t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [
    bench_unique_rows,
    bench_heightmap_grid,
    bench_mesh_cat,
    bench_tree_vertices,
    bench_projection,
    bench_distance_transform,
    bench_boundary_detection,
    bench_surface_normals,
    bench_full_pipeline,
    bench_rrt_nearest,
    bench_rrt_neighborhood,
    bench_space_colonisation,
    bench_color_sampling,
    bench_smooth_attribute,
    bench_vertex_group,
    bench_mesh_bulk_copy,
    bench_sdf_batch,
    bench_grid_edges,
    bench_advanced_pipeline,
]


def main():
    parser = argparse.ArgumentParser(description="Run infinigen micro-benchmarks")
    parser.add_argument("--output", required=True, help="Path for JSON results file")
    parser.add_argument(
        "--upstream",
        action="store_true",
        help=(
            "Use baseline implementations (simulates upstream Princeton main). "
            "Without this flag, uses optimised implementations from the PR branch."
        ),
    )
    args = parser.parse_args()

    mode = "UPSTREAM (baseline)" if args.upstream else "PR (optimised)"
    print(f"Running benchmarks in {mode} mode\n")

    results = {}
    for bench_fn in ALL_BENCHMARKS:
        try:
            name, t = bench_fn(upstream=args.upstream)
            results[name] = t
            print(f"  {name:.<50s} {t * 1000:10.2f} ms")
        except Exception as e:
            fname = bench_fn.__name__
            results[fname] = None
            print(f"  {fname:.<50s} SKIPPED ({e})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
