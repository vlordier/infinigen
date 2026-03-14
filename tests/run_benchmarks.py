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
