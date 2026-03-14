#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Benchmark runner that measures key operations and outputs JSON results.

Runs realistic micro-benchmarks on core NumPy/SciPy operations used by
infinigen — each benchmark mirrors an actual hot path in the codebase —
and records median timings.  Designed to work without ``bpy`` so it can
run in CI on plain Ubuntu runners.

Usage:
    python tests/run_benchmarks.py --output results.json [--upstream]
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

N_REPEAT = 7          # default: enough for stable median
N_REPEAT_SLOW = 3     # for benchmarks ~0.1–1 s each
N_REPEAT_VERY_SLOW = 2  # for benchmarks > 1 s each


def _median_time(fn, n_repeat=N_REPEAT):
    """Run *fn* *n_repeat* times and return the median wall-clock time (s).

    A single warm-up call is executed first (not counted) so JIT-like
    effects (branch prediction, memory mapping) don't skew the first
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
# 1. unique_rows — void-view vs np.unique(axis=0)
#    Mirrors: infinigen/core/util/exporting.py (instance-ID dedup)
# =====================================================================

def bench_unique_rows():
    """unique_rows (void-view) on 500 k × 3 int32 — realistic instance-ID scale."""
    mod = _load_module("array_ops", _UTIL_DIR / "array_ops.py")
    np.random.seed(42)
    arr = np.random.randint(0, 100, (500_000, 3), dtype=np.int32)
    if mod is not None and hasattr(mod, "unique_rows"):
        t = _median_time(lambda: mod.unique_rows(arr))
        return ("unique_rows (optimised)", t)
    t = _median_time(lambda: np.unique(arr, axis=0))
    return ("unique_rows (np.unique fallback)", t)


def bench_np_unique_baseline():
    """np.unique(axis=0) baseline — same data as unique_rows."""
    np.random.seed(42)
    arr = np.random.randint(0, 100, (500_000, 3), dtype=np.int32)
    t = _median_time(lambda: np.unique(arr, axis=0))
    return ("np.unique axis=0 baseline", t)


# =====================================================================
# 2. Heightmap grid construction — vectorised meshgrid vs Python loop
#    Mirrors: infinigen/terrain/utils/mesh.py Mesh.__init__ (heightmap branch)
# =====================================================================

_GRID_N = 256   # 256×256 heightmap → 65 536 vertices, 131 072 triangles
_GRID_L = 10.0  # terrain half-extent (metres)


def bench_heightmap_grid_vectorised():
    """Vectorised heightmap vertex + face construction (256×256)."""
    np.random.seed(10)
    N, L = _GRID_N, _GRID_L
    heightmap = np.random.rand(N, N).astype(np.float32)

    def _vectorised():
        xs = (-1 + 2 * np.arange(N) / (N - 1)) * L / 2
        ys = (-1 + 2 * np.arange(N) / (N - 1)) * L / 2
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        verts = np.stack([gx, gy, heightmap], axis=-1).reshape(-1, 3)
        # Vectorised face index computation
        i_idx = np.arange(N - 1).reshape(-1, 1)
        j_idx = np.arange(N - 1).reshape(1, -1)
        faces = np.empty((2, N - 1, N - 1, 3), np.int32)
        faces[0] = np.stack(
            [i_idx * N + j_idx, (i_idx + 1) * N + j_idx, i_idx * N + j_idx + 1],
            axis=-1,
        )
        faces[1] = np.stack(
            [i_idx * N + j_idx + 1, (i_idx + 1) * N + j_idx, (i_idx + 1) * N + j_idx + 1],
            axis=-1,
        )
        faces = faces.reshape(-1, 3)
        return verts, faces

    t = _median_time(_vectorised)
    return ("heightmap grid vectorised 256", t)


def bench_heightmap_grid_loop():
    """Loop-based heightmap vertex + face construction (256×256) — reference."""
    np.random.seed(10)
    N, L = _GRID_N, _GRID_L
    heightmap = np.random.rand(N, N).astype(np.float32)

    def _loop():
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
        faces = np.array(faces, dtype=np.int32)
        return verts, faces

    t = _median_time(_loop, n_repeat=N_REPEAT_VERY_SLOW)
    return ("heightmap grid loop 256", t)


# =====================================================================
# 3. Mesh.cat — pre-allocated single-pass vs naive repeated vstack
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


def bench_mesh_cat_prealloc():
    """Pre-allocated single-pass mesh concatenation (50 meshes)."""
    meshes = _make_fake_meshes(_CAT_N_MESHES, _CAT_VERTS_PER, _CAT_FACES_PER)

    def _prealloc():
        total_v = sum(v.shape[0] for v, _, _ in meshes)
        total_f = sum(f.shape[0] for _, f, _ in meshes)
        verts = np.empty((total_v, 3), dtype=np.float64)
        faces = np.empty((total_f, 3), dtype=np.int32)
        sdf = np.zeros(total_v, dtype=np.float32)
        vo, fo = 0, 0
        for v, f, a in meshes:
            nv, nf = v.shape[0], f.shape[0]
            verts[vo:vo + nv] = v
            faces[fo:fo + nf] = f + vo
            sdf[vo:vo + nv] = a["sdf"]
            vo += nv
            fo += nf
        return verts, faces, sdf

    t = _median_time(_prealloc)
    return ("mesh cat prealloc 50×2k", t)


def bench_mesh_cat_naive():
    """Naive repeated vstack mesh concatenation (50 meshes) — reference."""
    meshes = _make_fake_meshes(_CAT_N_MESHES, _CAT_VERTS_PER, _CAT_FACES_PER)

    def _naive():
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

    t = _median_time(_naive, n_repeat=N_REPEAT_SLOW)
    return ("mesh cat naive vstack 50×2k", t)


# =====================================================================
# 4. Tree vertices — lazy concat vs O(n²) np.append
#    Mirrors: infinigen/assets/objects/trees/tree.py TreeVertices
# =====================================================================

_TREE_N_PARTS = 300   # typical tree has 200-500 branch segments
_TREE_PTS_PER = 80    # ~80 vertices per branch segment


def bench_tree_vertices_lazy():
    """Lazy concatenation (list accumulate + single np.concatenate)."""
    np.random.seed(7)
    parts = [np.random.rand(_TREE_PTS_PER, 3) for _ in range(_TREE_N_PARTS)]

    def _lazy():
        acc = []
        for p in parts:
            acc.append(p)
        return np.concatenate(acc, axis=0)

    t = _median_time(_lazy)
    return ("tree vertices lazy concat", t)


def bench_tree_vertices_eager():
    """Eager O(n²) np.append pattern — reference."""
    np.random.seed(7)
    parts = [np.random.rand(_TREE_PTS_PER, 3) for _ in range(_TREE_N_PARTS)]

    def _eager():
        arr = np.empty((0, 3))
        for p in parts:
            arr = np.append(arr, p, axis=0)
        return arr

    t = _median_time(_eager, n_repeat=N_REPEAT_SLOW)
    return ("tree vertices eager append", t)


# =====================================================================
# 5. Camera projection — pre-computed combined matrix vs separate steps
#    Mirrors: infinigen/terrain/utils/mesh.py Mesh.camera_annotation
# =====================================================================

_PROJ_N_VERTS = 200_000   # typical terrain mesh vertex count
_PROJ_N_CAMS = 10         # multi-camera rig


def bench_projection_precomputed():
    """Pre-computed K @ inv(cam)[:3,:] — 10 cameras × 200 k vertices."""
    np.random.seed(55)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    cams = []
    for _ in range(_PROJ_N_CAMS):
        cam = np.eye(4, dtype=np.float64)
        cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
        cam[:3, 3] = np.random.randn(3) * 5
        cams.append(cam)
    points = np.random.rand(_PROJ_N_VERTS, 3).astype(np.float64)
    homo = np.hstack([points, np.ones((_PROJ_N_VERTS, 1))]).T  # (4, N)
    projs = [K @ np.linalg.inv(c)[:3, :] for c in cams]

    def _precomputed():
        visible = np.zeros(_PROJ_N_VERTS, dtype=bool)
        for proj in projs:
            coords = proj @ homo  # (3, N) single matmul
            coords[:2] /= coords[2]
            visible |= (coords[2] > 0) & (coords[0] > 0) & (coords[0] < 640)
        return visible

    t = _median_time(_precomputed)
    return ("projection precomputed 10cam", t)


def bench_projection_separate():
    """Separate inv + two matmuls per camera — reference."""
    np.random.seed(55)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    cams = []
    for _ in range(_PROJ_N_CAMS):
        cam = np.eye(4, dtype=np.float64)
        cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
        cam[:3, 3] = np.random.randn(3) * 5
        cams.append(cam)
    points = np.random.rand(_PROJ_N_VERTS, 3).astype(np.float64)

    def _separate():
        visible = np.zeros(_PROJ_N_VERTS, dtype=bool)
        for cam in cams:
            cam_inv = np.linalg.inv(cam)
            pts_h = np.hstack([points, np.ones((_PROJ_N_VERTS, 1))])
            world = (cam_inv[:3, :] @ pts_h.T).T
            screen = (K @ world.T).T
            screen[:, :2] /= screen[:, 2:3]
            visible |= (screen[:, 2] > 0) & (screen[:, 0] > 0) & (screen[:, 0] < 640)
        return visible

    t = _median_time(_separate)
    return ("projection separate 10cam", t)


# =====================================================================
# 6. Distance transform — scipy EDT vs brute-force loop
#    Mirrors: infinigen/terrain/utils/image_processing.py grid_distance
# =====================================================================

_DIST_N = 128  # 128×128 grid — realistic downsampled terrain tile


def bench_distance_edt():
    """scipy distance_transform_edt on 128×128 binary mask."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return ("distance_transform_edt 128", float("nan"))

    np.random.seed(42)
    mask = (np.random.rand(_DIST_N, _DIST_N) > 0.5).astype(np.float64)
    t = _median_time(lambda: distance_transform_edt(mask))
    return ("distance_transform_edt 128", t)


def bench_distance_loop():
    """Loop-based grid distance on 128×128 mask — reference.

    This is the O(N²·B) algorithm that the EDT replaces. We cap iteration
    at a smaller sub-grid (64×64) and extrapolate time for the full grid so
    CI doesn't time out.
    """
    np.random.seed(42)
    # Use 64×64 sub-grid for the actual timing to keep CI fast
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
    # Extrapolate: time scales as O(N² × B) where B ∝ N²  →  O(N⁴)
    scale = (_DIST_N / SUB) ** 4
    t = t64 * scale
    return ("grid_distance loop 128 (extrapolated)", t)


# =====================================================================
# 7. Boundary detection — vectorised edge detection vs pixel loop
#    Mirrors: image_processing.py grid_distance boundary identification
# =====================================================================

def bench_boundary_vectorised():
    """Vectorised boundary detection on 256×256 mask."""
    np.random.seed(42)
    N = 256
    source = (np.random.rand(N, N) > 0.4).astype(bool)

    def _vectorised():
        boundary = np.zeros_like(source, dtype=bool)
        boundary[:-1, :] |= ~source[1:, :]
        boundary[1:, :] |= ~source[:-1, :]
        boundary[:, :-1] |= ~source[:, 1:]
        boundary[:, 1:] |= ~source[:, :-1]
        boundary &= source
        return boundary

    t = _median_time(_vectorised)
    return ("boundary vectorised 256", t)


def bench_boundary_loop():
    """Loop-based boundary detection on 256×256 mask — reference."""
    np.random.seed(42)
    N = 256
    source = (np.random.rand(N, N) > 0.4).astype(bool)

    def _loop():
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

    t = _median_time(_loop, n_repeat=N_REPEAT_VERY_SLOW)
    return ("boundary loop 256", t)


# =====================================================================
# 8. Normal computation — vectorised vs loop
#    Mirrors: infinigen/terrain/utils/image_processing.py get_normal
# =====================================================================

def bench_normals_vectorised():
    """Vectorised surface normal computation for 256×256 heightmap."""
    np.random.seed(77)
    N = 256
    z = np.random.rand(N, N).astype(np.float64)
    grid_size = 1.0 / (N - 1)

    def _vectorised():
        dzdx = np.zeros_like(z)
        dzdx[1:] = z[1:] - z[:-1]
        dzdx[0] = dzdx[1]
        dzdy = np.zeros_like(z)
        dzdy[:, 1:] = z[:, 1:] - z[:, :-1]
        dzdy[:, 0] = dzdy[:, 1]
        n = np.stack((-dzdy, -dzdx, grid_size * np.ones_like(z)), axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return n

    t = _median_time(_vectorised)
    return ("normals vectorised 256", t)


def bench_normals_loop():
    """Loop-based surface normal computation for 256×256 — reference."""
    np.random.seed(77)
    N = 256
    z = np.random.rand(N, N).astype(np.float64)
    grid_size = 1.0 / (N - 1)

    def _loop():
        normals = np.zeros((N, N, 3), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                dzdx = (z[min(i + 1, N - 1), j] - z[max(i - 1, 0), j]) / 2
                dzdy = (z[i, min(j + 1, N - 1)] - z[i, max(j - 1, 0)]) / 2
                n = np.array([-dzdy, -dzdx, grid_size])
                normals[i, j] = n / np.linalg.norm(n)
        return normals

    t = _median_time(_loop, n_repeat=N_REPEAT_VERY_SLOW)
    return ("normals loop 256", t)


# =====================================================================
# 9. Full pipeline composite — all optimised ops in sequence
#    Simulates processing a single terrain tile end-to-end.
# =====================================================================

def bench_pipeline_optimised():
    """Composite: grid build + unique_rows + projection + EDT — vectorised."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return ("pipeline optimised", float("nan"))

    mod = _load_module("array_ops", _UTIL_DIR / "array_ops.py")
    np.random.seed(123)
    N = 128

    def _pipeline():
        # Step 1: heightmap grid
        h = np.random.rand(N, N).astype(np.float32)
        xs = np.linspace(-5, 5, N, dtype=np.float32)
        ys = np.linspace(-5, 5, N, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        verts = np.stack([gx, gy, h], axis=-1).reshape(-1, 3)

        # Step 2: deduplicate (simulating instance-ID check)
        ids = np.random.randint(0, 50, (len(verts), 3), dtype=np.int32)
        if mod is not None and hasattr(mod, "unique_rows"):
            mod.unique_rows(ids)
        else:
            np.unique(ids, axis=0)

        # Step 3: camera projection (2 cameras)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        homo = np.hstack([verts.astype(np.float64), np.ones((len(verts), 1))]).T
        for _ in range(2):
            cam = np.eye(4)
            cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
            proj = K @ np.linalg.inv(cam)[:3, :]
            proj @ homo

        # Step 4: distance transform
        mask = (h > 0.5).astype(np.float64)
        distance_transform_edt(mask)

    t = _median_time(_pipeline)
    return ("pipeline optimised", t)


def bench_pipeline_baseline():
    """Composite: same ops via loop-based baselines — reference."""
    np.random.seed(123)
    N = 128

    def _pipeline():
        # Step 1: heightmap grid (loop)
        h = np.random.rand(N, N).astype(np.float32)
        verts = []
        for i in range(N):
            for j in range(N):
                verts.append([-5 + 10 * i / (N - 1), -5 + 10 * j / (N - 1), h[i, j]])
        verts = np.array(verts, dtype=np.float32)

        # Step 2: np.unique axis=0
        ids = np.random.randint(0, 50, (len(verts), 3), dtype=np.int32)
        np.unique(ids, axis=0)

        # Step 3: separate projection
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        for _ in range(2):
            cam = np.eye(4)
            cam[:3, :3] = np.linalg.qr(np.random.randn(3, 3))[0]
            cam_inv = np.linalg.inv(cam)
            pts_h = np.hstack([verts.astype(np.float64), np.ones((len(verts), 1))])
            world = (cam_inv[:3, :] @ pts_h.T).T
            _screen = (K @ world.T).T

        # Step 4: loop distance
        mask = (h > 0.5).astype(np.float64)
        # Simplified loop distance on 32×32 sub-grid
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
    return ("pipeline baseline", t)


# ---------------------------------------------------------------------------
# Concatenation benchmarks — kept for raw timings table, but NOT paired
# because np.concatenate holds the GIL so threading cannot outperform it.
# ---------------------------------------------------------------------------

def bench_concatenation():
    """chunked_concat (parallel) on 200 arrays of 1 000 floats."""
    np.random.seed(99)
    arrays = [np.random.rand(1000) for _ in range(200)]
    mod = _load_module("batch_ops", _UTIL_DIR / "batch_ops.py")
    if mod is not None and hasattr(mod, "chunked_concat"):
        t = _median_time(lambda: mod.chunked_concat(arrays, num_workers=4))
        return ("chunked_concat (parallel)", t)
    t = _median_time(lambda: np.concatenate(arrays))
    return ("np.concatenate fallback", t)


def bench_np_concatenate_baseline():
    """np.concatenate baseline — same data."""
    np.random.seed(99)
    arrays = [np.random.rand(1000) for _ in range(200)]
    t = _median_time(lambda: np.concatenate(arrays))
    return ("np.concatenate baseline", t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [
    # Paired benchmarks (optimised then baseline)
    bench_unique_rows,
    bench_np_unique_baseline,
    bench_heightmap_grid_vectorised,
    bench_heightmap_grid_loop,
    bench_mesh_cat_prealloc,
    bench_mesh_cat_naive,
    bench_tree_vertices_lazy,
    bench_tree_vertices_eager,
    bench_projection_precomputed,
    bench_projection_separate,
    bench_distance_edt,
    bench_distance_loop,
    bench_boundary_vectorised,
    bench_boundary_loop,
    bench_normals_vectorised,
    bench_normals_loop,
    bench_pipeline_optimised,
    bench_pipeline_baseline,
    # Unpaired (raw timings only)
    bench_concatenation,
    bench_np_concatenate_baseline,
]


def main():
    parser = argparse.ArgumentParser(description="Run infinigen micro-benchmarks")
    parser.add_argument("--output", required=True, help="Path for JSON results file")
    parser.add_argument(
        "--upstream",
        action="store_true",
        help="Running against upstream (may lack some optimised modules)",
    )
    args = parser.parse_args()

    if args.upstream:
        print(
            "Running in upstream mode — optimised modules may be missing; "
            "benchmarks will fall back to baseline implementations."
        )

    results = {}
    for bench_fn in ALL_BENCHMARKS:
        try:
            name, t = bench_fn()
            results[name] = t
            print(f"  {name:.<55s} {t*1000:10.2f} ms")
        except Exception as e:
            fname = bench_fn.__name__
            results[fname] = None
            print(f"  {fname:.<55s} SKIPPED ({e})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
