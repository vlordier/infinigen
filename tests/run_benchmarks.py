#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Benchmark runner that measures key operations and outputs JSON results.

Runs a suite of micro-benchmarks on core NumPy/SciPy operations used by
infinigen and records median timings.  Designed to work without ``bpy`` so
it can run in CI on plain Ubuntu runners.

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
# Individual benchmarks – each returns (name, median_seconds)
# ---------------------------------------------------------------------------

N_REPEAT = 5


def _median_time(fn, n_repeat=N_REPEAT):
    """Run fn n_repeat times and return the median wall-clock time in seconds."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_unique_rows():
    """unique_rows (void-view) on 500k×3 int32 array."""
    mod = _load_module("array_ops", _UTIL_DIR / "array_ops.py")
    np.random.seed(42)
    arr = np.random.randint(0, 100, (500_000, 3), dtype=np.int32)
    if mod is not None and hasattr(mod, "unique_rows"):
        t = _median_time(lambda: mod.unique_rows(arr))
        return ("unique_rows (optimised)", t)
    # Fallback for upstream that lacks unique_rows
    t = _median_time(lambda: np.unique(arr, axis=0))
    return ("unique_rows (np.unique fallback)", t)


def bench_np_unique_baseline():
    """np.unique(..., axis=0) baseline on same data as unique_rows."""
    np.random.seed(42)
    arr = np.random.randint(0, 100, (500_000, 3), dtype=np.int32)
    t = _median_time(lambda: np.unique(arr, axis=0))
    return ("np.unique axis=0 baseline", t)


def bench_concatenation():
    """Concatenation of 200 arrays of 1000 floats."""
    arrays = [np.random.rand(1000) for _ in range(200)]
    mod = _load_module("batch_ops", _UTIL_DIR / "batch_ops.py")
    if mod is not None and hasattr(mod, "chunked_concat"):
        t = _median_time(lambda: mod.chunked_concat(arrays, num_workers=4))
        return ("chunked_concat (parallel)", t)
    t = _median_time(lambda: np.concatenate(arrays))
    return ("np.concatenate fallback", t)


def bench_np_concatenate_baseline():
    """np.concatenate baseline for same data."""
    arrays = [np.random.rand(1000) for _ in range(200)]
    t = _median_time(lambda: np.concatenate(arrays))
    return ("np.concatenate baseline", t)


def bench_meshgrid_vectorised():
    """Vectorised meshgrid construction for a 256×256 heightmap."""
    N = 256
    h = np.random.rand(N, N).astype(np.float32)

    def _vectorised():
        xs = np.linspace(0, 1, N, dtype=np.float32)
        ys = np.linspace(0, 1, N, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        verts = np.stack([gx, h, gy], axis=-1).reshape(-1, 3)
        return verts

    t = _median_time(_vectorised)
    return ("meshgrid vectorised 256x256", t)


def bench_meshgrid_loop():
    """Loop-based meshgrid construction (reference) for 256×256."""
    N = 256
    h = np.random.rand(N, N).astype(np.float32)

    def _loop():
        verts = []
        for i in range(N):
            for j in range(N):
                verts.append([i / (N - 1), h[i, j], j / (N - 1)])
        return np.array(verts, dtype=np.float32)

    t = _median_time(_loop, n_repeat=2)  # loops are slow, fewer repeats
    return ("meshgrid loop 256x256", t)


def bench_tree_vertices_lazy():
    """Lazy concatenation pattern (append list + final concat)."""
    np.random.seed(7)
    parts = [np.random.rand(100, 3) for _ in range(500)]

    def _lazy():
        acc = []
        for p in parts:
            acc.append(p)
        return np.concatenate(acc, axis=0)

    t = _median_time(_lazy)
    return ("tree vertices lazy concat", t)


def bench_tree_vertices_eager():
    """Eager O(n²) np.append pattern (reference)."""
    np.random.seed(7)
    parts = [np.random.rand(100, 3) for _ in range(200)]  # fewer for speed

    def _eager():
        arr = np.empty((0, 3))
        for p in parts:
            arr = np.append(arr, p, axis=0)
        return arr

    t = _median_time(_eager, n_repeat=2)
    return ("tree vertices eager append", t)


def bench_projection_precomputed():
    """Pre-computed combined projection matrix K @ inv(cam)[:3,:]."""
    K = np.random.rand(3, 3).astype(np.float64)
    cam = np.eye(4, dtype=np.float64)
    cam[:3, :3] = np.random.rand(3, 3)
    cam[:3, 3] = np.random.rand(3)
    points = np.random.rand(100_000, 3).astype(np.float64)

    cam_inv = np.linalg.inv(cam)
    combined = K @ cam_inv[:3, :]

    def _precomputed():
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        return (combined @ pts_h.T).T

    t = _median_time(_precomputed)
    return ("projection precomputed", t)


def bench_projection_separate():
    """Separate inv + multiply (reference)."""
    K = np.random.rand(3, 3).astype(np.float64)
    cam = np.eye(4, dtype=np.float64)
    cam[:3, :3] = np.random.rand(3, 3)
    cam[:3, 3] = np.random.rand(3)
    points = np.random.rand(100_000, 3).astype(np.float64)

    def _separate():
        cam_inv = np.linalg.inv(cam)
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        world = (cam_inv[:3, :] @ pts_h.T).T
        return (K @ world.T).T

    t = _median_time(_separate)
    return ("projection separate steps", t)


def bench_distance_transform():
    """scipy distance_transform_edt on 512×512 binary mask."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return ("distance_transform_edt", float("nan"))

    np.random.seed(42)
    mask = (np.random.rand(512, 512) > 0.5).astype(np.float64)
    t = _median_time(lambda: distance_transform_edt(mask))
    return ("distance_transform_edt 512x512", t)


def bench_distance_loop():
    """Loop-based grid distance (reference) on small 64×64 mask."""
    np.random.seed(42)
    N = 64
    mask = (np.random.rand(N, N) > 0.5).astype(np.float64)
    seeds = np.argwhere(mask > 0)

    def _loop():
        dist = np.full((N, N), float("inf"))
        for si, sj in seeds:
            for i in range(N):
                for j in range(N):
                    d = (i - si) ** 2 + (j - sj) ** 2
                    if d < dist[i, j]:
                        dist[i, j] = d
        return np.sqrt(dist)

    t = _median_time(_loop, n_repeat=1)
    return ("grid_distance loop 64x64", t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [
    bench_unique_rows,
    bench_np_unique_baseline,
    bench_concatenation,
    bench_np_concatenate_baseline,
    bench_meshgrid_vectorised,
    bench_meshgrid_loop,
    bench_tree_vertices_lazy,
    bench_tree_vertices_eager,
    bench_projection_precomputed,
    bench_projection_separate,
    bench_distance_transform,
    bench_distance_loop,
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

    results = {}
    for bench_fn in ALL_BENCHMARKS:
        try:
            name, t = bench_fn()
            results[name] = t
            print(f"  {name:.<50s} {t*1000:8.2f} ms")
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
