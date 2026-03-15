#!/usr/bin/env python3

import argparse
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from infinigen.assets.utils.isosurface import marching_cubes_vertices_faces


def laplacian_like_volume(n: int = 64) -> np.ndarray:
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, n),
        np.linspace(-1, 1, n),
        np.linspace(0, 1.5, n),
        indexing="ij",
    )
    field = 1.0 - (x**2 + y**2 + (z - 0.4) ** 2 / 0.3)
    field += 0.08 * np.sin(12 * x) * np.cos(9 * y)
    return np.pad(field.astype(np.float32), 1)


def cloud_like_volume(n: int = 96) -> np.ndarray:
    grid = np.indices((n, n, n), dtype=np.float32)
    center = np.array([(n - 1) / 2] * 3, dtype=np.float32).reshape(3, 1, 1, 1)
    radius = n / 3.2
    points = np.sqrt(((grid - center) ** 2).sum(axis=0)) < radius
    dists = distance_transform_edt(points)
    dists /= max(float(dists.max()), 1e-6)
    dists[dists < 0.01] = 0
    return dists.astype(np.float32)


def benchmark_case(name: str, volume: np.ndarray, level: float, repeats: int):
    rows = []
    for backend in ("skimage", "mcubes"):
        try:
            marching_cubes_vertices_faces(volume, level, backend=backend)
        except ImportError as exc:
            rows.append((backend, None, None, None, str(exc)))
            continue
        t0 = time.perf_counter()
        verts = faces = None
        for _ in range(repeats):
            verts, faces = marching_cubes_vertices_faces(volume, level, backend=backend)
        dt = (time.perf_counter() - t0) / repeats
        rows.append((backend, dt, verts.shape, faces.shape, None))

    print(f"[{name}]")
    for backend, dt, vshape, fshape, error in rows:
        if error is not None:
            print(f"  {backend:<8s} unavailable: {error}")
            continue
        print(f"  {backend:<8s} avg={dt:.6f}s verts={vshape} faces={fshape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    benchmark_case("laplacian_like", laplacian_like_volume(), 0.5, args.repeats)
    benchmark_case("cloud_like", cloud_like_volume(), 0.08, args.repeats)


if __name__ == "__main__":
    main()