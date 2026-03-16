#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory.

"""
Benchmark placement visibility filtering on real Blender scenes.

Compares:
- legacy per-placeholder visibility filtering
- current batched visibility filtering

Run inside Blender, for example:

  blender -b your_scene.blend --python scripts/benchmark_filter_populate_targets.py -- \
    --placeholder-collection "placeholders:MyFactory(0)" \
    --camera-collection "Cameras" \
    --dist-cull 200 --vis-cull 0 --repeat 3 \
    --output /tmp/filter_populate_targets_bench.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bpy
import numpy as np

from infinigen.core.placement import split_in_view
from infinigen.core.placement.placement import (
    filter_populate_targets,
    get_placeholder_points,
    parse_asset_name,
)


def filter_populate_targets_legacy(
    placeholders: list[bpy.types.Object],
    cameras: list[bpy.types.Object],
    dist_cull: float,
    vis_cull: float,
):
    results = []
    for p in placeholders:
        classname, *_ = parse_asset_name(p.name)
        if classname is None:
            raise ValueError(f"Could not parse p.name={p.name}")

        mask, min_dists, min_vis_dists = split_in_view.compute_inview_distances(
            get_placeholder_points(p),
            cameras,
            dist_max=dist_cull,
            vis_margin=vis_cull,
            verbose=False,
        )

        dist = float(min_dists.min())
        vis_dist = float(min_vis_dists.min())

        if mask.any():
            results.append((p, dist, vis_dist))

    return results


def _median_time(fn, repeat: int):
    # Warm-up for fairer timing.
    fn()
    timings = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings)), out


def _get_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        raise ValueError(f"Collection not found: {name}")
    return col


def _get_cameras(camera_collection_name: str):
    cam_col = _get_collection(camera_collection_name)
    cams = [o for o in cam_col.objects if o.type == "CAMERA"]
    if not cams:
        raise ValueError(f"No cameras found in collection: {camera_collection_name}")
    return cams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placeholder-collection", required=True)
    parser.add_argument("--camera-collection", required=True)
    parser.add_argument("--dist-cull", type=float, default=200.0)
    parser.add_argument("--vis-cull", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/tmp/filter_populate_targets_bench.json"))
    args = parser.parse_args()

    placeholders_col = _get_collection(args.placeholder_collection)
    placeholders = [o for o in placeholders_col.objects if o.parent is None]
    cameras = _get_cameras(args.camera_collection)

    if not placeholders:
        raise ValueError(f"No placeholders found in collection: {args.placeholder_collection}")

    legacy_t, legacy_out = _median_time(
        lambda: filter_populate_targets_legacy(
            placeholders,
            cameras,
            dist_cull=args.dist_cull,
            vis_cull=args.vis_cull,
        ),
        repeat=max(1, args.repeat),
    )

    batched_t, batched_out = _median_time(
        lambda: filter_populate_targets(
            placeholders,
            cameras,
            dist_cull=args.dist_cull,
            vis_cull=args.vis_cull,
            verbose=False,
        ),
        repeat=max(1, args.repeat),
    )

    report = {
        "placeholder_collection": args.placeholder_collection,
        "camera_collection": args.camera_collection,
        "num_placeholders": len(placeholders),
        "num_cameras": len(cameras),
        "repeat": int(max(1, args.repeat)),
        "dist_cull": float(args.dist_cull),
        "vis_cull": float(args.vis_cull),
        "legacy_seconds_median": legacy_t,
        "batched_seconds_median": batched_t,
        "speedup_batched_over_legacy": (legacy_t / batched_t) if batched_t > 0 else float("inf"),
        "legacy_targets": len(legacy_out),
        "batched_targets": len(batched_out),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
