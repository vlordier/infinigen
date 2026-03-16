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
import logging
import re
import sys
import time
from pathlib import Path

import bpy
import numpy as np

logger = logging.getLogger(__name__)

_IMPORT_ERROR = None
try:
    from infinigen.core.placement import split_in_view as _split_in_view
    from infinigen.core.placement.placement import (
        filter_populate_targets as _filter_populate_targets_impl,
    )
    from infinigen.core.placement.placement import (
        get_placeholder_points as _get_placeholder_points_impl,
    )
    from infinigen.core.placement.placement import (
        parse_asset_name as _parse_asset_name_impl,
    )
except Exception as e:  # pragma: no cover - fallback path depends on local environment
    _IMPORT_ERROR = str(e)
    _split_in_view = None
    _filter_populate_targets_impl = None
    _get_placeholder_points_impl = None
    _parse_asset_name_impl = None


def _apply_world_matrix(obj: bpy.types.Object, points: np.ndarray) -> np.ndarray:
    mat = np.array(obj.matrix_world, dtype=np.float64)
    pts = np.c_[points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)]
    return (pts @ mat.T)[:, :3]


def _parse_asset_name_local(name):
    match = re.fullmatch(r"(.*)\((\d+)\)\..*_(.*)\((\d+)\)", name)
    if not match:
        return None, None, None, None
    return list(match.groups())


def _get_placeholder_points_local(obj: bpy.types.Object) -> np.ndarray:
    if obj.type == "MESH":
        verts = np.zeros((len(obj.data.vertices), 3), dtype=np.float64)
        obj.data.vertices.foreach_get("co", verts.reshape(-1))
        return _apply_world_matrix(obj, verts)
    if obj.type == "EMPTY" and obj.empty_display_type == "CUBE":
        extent = obj.empty_display_size * np.array([-1.0, 1.0], dtype=np.float64)
        verts = np.stack(np.meshgrid(extent, extent, extent), axis=-1).reshape(-1, 3)
        return _apply_world_matrix(obj, verts)
    return np.array([obj.matrix_world.translation], dtype=np.float64).reshape(1, 3)


def _compute_inview_distances_local(
    points: np.ndarray,
    cameras: list[bpy.types.Object],
    dist_max,
    vis_margin,
    frame_start: int,
    frame_end: int,
):
    mask = np.zeros(len(points), dtype=bool)
    min_dists = np.full(len(points), 1e7, dtype=np.float64)
    min_vis_dists = np.full(len(points), 1e7, dtype=np.float64)

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        for cam in cameras:
            cam_loc = np.array(cam.matrix_world.translation, dtype=np.float64)
            dists = np.linalg.norm(points - cam_loc[None, :], axis=-1)
            vis_dists = np.zeros_like(dists)

            frame_cam_mask = np.ones(len(points), dtype=bool)
            if dist_max is not None:
                frame_cam_mask &= dists < dist_max
            if vis_margin is not None:
                frame_cam_mask &= vis_dists < vis_margin

            if frame_cam_mask.any():
                np.minimum(dists, min_dists, where=frame_cam_mask, out=min_dists)
                np.minimum(vis_dists, min_vis_dists, where=frame_cam_mask, out=min_vis_dists)
            mask |= frame_cam_mask

    return mask, min_dists, min_vis_dists


def _compute_inview_distances(points, cameras, dist_cull, vis_cull, frame_start, frame_end):
    if _split_in_view is not None:
        return _split_in_view.compute_inview_distances(
            points,
            cameras,
            dist_max=dist_cull,
            vis_margin=vis_cull,
            frame_start=frame_start,
            frame_end=frame_end,
            verbose=False,
        )
    return _compute_inview_distances_local(points, cameras, dist_cull, vis_cull, frame_start, frame_end)


def _get_placeholder_points(obj: bpy.types.Object) -> np.ndarray:
    if _get_placeholder_points_impl is not None:
        return _get_placeholder_points_impl(obj)
    return _get_placeholder_points_local(obj)


def _parse_asset_name(name: str):
    if _parse_asset_name_impl is not None:
        return _parse_asset_name_impl(name)
    return _parse_asset_name_local(name)


def _filter_populate_targets_batched(
    placeholders: list[bpy.types.Object],
    cameras: list[bpy.types.Object],
    dist_cull: float,
    vis_cull: float,
    frame_start: int,
    frame_end: int,
):
    if _filter_populate_targets_impl is not None:
        return _filter_populate_targets_impl(
            placeholders,
            cameras,
            dist_cull=dist_cull,
            vis_cull=vis_cull,
            verbose=False,
        )

    checked_placeholders = []
    point_sets = []
    point_counts = []
    for p in placeholders:
        classname, *_ = _parse_asset_name_local(p.name)
        if classname is None:
            raise ValueError(f"Could not parse p.name={p.name}")
        pts = _get_placeholder_points_local(p)
        checked_placeholders.append(p)
        point_sets.append(pts)
        point_counts.append(len(pts))

    results = []
    max_points_per_batch = 200_000
    n = len(checked_placeholders)
    i = 0
    while i < n:
        batch_points = 0
        j = i
        while j < n and (batch_points + point_counts[j] <= max_points_per_batch or j == i):
            batch_points += point_counts[j]
            j += 1

        batch_concat = np.concatenate(point_sets[i:j], axis=0)
        mask, min_dists, min_vis_dists = _compute_inview_distances_local(
            batch_concat,
            cameras,
            dist_cull,
            vis_cull,
            frame_start,
            frame_end,
        )

        start = 0
        for k in range(i, j):
            cnt = point_counts[k]
            end = start + cnt
            p_mask = mask[start:end]
            p_min_dists = min_dists[start:end]
            p_min_vis_dists = min_vis_dists[start:end]
            p = checked_placeholders[k]

            if p_mask.any():
                results.append((p, float(p_min_dists.min()), float(p_min_vis_dists.min())))
            start = end

        i = j

    return results


def filter_populate_targets_legacy(
    placeholders: list[bpy.types.Object],
    cameras: list[bpy.types.Object],
    dist_cull: float,
    vis_cull: float,
    frame_start: int,
    frame_end: int,
):
    results = []
    for p in placeholders:
        classname, *_ = _parse_asset_name(p.name)
        if classname is None:
            raise ValueError(f"Could not parse p.name={p.name}")

        mask, min_dists, min_vis_dists = _compute_inview_distances(
            _get_placeholder_points(p),
            cameras,
            dist_cull,
            vis_cull,
            frame_start,
            frame_end,
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


def _find_placeholder_collection_name() -> str:
    names = sorted(c.name for c in bpy.data.collections if c.name.startswith("placeholders:"))
    if not names:
        raise ValueError("No placeholder collection found (expected prefix 'placeholders:').")
    return names[0]


def _get_cameras(camera_collection_name: str | None):
    if camera_collection_name is None:
        cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
        if not cams:
            raise ValueError("No cameras found in scene.")
        return cams
    cam_col = _get_collection(camera_collection_name)
    cams = [o for o in cam_col.objects if o.type == "CAMERA"]
    if not cams:
        raise ValueError(f"No cameras found in collection: {camera_collection_name}")
    return cams


def _configure_cameras_for_full_mode(cameras: list[bpy.types.Object]):
    if _IMPORT_ERROR is not None:
        return
    try:
        from infinigen.core.placement import camera as cam_mod

        for cam in cameras:
            cam_mod.adjust_camera_sensor(cam)
    except Exception as e:  # pragma: no cover - depends on scene and optional internals
        logger.warning("Could not auto-adjust camera sensors for full mode: %s", e)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--placeholder-collection", default=None)
    parser.add_argument("--camera-collection", default=None)
    parser.add_argument("--dist-cull", type=float, default=200.0)
    parser.add_argument("--vis-cull", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--require-full-infinigen", action="store_true")
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("/tmp/filter_populate_targets_bench.json"))
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)

    torch_backend = "none"
    try:
        import torch

        if torch.cuda.is_available():
            torch_backend = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_backend = "mps"
        else:
            torch_backend = "cpu"
    except Exception:
        torch_backend = "none"

    if _IMPORT_ERROR is not None:
        logger.warning("Using fallback benchmark mode due to import error: %s", _IMPORT_ERROR)
        if args.require_full_infinigen:
            raise RuntimeError(f"Full infinigen mode required but unavailable: {_IMPORT_ERROR}")

    if args.placeholder_collection is None:
        args.placeholder_collection = _find_placeholder_collection_name()
    placeholders_col = _get_collection(args.placeholder_collection)
    placeholders = [o for o in placeholders_col.objects if o.parent is None]
    cameras = _get_cameras(args.camera_collection)
    _configure_cameras_for_full_mode(cameras)

    if not placeholders:
        raise ValueError(f"No placeholders found in collection: {args.placeholder_collection}")

    if args.frame_start is None:
        args.frame_start = int(bpy.context.scene.frame_current)
    if args.frame_end is None:
        args.frame_end = int(args.frame_start)

    legacy_t, legacy_out = _median_time(
        lambda: filter_populate_targets_legacy(
            placeholders,
            cameras,
            dist_cull=args.dist_cull,
            vis_cull=args.vis_cull,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        ),
        repeat=max(1, args.repeat),
    )

    batched_t, batched_out = _median_time(
        lambda: _filter_populate_targets_batched(
            placeholders,
            cameras,
            args.dist_cull,
            args.vis_cull,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        ),
        repeat=max(1, args.repeat),
    )

    report = {
        "run_label": args.run_label,
        "placeholder_collection": args.placeholder_collection,
        "camera_collection": args.camera_collection,
        "num_placeholders": len(placeholders),
        "num_cameras": len(cameras),
        "repeat": int(max(1, args.repeat)),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "mode": "full-infinigen" if _IMPORT_ERROR is None else "fallback",
        "torch_backend": torch_backend,
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
    logger.info("%s", json.dumps(report, indent=2))
    logger.info("WROTE %s", args.output)


if __name__ == "__main__":
    main()
