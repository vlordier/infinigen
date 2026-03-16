#!/usr/bin/env python3
"""OcMesher backend compatibility self-test.

Run inside Blender to instantiate the configured backend and perform a tiny
contract dry-run before full terrain export.

Usage:
  blender -b --python scripts/ocmesher_backend_self_test.py -- \
    --output /tmp/ocmesher_self_test.json
"""

import argparse
import json
import logging
from pathlib import Path

import bpy

from infinigen.terrain.core import ocmesher_backend_self_test

logger = logging.getLogger(__name__)


def _ensure_camera():
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if cameras:
        return cameras

    bpy.ops.object.camera_add(location=(0.0, -5.0, 2.0))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    return [cam]


def _parse_args():
    argv = list(__import__("sys").argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="OcMesher backend self-test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        default=[-10.0, 10.0, -10.0, 10.0, -10.0, 10.0],
        help="Bounds as xmin xmax ymin ymax zmin zmax",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail with non-zero code on backend self-test failure",
    )
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    cameras = _ensure_camera()
    result = ocmesher_backend_self_test(cameras, tuple(args.bounds), strict=args.strict)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info("OcMesher backend self-test result: %s", json.dumps(result, indent=2))

    if args.strict and not result.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
