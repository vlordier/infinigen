# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from suffixes import get_suffix, parse_suffix


_grid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _get_base_grid(H, W):
    key = (int(H), int(W))
    cached = _grid_cache.get(key)
    if cached is not None:
        return cached
    y, x = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    _grid_cache[key] = (y, x)
    return y, x


def _sample_depth(depth, map_x, map_y):
    return cv2.remap(
        depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def get_mask(depth, flow, dst_depth):
    H, W = depth.shape
    base_y, base_x = _get_base_grid(H, W)
    dst_depth = dst_depth.astype(np.float32, copy=False)
    target_y = base_y + flow[:, :, 1].astype(np.float32, copy=False) * 2.0
    target_x = base_x + flow[:, :, 0].astype(np.float32, copy=False) * 2.0
    target_z = depth.astype(np.float32, copy=False) + flow[:, :, 2].astype(np.float32, copy=False)

    mask = np.zeros((H, W), dtype=bool)
    for dy, dx in ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        sampled = _sample_depth(dst_depth, target_x + dx, target_y + dy)
        mask |= (target_z >= 0) & (target_z <= sampled)

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_frames_dir", type=Path)
    parser.add_argument("point_traj_source_frame", type=int)
    args = parser.parse_args()
    assert args.target_frames_dir.exists()
    assert args.target_frames_dir.name.startswith("frames_")

    for file_path in args.target_frames_dir.glob("*.npy"):
        info = parse_suffix(file_path.name)
        data_type = file_path.name.split("_")[0]
        if not file_path.name.endswith(".npy"):
            continue
        if data_type == "Flow3D":
            depth_info = dict(info)
            depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
            depth_info["frame"] += 1
            dst_depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
        elif data_type == "PointTraj3D":
            depth_info = dict(info)
            depth_info["frame"] = args.point_traj_source_frame
            depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
            depth_info["frame"] = info["frame"]
            dst_depth = np.load(
                file_path.parent / ("Depth" + get_suffix(depth_info) + ".npy")
            )
        else:
            continue
        mask = get_mask(depth, np.load(file_path), dst_depth)
        np.save(
            file_path.parent / (data_type + "Mask" + file_path.name[len(data_type) :]),
            mask,
        )
        cv2.imwrite(
            str(
                file_path.parent
                / (data_type + "Mask" + file_path.name[len(data_type) : -4] + ".png")
            ),
            mask.astype(np.uint8) * 255,
        )
