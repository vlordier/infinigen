# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from suffixes import get_suffix, parse_suffix

from infinigen.core.util.device import get_torch_device, setup_torch_runtime

_grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_torch_grid_cache: dict[tuple[str, int, int], torch.Tensor] = {}
_offset_cache_torch: dict[str, torch.Tensor] = {}


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


def _torch_base_grid(H, W, device):
    key = (str(device), int(H), int(W))
    cached = _torch_grid_cache.get(key)
    if cached is not None:
        return cached
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cached = torch.stack((x, y), dim=-1)
    _torch_grid_cache[key] = cached
    return cached


def _torch_neighbor_offsets(device):
    key = str(device)
    cached = _offset_cache_torch.get(key)
    if cached is not None:
        return cached
    cached = torch.tensor(
        ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        dtype=torch.float32,
        device=device,
    )
    _offset_cache_torch[key] = cached
    return cached


def get_mask_torch(depth, flow, dst_depth, device):
    depth_t = torch.as_tensor(depth, device=device, dtype=torch.float32)
    flow_t = torch.as_tensor(flow, device=device, dtype=torch.float32)
    dst_depth_t = torch.as_tensor(dst_depth, device=device, dtype=torch.float32)

    H, W = depth_t.shape
    base_xy = _torch_base_grid(H, W, device)
    target_xy = base_xy + flow_t[..., :2] * 2.0
    target_z = depth_t + flow_t[..., 2]

    dst_depth_t = dst_depth_t[None, None]
    denom_x = max(W - 1, 1)
    denom_y = max(H - 1, 1)
    offsets = _torch_neighbor_offsets(device)
    sx = target_xy[..., 0].unsqueeze(0) + offsets[:, 0].view(-1, 1, 1)
    sy = target_xy[..., 1].unsqueeze(0) + offsets[:, 1].view(-1, 1, 1)
    gx = 2.0 * sx / float(denom_x) - 1.0
    gy = 2.0 * sy / float(denom_y) - 1.0
    grid = torch.stack((gx, gy), dim=-1)

    sampled = F.grid_sample(
        dst_depth_t.expand(offsets.shape[0], -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]

    valid = target_z >= 0
    mask = valid & (target_z.unsqueeze(0) <= sampled)
    return mask.any(dim=0).cpu().numpy()


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
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    assert args.target_frames_dir.exists()
    assert args.target_frames_dir.name.startswith("frames_")

    device = get_torch_device(args.device)
    setup_torch_runtime(device)
    use_torch = device.type != "cpu"

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
        flow = np.load(file_path)
        if use_torch:
            mask = get_mask_torch(depth, flow, dst_depth, device)
        else:
            mask = get_mask(depth, flow, dst_depth)
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
