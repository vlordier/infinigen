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
    return get_mask_torch_batch(depth[None], flow[None], dst_depth[None], device)[0]


def get_mask_torch_batch(depth, flow, dst_depth, device):
    depth_t = torch.as_tensor(depth, device=device, dtype=torch.float32)
    flow_t = torch.as_tensor(flow, device=device, dtype=torch.float32)
    dst_depth_t = torch.as_tensor(dst_depth, device=device, dtype=torch.float32)

    B, H, W = depth_t.shape
    base_xy = _torch_base_grid(H, W, device).unsqueeze(0)
    target_xy = base_xy + flow_t[..., :2] * 2.0
    target_z = depth_t + flow_t[..., 2]

    dst_depth_t = dst_depth_t[:, None]
    denom_x = max(W - 1, 1)
    denom_y = max(H - 1, 1)
    offsets = _torch_neighbor_offsets(device)
    K = offsets.shape[0]
    sx = target_xy[..., 0].unsqueeze(1) + offsets[:, 0].view(1, -1, 1, 1)
    sy = target_xy[..., 1].unsqueeze(1) + offsets[:, 1].view(1, -1, 1, 1)
    gx = 2.0 * sx / float(denom_x) - 1.0
    gy = 2.0 * sy / float(denom_y) - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(B * K, H, W, 2)

    sampled = F.grid_sample(
        dst_depth_t.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, 1, H, W),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0].reshape(B, K, H, W)

    valid = target_z.unsqueeze(1) >= 0
    mask = valid & (target_z.unsqueeze(1) <= sampled)
    return mask.any(dim=1).cpu().numpy()


def save_mask(file_path, data_type, mask):
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


def flush_torch_batch(pending, device):
    if not pending:
        return
    depth = np.stack([x[2] for x in pending], axis=0)
    flow = np.stack([x[4] for x in pending], axis=0)
    dst_depth = np.stack([x[3] for x in pending], axis=0)
    masks = get_mask_torch_batch(depth, flow, dst_depth, device)
    for (file_path, data_type, *_), mask in zip(pending, masks):
        save_mask(file_path, data_type, mask)


def choose_torch_mode(backend, device_type, workload_size):
    if backend == "cpu":
        return False
    if backend == "torch":
        return device_type != "cpu"
    if device_type == "cuda":
        return True
    if device_type == "mps":
        # MPS shows gains when enough frames are processed per run.
        return workload_size >= 8
    return False


def auto_batch_size(device_type, workload_size):
    if device_type == "cuda":
        return 32
    if device_type == "mps":
        if workload_size >= 16:
            return 16
        return 8
    return 1


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
    parser.add_argument("--backend", type=str, choices=("auto", "cpu", "torch"), default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    assert args.target_frames_dir.exists()
    assert args.target_frames_dir.name.startswith("frames_")

    candidate_files = [
        file_path
        for file_path in sorted(args.target_frames_dir.glob("*.npy"))
        if file_path.name.split("_")[0] in {"Flow3D", "PointTraj3D"}
    ]

    device = get_torch_device(args.device)
    setup_torch_runtime(device)
    use_torch = choose_torch_mode(args.backend, device.type, len(candidate_files))
    batch_size = args.batch_size if args.batch_size > 0 else auto_batch_size(device.type, len(candidate_files))

    if use_torch and batch_size < 1:
        batch_size = 1

    pending = []
    for file_path in candidate_files:
        info = parse_suffix(file_path.name)
        data_type = file_path.name.split("_")[0]
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
            sample_shape = flow.shape
            if pending and pending[0][4].shape != sample_shape:
                flush_torch_batch(pending, device)
                pending = []
            pending.append((file_path, data_type, depth, dst_depth, flow))
            if len(pending) >= batch_size:
                flush_torch_batch(pending, device)
                pending = []
        else:
            mask = get_mask(depth, flow, dst_depth)
            save_mask(file_path, data_type, mask)
    if use_torch:
        flush_torch_batch(pending, device)
