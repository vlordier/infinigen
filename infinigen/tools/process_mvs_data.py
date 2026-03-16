# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import submitit
import torch
import torch.nn.functional as F
from tqdm import tqdm

from infinigen.core.util.device import get_torch_device, setup_torch_runtime
from infinigen.tools.suffixes import parse_suffix


# these functions till check_cycle_consistency are from https://github.com/princeton-vl/SEA-RAFT
def transform(T, p):
    assert T.shape == (4, 4)
    return np.einsum("H W j, i j -> H W i", p, T[:3, :3]) + T[:3, 3]


def from_homog(x):
    return x[..., :-1] / x[..., [-1]]


_coords_cache: Dict[Tuple[str, int, int, int], torch.Tensor] = {}


def coords_grid(batch, ht, wd, device):
    key = (str(device), int(batch), int(ht), int(wd))
    cached = _coords_cache.get(key)
    if cached is not None:
        return cached
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(batch, 1, 1, 1)
    _coords_cache[key] = grid
    return grid


def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum(
        "H W, H W j, i j -> H W i", depth1, img_1_coords, np.linalg.inv(K1)
    )
    rel_pose = np.linalg.inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum("H W j, i j -> H W i", cam2_coords, K2))


def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data["T0"], data["T1"], data["K0"], data["K1"])

    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords1 = reproject(depth1, data["T1"], data["T0"], data["K1"], data["K0"])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0

    return flow_01, flow_10


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def check_cycle_consistency(flow_01, flow_10, threshold=1, device=None):
    if device is None:
        device = get_torch_device()
    with torch.no_grad():
        flow_01 = torch.as_tensor(flow_01, device=device).permute(2, 0, 1)[None].float()
        flow_10 = torch.as_tensor(flow_10, device=device).permute(2, 0, 1)[None].float()
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < threshold).float()
    return mask[0].cpu().numpy()


def compute_covisibility(depth0, depth1, camview0, camview1, device=None):
    data = {}
    data["K0"] = camview0["K"]
    data["K1"] = camview1["K"]
    data["T0"] = camview0["T"]
    data["T1"] = camview1["T"]
    flow_01, flow_10 = induced_flow(depth0, depth1, data)
    mask = check_cycle_consistency(flow_01, flow_10, device=device)
    return mask.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", type=Path, default=None)
    parser.add_argument("--target_folder", type=Path)
    parser.add_argument("--postprocess_only", type=int, default=False)
    args = parser.parse_args()

    source_folder = args.source_folder
    target_folder = args.target_folder

    if not args.postprocess_only:
        scenes = [
            x for x in os.listdir(source_folder) if os.path.isdir(source_folder / x)
        ]
        for scene in tqdm(scenes):
            image_dir = source_folder / scene / "frames/Image/camera_0"
            if not os.path.exists(image_dir):
                continue
            images = [x for x in os.listdir(image_dir) if x.endswith(".png")]
            for image in images:
                im = cv2.imread(image_dir / image)
                if im.mean() < 20:
                    continue
                camera_path = (
                    source_folder
                    / scene
                    / f"frames/camview/camera_0/camview{image[5:-4]}.npz"
                )
                depth_path = (
                    source_folder
                    / scene
                    / f"frames/Depth/camera_0/Depth{image[5:-4]}.npy"
                )
                if not os.path.exists(camera_path):
                    continue
                if not os.path.exists(depth_path):
                    continue
                (target_folder / scene / "images").mkdir(parents=True, exist_ok=True)
                (target_folder / scene / "cameras").mkdir(parents=True, exist_ok=True)
                (target_folder / scene / "depths").mkdir(parents=True, exist_ok=True)
                cam_id = parse_suffix(image)["cam_rig"]
                shutil.copy(
                    image_dir / image,
                    target_folder / scene / "images" / f"{cam_id:04d}.png",
                )
                shutil.copy(
                    camera_path, target_folder / scene / "cameras" / f"{cam_id:04d}.npz"
                )
                shutil.copy(
                    depth_path, target_folder / scene / "depths" / f"{cam_id:04d}.npy"
                )

    scenes = os.listdir(target_folder)
    device = get_torch_device()
    setup_torch_runtime(device)

    def worker(scene):
        cam_ids = sorted(
            [
            x[:-4]
            for x in os.listdir(target_folder / scene / "images")
            if x.endswith(".png")
            ]
        )
        depth_cache = {
            cam_id: np.load(target_folder / scene / f"depths/{cam_id}.npy")
            for cam_id in cam_ids
        }
        cam_cache = {}
        for cam_id in cam_ids:
            with np.load(target_folder / scene / f"cameras/{cam_id}.npz") as cam_npz:
                cam_cache[cam_id] = {"K": cam_npz["K"], "T": cam_npz["T"]}

        n_cams = len(cam_ids)
        cov_matrix = np.ones((n_cams, n_cams), dtype=np.float32)
        for i in range(n_cams):
            depth0 = depth_cache[cam_ids[i]]
            camview0 = cam_cache[cam_ids[i]]
            for j in range(i + 1, n_cams):
                depth1 = depth_cache[cam_ids[j]]
                camview1 = cam_cache[cam_ids[j]]
                cov = compute_covisibility(depth0, depth1, camview0, camview1, device=device)
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov

        with open(target_folder / scene / "pairs.txt", "w") as f:
            for i, cam_id0 in enumerate(cam_ids):
                f.write(f"{cam_id0} ")
                for j, cam_id1 in enumerate(cam_ids):
                    if i == j:
                        continue
                    cov = cov_matrix[i, j]
                    f.write(f" {cam_id1} {cov}")
                f.write("\n")

        thumbnails = []
        for cam_id in cam_ids:
            im = cv2.imread(str(target_folder / scene / "images" / f"{cam_id}.png"))
            H, W = im.shape[:2]
            thumbnails.append(cv2.resize(im, (W // 10, H // 10)))
        thumbnails = np.concatenate(thumbnails, 1)
        cv2.imwrite(str(target_folder / scene / "thumbnails.png"), thumbnails)

    log_folder = "~/sc/logs/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=10, slurm_partition="allcs")
    for scene in scenes:
        job = executor.submit(worker, scene)
