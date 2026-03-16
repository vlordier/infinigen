# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import submitit
import torch
import torch.nn.functional as F
from tqdm import tqdm

from infinigen.core.util.device import get_torch_device, setup_torch_runtime
from infinigen.tools.suffixes import parse_suffix

_coords_cache: dict[tuple[str, int, int], torch.Tensor] = {}


def coords_grid(batch, ht, wd, device):
    key = (str(device), int(ht), int(wd))
    cached = _coords_cache.get(key)
    if cached is None:
        coords = torch.meshgrid(
            torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
        )
        coords = torch.stack(coords[::-1], dim=0).float()
        cached = coords.unsqueeze(0)
        _coords_cache[key] = cached
    if batch == 1:
        return cached
    return cached.expand(batch, -1, -1, -1)


def reproject_torch(depth, src_cam, dst_cam, device):
    H, W = depth.shape
    coords0 = coords_grid(1, H, W, device)[0].permute(1, 2, 0)
    ones = torch.ones((H, W, 1), dtype=depth.dtype, device=device)
    img_coords = torch.cat((coords0, ones), dim=-1)

    cam_src = depth.unsqueeze(-1) * torch.matmul(img_coords, src_cam["K_inv"].T)
    rel_pose = torch.matmul(dst_cam["T_inv"], src_cam["T"])
    cam_dst = torch.matmul(cam_src, rel_pose[:3, :3].T) + rel_pose[:3, 3]
    proj = torch.matmul(cam_dst, dst_cam["K"].T)

    z = proj[..., 2:3]
    z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    return proj[..., :2] / z


def induced_flow_torch(depth0, depth1, cam0, cam1, device):
    H, W = depth0.shape
    coords0 = coords_grid(1, H, W, device)[0].permute(1, 2, 0)
    coords1 = reproject_torch(depth0, cam0, cam1, device)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords0 = coords_grid(1, H, W, device)[0].permute(1, 2, 0)
    coords1 = reproject_torch(depth1, cam1, cam0, device)
    flow_10 = coords1 - coords0

    return flow_01, flow_10


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def check_cycle_consistency(flow_01, flow_10, threshold=1, device=None):
    if device is None:
        device = get_torch_device()
    with torch.inference_mode():
        if torch.is_tensor(flow_01):
            flow_01 = flow_01.to(device=device, dtype=torch.float32)
        else:
            flow_01 = torch.as_tensor(flow_01, device=device, dtype=torch.float32)
        if torch.is_tensor(flow_10):
            flow_10 = flow_10.to(device=device, dtype=torch.float32)
        else:
            flow_10 = torch.as_tensor(flow_10, device=device, dtype=torch.float32)

        if flow_01.ndim == 3:
            flow_01 = flow_01.permute(2, 0, 1)[None]
        if flow_10.ndim == 3:
            flow_10 = flow_10.permute(2, 0, 1)[None]

        H, W = flow_01.shape[-2:]
        coords = coords_grid(1, H, W, flow_01.device)
        coords1 = coords + flow_01
        flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
        cycle = flow_reprojected + flow_01
        cycle = torch.norm(cycle, dim=1)
        mask = (cycle < threshold).float()
    return mask


def compute_covisibility(depth0, depth1, camview0, camview1, device=None):
    if device is None:
        device = get_torch_device()

    def _as_cam_tensors(cam):
        K = cam.get("K")
        K_inv = cam.get("K_inv")
        T = cam.get("T")
        T_inv = cam.get("T_inv")

        K_t = torch.as_tensor(K, device=device, dtype=torch.float32)
        T_t = torch.as_tensor(T, device=device, dtype=torch.float32)
        if K_inv is None:
            K_inv_t = torch.linalg.inv(K_t)
        else:
            K_inv_t = torch.as_tensor(K_inv, device=device, dtype=torch.float32)
        if T_inv is None:
            T_inv_t = torch.linalg.inv(T_t)
        else:
            T_inv_t = torch.as_tensor(T_inv, device=device, dtype=torch.float32)
        return {"K": K_t, "K_inv": K_inv_t, "T": T_t, "T_inv": T_inv_t}

    camview0 = _as_cam_tensors(camview0)
    camview1 = _as_cam_tensors(camview1)
    depth0 = torch.as_tensor(depth0, device=device, dtype=torch.float32)
    depth1 = torch.as_tensor(depth1, device=device, dtype=torch.float32)
    return compute_covisibility_prepared(depth0, depth1, camview0, camview1, device)


def compute_covisibility_prepared(depth0, depth1, camview0, camview1, device):
    flow_01, flow_10 = induced_flow_torch(depth0, depth1, camview0, camview1, device)
    mask = check_cycle_consistency(flow_01, flow_10, device=device)
    return float(mask.mean().item())


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
            cam_id: torch.as_tensor(
                np.load(target_folder / scene / f"depths/{cam_id}.npy"),
                dtype=torch.float32,
                device=device,
            )
            for cam_id in cam_ids
        }
        cam_cache = {}
        for cam_id in cam_ids:
            with np.load(target_folder / scene / f"cameras/{cam_id}.npz") as cam_npz:
                K = torch.as_tensor(cam_npz["K"], dtype=torch.float32, device=device)
                T = torch.as_tensor(cam_npz["T"], dtype=torch.float32, device=device)
                cam_cache[cam_id] = {
                    "K": K,
                    "K_inv": torch.linalg.inv(K),
                    "T": T,
                    "T_inv": torch.linalg.inv(T),
                }

        n_cams = len(cam_ids)
        cov_matrix = np.ones((n_cams, n_cams), dtype=np.float32)
        for i in range(n_cams):
            depth0 = depth_cache[cam_ids[i]]
            camview0 = cam_cache[cam_ids[i]]
            for j in range(i + 1, n_cams):
                depth1 = depth_cache[cam_ids[j]]
                camview1 = cam_cache[cam_ids[j]]
                cov = compute_covisibility_prepared(depth0, depth1, camview0, camview1, device)
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
