# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2
import imageio
import numpy as np
import torch
from imageio.v3 import imread, imwrite

from infinigen.core.util.device import get_torch_device, setup_torch_runtime
from infinigen.tools.dataset_loader import get_frame_path

logger = logging.getLogger(__name__)

"""
Usage: python -m tools.ground_truth.depth_to_normals <scene-folder> <frame-index>
Output:
- testbed
    - A.png # Original image
    - B.png # Surface normals from depth + finite-difference
    - C.png # Surface normals from geometry
"""


_coord_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}


def unproject_torch(depth, K_inv):
    H, W = depth.shape
    key = (str(depth.device), int(H), int(W))
    img_coords = _coord_cache.get(key)
    if img_coords is None:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=depth.dtype),
            torch.arange(W, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        img_coords = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
        _coord_cache[key] = img_coords
    return depth.unsqueeze(-1) * torch.matmul(img_coords, K_inv.T)


def normalize_torch(v, eps=1e-8):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("frame", type=int)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("testbed"))
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True)

    depth_path = get_frame_path(args.folder, 0, args.frame, "Depth_npy")
    normal_path = get_frame_path(args.folder, 0, args.frame, "SurfaceNormal_png")
    image_path = get_frame_path(args.folder, 0, args.frame, "Image_png")
    camview_path = get_frame_path(args.folder, 0, args.frame, "camview_npz")
    assert depth_path.exists(), depth_path
    assert image_path.exists(), image_path
    assert camview_path.exists(), camview_path
    assert normal_path.exists(), normal_path

    image = imread(image_path)
    depth = np.load(depth_path)
    K = np.load(camview_path)["K"]

    device = get_torch_device(args.device)
    setup_torch_runtime(device)

    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    K_inv = torch.linalg.inv(K_t)
    cam_coords = unproject_torch(depth_t, K_inv)
    cam_coords = cam_coords * torch.tensor([1.0, -1.0, -1.0], device=device)

    mask = ~torch.isinf(depth_t)

    vy = normalize_torch(cam_coords[1:, 1:] - cam_coords[:-1, 1:])
    vx = normalize_torch(cam_coords[1:, 1:] - cam_coords[1:, :-1])
    cross_prod = torch.cross(vy, vx, dim=-1)
    normals = normalize_torch(cross_prod)
    normals = torch.nan_to_num(normals)
    normals[~mask[1:, 1:]] = 0

    normals_color = torch.round((normals + 1) * (255 / 2)).to(torch.uint8).cpu().numpy()
    target_shape = imageio.imread(normal_path).shape[:2][::-1]
    normals_color = cv2.resize(normals_color, target_shape)

    imwrite(args.output / "A.png", image)
    logger.info(f"Wrote {args.output / 'A.png'}")
    imwrite(args.output / "B.png", normals_color)
    logger.info(f"Wrote {args.output / 'B.png'}")
    shutil.copyfile(normal_path, args.output / "C.png")
    logger.info(f"Wrote {args.output / 'C.png'}")
