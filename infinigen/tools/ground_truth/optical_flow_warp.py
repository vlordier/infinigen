# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imageio.v3 import imread, imwrite

from infinigen.core.util.device import get_torch_device, setup_torch_runtime
from infinigen.tools.dataset_loader import get_frame_path

logger = logging.getLogger(__name__)


_coord_cache: dict[tuple[int, int], np.ndarray] = {}
_coord_cache_torch: dict[tuple[str, int, int], torch.Tensor] = {}


def _base_coords(H, W):
    key = (int(H), int(W))
    cached = _coord_cache.get(key)
    if cached is not None:
        return cached
    cached = np.stack(
        np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"),
        axis=-1,
    )
    _coord_cache[key] = cached
    return cached


def _base_coords_torch(H, W, device):
    key = (str(device), int(H), int(W))
    cached = _coord_cache_torch.get(key)
    if cached is not None:
        return cached
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cached = torch.stack((xx, yy), dim=-1)
    _coord_cache_torch[key] = cached
    return cached

"""
Usage: python -m tools.ground_truth.rigid_warp <scene-folder> <frame-index-i>
Output:
- testbed
    - A.png # Image at frame i
    - B.png # Image at frame i+1, warped to i
    - C.png # Image at frame i+1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("frame", type=int)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("testbed"))
    args = parser.parse_args()

    device = get_torch_device(args.device)
    setup_torch_runtime(device)
    use_torch = device.type != "cpu"
    flow3d_path = get_frame_path(args.folder, 0, args.frame, "Flow3D_npy")
    image1_path = get_frame_path(args.folder, 0, args.frame, "Image_png")
    image2_path = get_frame_path(args.folder, 0, args.frame + 1, "Image_png")
    assert flow3d_path.exists()
    assert image1_path.exists()
    assert image2_path.exists()

    image2 = imread(image2_path)
    image1 = imread(image1_path)
    H, W, _ = image1.shape

    flow = np.load(flow3d_path)
    if use_torch:
        flow_t = torch.as_tensor(flow[..., :2], dtype=torch.float32, device=device)
        flow_t = flow_t.permute(2, 0, 1).unsqueeze(0)
        flow2d = F.interpolate(flow_t, size=(H, W), mode="bilinear", align_corners=False)
        flow2d = flow2d[0].permute(1, 2, 0)
        new_coords = (flow2d + _base_coords_torch(H, W, device)).cpu().numpy()
    else:
        flow2d = cv2.resize(flow, dsize=(W, H), interpolation=cv2.INTER_LINEAR)[..., :2]
        new_coords = flow2d.astype(np.float32, copy=False) + _base_coords(H, W)

    warped_image = cv2.remap(
        image2, new_coords.astype(np.float32, copy=False), None, interpolation=cv2.INTER_LINEAR
    )

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image1)
    logger.info(f"Wrote {args.output / 'A.png'}")
    imwrite(args.output / "C.png", image2)
    logger.info(f"Wrote {args.output / 'C.png'}")
    imwrite(args.output / "B.png", warped_image)
    logger.info(f"Wrote {args.output / 'B.png'}")
