# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from imageio.v3 import imread, imwrite

from infinigen.tools.dataset_loader import get_frame_path

logger = logging.getLogger(__name__)

_grid_cache: dict[tuple[int, int], np.ndarray] = {}

"""
Usage: python -m tools.ground_truth.rigid_warp <scene-folder> <frame-index-i> <frame-index-j>
Output:
- testbed
    - A.png # Image at frame i
    - B.png # Image at frame j, warped to i
    - C.png # Image at frame j
"""


def transform(T, p):
    assert T.shape == (4, 4)
    return np.einsum("h w j, i j -> h w i", p, T[:3, :3]) + T[:3, 3]


def from_homog(x):
    return x[..., :-1] / x[..., [-1]]


def pixel_grid(H, W):
    key = (int(H), int(W))
    cached = _grid_cache.get(key)
    if cached is not None:
        return cached
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    cached = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    _grid_cache[key] = cached
    return cached


def reproject(depth1, pose1, pose2_inv, K1_inv, K2):
    H, W = depth1.shape
    img_1_coords = pixel_grid(H, W)
    cam1_coords = np.einsum("h w, h w j, i j -> h w i", depth1, img_1_coords, K1_inv)
    rel_pose = pose2_inv @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum("h w j, i j -> h w i", cam2_coords, K2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("frame_1", type=int)
    parser.add_argument("frame_2", type=int)
    parser.add_argument("--output", type=Path, default=Path("testbed"))
    args = parser.parse_args()

    depth_path = get_frame_path(args.folder, 0, args.frame_1, "Depth_npy")
    image1_path = get_frame_path(args.folder, 0, args.frame_1, "Image_png")
    image2_path = get_frame_path(args.folder, 0, args.frame_2, "Image_png")
    camview1_path = get_frame_path(args.folder, 0, args.frame_1, "camview_npz")
    camview2_path = get_frame_path(args.folder, 0, args.frame_2, "camview_npz")

    image2 = imread(image2_path)
    image1 = imread(image1_path)
    depth1 = np.load(depth_path)
    pose1 = np.load(camview1_path)["T"]
    pose2 = np.load(camview2_path)["T"]
    K1 = np.load(camview1_path)["K"]
    K2 = np.load(camview2_path)["K"]
    pose2_inv = np.linalg.inv(pose2)
    K1_inv = np.linalg.inv(K1)

    H, W, _ = image1.shape
    depth1 = cv2.resize(
        np.load(depth_path), dsize=(W, H), interpolation=cv2.INTER_LINEAR
    )

    img2_coords = reproject(depth1, pose1, pose2_inv, K1_inv, K2)

    warped_image = cv2.remap(
        image2, img2_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR
    )

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image1)
    logger.info(f"Wrote {args.output / 'A.png'}")
    imwrite(args.output / "C.png", image2)
    logger.info(f"Wrote {args.output / 'C.png'}")
    imwrite(args.output / "B.png", warped_image)
    logger.info(f"Wrote {args.output / 'B.png'}")
