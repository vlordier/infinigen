# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import colorsys
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from imageio.v3 import imread, imwrite

from infinigen.core.util.array_ops import unique_rows
from infinigen.tools.compress_masks import recover
from infinigen.tools.dataset_loader import get_frame_path

logger = logging.getLogger(__name__)

try:
    from einops import pack, rearrange
except ImportError as e:
    raise ImportError(
        "GT visualization requires `einops`. Please install optional extras via `pip install .[vis]`."
    ) from e

"""
Usage: python -m tools.ground_truth.segmentation_lookup <scene-folder> <frame-index> [--query <query>] [--boxes]
Output:
- testbed
    - A.png # Original image
    - B.png # Original image + mask/2D-bounding-boxes for the provided query
"""


def should_highlight_pixel(arr2d, set1d):
    """Compute boolean mask for items in arr2d that are also in set1d"""
    return np.isin(arr2d, set1d)


def compute_boxes(indices, binary_tag_mask):
    """Compute 2d bounding boxes for highlighted pixels"""
    H, W = binary_tag_mask.shape
    num_u = int(indices.max()) + 1
    x_min = np.full(num_u, W - 1, dtype=np.int32)
    y_min = np.full(num_u, H - 1, dtype=np.int32)
    x_max = np.full(num_u, -1, dtype=np.int32)
    y_max = np.full(num_u, -1, dtype=np.int32)

    flat_mask = binary_tag_mask.reshape(-1)
    if flat_mask.any():
        linear = np.flatnonzero(flat_mask)
        ys, xs = np.divmod(linear, W)
        obj_idx = indices.reshape(-1)[linear]
        np.minimum.at(x_min, obj_idx, xs)
        np.maximum.at(x_max, obj_idx, xs)
        np.minimum.at(y_min, obj_idx, ys)
        np.maximum.at(y_max, obj_idx, ys)

    return np.stack((x_min, y_min, x_max, y_max), axis=-1)


# Deterministic, but probably slow. Good enough for visualization.
def arr2color(e):
    s = np.random.RandomState(np.array(e, dtype=np.uint32))
    return (
        np.asarray(colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1)) * 255
    ).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("frame", type=int)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--boxes", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("testbed"))
    args = parser.parse_args()

    # Load images & masks
    object_segmentation_mask = recover(
        np.load(get_frame_path(args.folder, 0, args.frame, "ObjectSegmentation_npz"))
    )
    instance_segmentation_mask = recover(
        np.load(get_frame_path(args.folder, 0, args.frame, "InstanceSegmentation_npz"))
    )
    image = imread(get_frame_path(args.folder, 0, args.frame, "Image_png"))
    object_json = json.loads(
        get_frame_path(args.folder, 0, args.frame, "Objects_json").read_text()
    )
    H, W = object_segmentation_mask.shape
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

    # Identify objects visible in the image
    unique_object_idxs = set(np.unique(object_segmentation_mask))
    present_objects = [
        obj for obj in object_json if (obj["object_index"] in unique_object_idxs)
    ]

    # Complain if the query isn't valid/present
    unique_names = sorted({q["name"] for q in present_objects})
    if args.query is None:
        logger.info('`--query` not specified. Choices are:')
        for qn in unique_names:
            logger.info(f'- {qn}')
        sys.exit(0)
    elif not any((args.query.lower() in name.lower()) for name in unique_names):
        logger.info(f'''"{args.query}" doesn't match any object names in this image. Choices are:''')
        for qn in unique_names:
            logger.info(f'- {qn}')
        sys.exit(0)

    # Mask the pixels with any relevant object
    objects_to_highlight = [
        obj for obj in present_objects if (args.query.lower() in obj["name"].lower())
    ]
    highlighted_pixels = should_highlight_pixel(
        object_segmentation_mask,
        np.array([o["object_index"] for o in objects_to_highlight]),
    )
    assert highlighted_pixels.dtype == bool

    # Assign unique colors to each object instance
    combined_mask, _ = pack(
        [object_segmentation_mask, instance_segmentation_mask], "h w *"
    )
    combined_mask = rearrange(combined_mask, "h w d -> (h w) d")
    uniq_instances, indices = unique_rows(combined_mask, return_inverse=True)
    unique_colors = np.stack([arr2color(row) for row in uniq_instances])

    if args.boxes:
        bbox = compute_boxes(indices.reshape((H, W)), highlighted_pixels)
        m = bbox[:, 3] >= 0  # Ignore objects which weren't queried
        bbox = bbox[m]
        unique_colors = unique_colors[m]
        uniq_instances = uniq_instances[m]
        canvas = np.copy(image)
        for (x_min, y_min, x_max, y_max), color, idx, ui in zip(
            bbox, unique_colors, np.arange(m.size)[m], uniq_instances
        ):
            canvas = cv2.rectangle(
                canvas,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                color=color.tolist(),
                thickness=2,
            )
    else:
        colors_for_instances = unique_colors[indices].reshape((H, W, 3))
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        highlight_ids = np.array([obj["object_index"] for obj in objects_to_highlight])
        m = should_highlight_pixel(object_segmentation_mask, highlight_ids)
        canvas[m] = colors_for_instances[m]

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image)
    logger.info(f"Wrote {args.output / 'A.png'}")
    imwrite(args.output / "B.png", canvas)
    logger.info(f"Wrote {args.output / 'B.png'}")
