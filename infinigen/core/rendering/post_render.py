# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import argparse
import logging
import os

import OpenEXR

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.

import colorsys
from pathlib import Path

import cv2
import numpy as np
from imageio import imwrite
from matplotlib import pyplot as plt

from infinigen.core.util.array_ops import unique_rows
from infinigen.core.util.camera import get_3x4_RT_matrix_from_blender

logger = logging.getLogger(__name__)


def load_exr(path):
    assert Path(path).exists() and Path(path).suffix == ".exr", path
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        return _load_exr_openexr(path)

    # cv2 may return HxW (single channel) or HxWxC depending on EXR layout.
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)

    if img.ndim == 3:
        if img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        if img.shape[2] == 2:
            pad = np.zeros_like(img[..., :1])
            img = np.concatenate([img, pad], axis=2)
        if img.shape[2] >= 3:
            return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)

    # Last-resort path for unusual EXR channel encodings
    return _load_exr_openexr(path)


def _load_exr_openexr(path):
    file = OpenEXR.InputFile(str(path))
    header = file.header()
    channels = header["channels"]
    dw = header["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1

    def _read_chan(name):
        c = channels.get(name)
        if c is None:
            return None
        data = np.frombuffer(file.channel(name, c.type), np.float32)
        return data.reshape((h, w))

    r = _read_chan("R")
    g = _read_chan("G")
    b = _read_chan("B")

    if r is not None and g is not None and b is not None:
        return np.stack([r, g, b], axis=2)

    # Fallback to first available channel replicated to RGB
    first_name, first_meta = next(iter(channels.items()))
    data = np.frombuffer(file.channel(first_name, first_meta.type), np.float32)
    single = data.reshape((h, w))
    return np.repeat(single[..., None], 3, axis=2)


load_flow = load_exr


def load_single_channel(p):
    file = OpenEXR.InputFile(str(p))
    channel, channel_type = next(iter(file.header()["channels"].items()))
    match str(channel_type.type):
        case "FLOAT":
            np_type = np.float32
        case _:
            np_type = np.uint8
    data = np.frombuffer(file.channel(channel, channel_type.type), np_type)
    dw = file.header()["dataWindow"]
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    return data.reshape(sz)


def load_depth(p):
    return load_single_channel(p)


def load_normals(path, camera=None) -> np.ndarray:
    data = load_exr(path)
    if camera is not None:
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_np = np.array(RT)
        R_world2cv = RT_np[:3, :3]

        original_shape = data.shape
        normals_flat = data.reshape(-1, 3)

        # Transform normals from world space to camera space
        # Since normals are direction vectors, only rotation is applied (no translation)
        normals_cam = (R_world2cv @ normals_flat.T).T

        norms = np.linalg.norm(normals_cam, axis=1, keepdims=True)
        valid_mask = norms > 1e-6
        normals_cam[valid_mask.flatten()] /= norms[valid_mask.flatten()]

        data = normals_cam.reshape(original_shape)

    return data


def load_seg_mask(p):
    # load_single_channel reads the first channel in the EXR, which may be
    # IndexOB.A (alpha=1.0) rather than the actual IndexOB.R data.  Read
    # IndexOB.R explicitly when that channel is present.
    import OpenEXR
    file = OpenEXR.InputFile(str(p))
    hdr = file.header()
    channels = hdr["channels"]
    dw = hdr["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1

    def _read_chan(name):
        c = channels.get(name)
        if c is None:
            return None
        return np.frombuffer(file.channel(name, c.type), np.float32).reshape((h, w))

    r = _read_chan("IndexOB.R")
    if r is None:
        r = _read_chan("IndexMA.R")
    if r is not None:
        data = r
    else:
        data = load_single_channel(p)
    if np.issubdtype(data.dtype, np.floating):
        data = np.rint(data)
    return data.astype(np.int64)


def load_uniq_inst(p):
    """Load unique-instance segmentation from an emission-render EXR.

    Flat-shading renders write emission RGB colors (one unique color per
    instance) to the "UniqueInstances" file.  The EXR channels are named
    ``UniqueInstances.A``, ``UniqueInstances.B``, ``UniqueInstances.G``,
    ``UniqueInstances.R`` — we must read them in the correct RGB order
    rather than relying on the arbitrary ordering that OpenEXR returns.
    """
    import OpenEXR
    file = OpenEXR.InputFile(str(p))
    hdr = file.header()
    channels = hdr["channels"]
    dw = hdr["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1

    def _read_chan(name):
        c = channels.get(name)
        if c is None:
            return None
        data = np.frombuffer(file.channel(name, c.type), np.float32)
        return data.reshape((h, w))

    r = _read_chan("UniqueInstances.R")
    g = _read_chan("UniqueInstances.G")
    b = _read_chan("UniqueInstances.B")
    if r is not None and g is not None and b is not None:
        data = np.stack([r, g, b], axis=2)
    else:
        # Fallback: load_exr handles generic channel layouts
        data = load_exr(p)

    if np.issubdtype(data.dtype, np.floating):
        # Scale emission floats to [0, 65535] uint16 for stable color hashing
        data = np.clip(np.rint(data * 65535.0), 0, 65535).astype(np.uint16)
    return data


def colorize_flow(optical_flow):
    try:
        import flow_vis
    except ImportError:
        logger.warning(
            "Flow visualization requires the 'flow_vis' package. Please install via `pip install .[vis]."
        )
        return None

    flow_uv = optical_flow[..., :2]
    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_color


def colorize_normals(surface_normals):
    assert surface_normals.max() < 1 + 1e-4
    assert surface_normals.min() > -1 - 1e-4
    norm = np.linalg.norm(surface_normals, axis=2)
    color = np.round((surface_normals + 1) * (255 / 2)).astype(np.uint8)
    color[norm < 1e-4] = 0
    return color


def colorize_depth(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[..., :3] * 255, dtype=np.uint8)


def colorize_int_array(data, color_seed=0):
    H, W, *_ = data.shape
    data = data.reshape((H * W, -1))
    uniq, indices = unique_rows(data, return_inverse=True)
    # Keep visualization high-contrast: reserve pure black for the all-zero label
    # (commonly background). For non-zero labels, use a vivid categorical palette
    # first, then deterministic HSV colors as overflow for very high label counts.
    unique_colors = np.zeros((len(uniq), 3), dtype=np.uint8)

    vivid_palette = np.array(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 128, 255),
            (255, 0, 128),
            (0, 255, 128),
            (128, 255, 0),
        ],
        dtype=np.uint8,
    )
    palette_size = len(vivid_palette)
    palette_offset = int(color_seed) % palette_size
    palette_stride = 5  # coprime with 12 to traverse all palette entries

    golden_ratio_conjugate = 0.6180339887498949
    seed_phase = (float(color_seed) * 0.17320508075688773) % 1.0
    nonzero_rank = 0
    for i, row in enumerate(uniq):
        if np.all(row == 0):
            unique_colors[i] = (0, 0, 0)
            continue

        if nonzero_rank < palette_size:
            palette_idx = (palette_offset + nonzero_rank * palette_stride) % palette_size
            unique_colors[i] = vivid_palette[palette_idx]
        else:
            overflow_rank = nonzero_rank - palette_size
            h = (seed_phase + overflow_rank * golden_ratio_conjugate) % 1.0
            s = 0.95
            v = 0.95
            unique_colors[i] = (np.asarray(colorsys.hsv_to_rgb(h, s, v)) * 255).astype(
                np.uint8
            )
        nonzero_rank += 1

    return unique_colors[indices].reshape((H, W, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_path", type=Path, default=None)
    parser.add_argument("--depth_path", type=Path, default=None)
    parser.add_argument("--seg_path", type=Path, default=None)
    parser.add_argument("--uniq_inst_path", type=Path, default=None)
    parser.add_argument("--normals_path", type=Path, default=None)
    args = parser.parse_args()

    if args.flow_path is not None:
        try:
            flow_color = colorize_flow(load_flow(args.flow_path))
            if flow_color is not None:
                output_path = args.flow_path.with_suffix(".png")
                imwrite(output_path, flow_color)
                logger.info(f'Wrote {output_path}')
        except ModuleNotFoundError:
            logger.info("Flow visualization requires the 'flow_vis' package. Install it with 'pip install flow_vis'")
            pass

    if args.normals_path is not None:
        normal_color = colorize_normals(load_normals(args.normals_path))
        output_path = args.normals_path.with_suffix(".png")
        imwrite(output_path, normal_color)
        logger.info(f'Wrote {output_path}')

    if args.depth_path is not None:
        depth_color = colorize_depth(load_depth(args.depth_path))
        output_path = args.depth_path.with_suffix(".png")
        imwrite(output_path, depth_color)
        logger.info(f'Wrote {output_path}')

    if args.uniq_inst_path is not None:
        mask_color = colorize_int_array(load_uniq_inst(args.uniq_inst_path))
        output_path = args.uniq_inst_path.with_suffix(".png")
        imwrite(output_path, mask_color)
        logger.info(f'Wrote {output_path}')

    if args.seg_path is not None:
        mask_color = colorize_int_array(load_seg_mask(args.seg_path))
        output_path = args.seg_path.with_suffix(".png")
        imwrite(output_path, mask_color)
        logger.info(f'Wrote {output_path}')
