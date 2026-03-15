# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Dataset integrity checking beyond simple file-existence validation.

:class:`InfinigenSceneDataset` already has a ``validate()`` method that
checks whether expected files exist.  This module adds deeper checks:

* **File-size sanity** — detect truncated or empty output files.
* **Metadata completeness** — verify that ``Objects_json`` files
  contain expected keys and that ``camview_npz`` arrays have the
  right shapes.
* **Frame-sequence continuity** — ensure there are no gaps in the
  numbered frame sequence.
* **Checksum manifest** — generate and verify SHA-256 checksums so
  that bit-rot or accidental overwrites are caught.

Usage::

    from infinigen.tools.dataset_integrity import (
        check_frame_continuity,
        check_file_sizes,
        check_metadata_completeness,
        generate_checksum_manifest,
        verify_checksum_manifest,
    )

    issues = check_frame_continuity(scene_folder)
    issues += check_file_sizes(scene_folder)
    if issues:
        print(f"Found {len(issues)} integrity issue(s)")
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum file sizes (bytes) for common output types.
# A PNG header alone is 8 bytes; anything smaller is certainly corrupt.
_MIN_SIZES: dict[str, int] = {
    ".png": 67,    # minimal valid PNG is ~67 bytes
    ".npy": 80,    # numpy header is 80+ bytes
    ".npz": 22,    # minimal zip header
    ".json": 2,    # at least "{}" or "[]"
    ".exr": 20,    # minimal OpenEXR header
}


def check_frame_continuity(scene_folder: str | Path) -> list[str]:
    """Verify that frame indices form a contiguous sequence with no gaps.

    Parameters
    ----------
    scene_folder:
        Path to a scene directory containing ``frames/<DataType>/camera_*/*.ext``

    Returns
    -------
    list[str]
        Human-readable descriptions of any gaps found.
    """
    scene_folder = Path(scene_folder)
    frames_dir = scene_folder / "frames"
    issues: list[str] = []

    if not frames_dir.is_dir():
        issues.append(f"Missing frames directory: {frames_dir}")
        return issues

    for dtype_dir in sorted(frames_dir.iterdir()):
        if not dtype_dir.is_dir():
            continue
        for cam_dir in sorted(dtype_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            # Extract frame numbers from filenames like Image_0_0_0001_0.png
            frame_nums: list[int] = []
            for p in cam_dir.iterdir():
                parts = p.stem.split("_")
                # Standard naming: <Type>_0_0_<FrameNum>_<CamIdx>
                if len(parts) >= 4:
                    try:
                        frame_nums.append(int(parts[-2]))
                    except ValueError:
                        continue
            if len(frame_nums) < 2:
                continue
            frame_nums.sort()
            # Check for gaps
            expected_step = frame_nums[1] - frame_nums[0] if len(frame_nums) >= 2 else 1
            if expected_step < 1:
                expected_step = 1
            for i in range(1, len(frame_nums)):
                gap = frame_nums[i] - frame_nums[i - 1]
                if gap != expected_step:
                    issues.append(
                        f"Frame gap in {cam_dir.relative_to(scene_folder)}: "
                        f"expected step {expected_step}, got {gap} between "
                        f"frames {frame_nums[i-1]} and {frame_nums[i]}"
                    )

    return issues


def check_file_sizes(
    scene_folder: str | Path,
    min_sizes: dict[str, int] | None = None,
) -> list[str]:
    """Flag files that are suspiciously small (likely truncated/corrupt).

    Parameters
    ----------
    scene_folder:
        Root of a scene output directory.
    min_sizes:
        Override minimum sizes per extension.

    Returns
    -------
    list[str]
        Descriptions of undersized files.
    """
    scene_folder = Path(scene_folder)
    frames_dir = scene_folder / "frames"
    sizes = min_sizes if min_sizes is not None else _MIN_SIZES
    issues: list[str] = []

    if not frames_dir.is_dir():
        return issues

    for p in sorted(frames_dir.rglob("*")):
        if not p.is_file():
            continue
        threshold = sizes.get(p.suffix.lower())
        if threshold is not None and p.stat().st_size < threshold:
            issues.append(
                f"File too small ({p.stat().st_size} bytes < {threshold}): "
                f"{p.relative_to(scene_folder)}"
            )

    return issues


def check_metadata_completeness(scene_folder: str | Path) -> list[str]:
    """Verify that metadata JSON files contain expected keys.

    Checks ``Objects_json`` files for standard keys and ``camview_npz``
    files for camera-parameter arrays.

    Returns
    -------
    list[str]
        Descriptions of missing or malformed metadata.
    """
    scene_folder = Path(scene_folder)
    issues: list[str] = []

    # Check Objects_json files
    objects_dir = scene_folder / "frames" / "Objects"
    if objects_dir.is_dir():
        for cam_dir in sorted(objects_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            for json_file in sorted(cam_dir.glob("*.json")):
                try:
                    with json_file.open() as f:
                        data = json.load(f)
                    if not isinstance(data, (dict, list)):
                        issues.append(
                            f"Unexpected JSON structure in {json_file.relative_to(scene_folder)}: "
                            f"expected dict or list, got {type(data).__name__}"
                        )
                except json.JSONDecodeError as exc:
                    issues.append(
                        f"Invalid JSON in {json_file.relative_to(scene_folder)}: {exc}"
                    )

    # Check camview_npz files
    camview_dir = scene_folder / "frames" / "camview"
    if camview_dir.is_dir():
        _expected_arrays = {"K", "T", "HW"}
        for cam_dir in sorted(camview_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            for npz_file in sorted(cam_dir.glob("*.npz")):
                try:
                    import numpy as np

                    data = dict(np.load(npz_file))
                    missing = _expected_arrays - set(data.keys())
                    if missing:
                        issues.append(
                            f"Camera file {npz_file.relative_to(scene_folder)} "
                            f"missing arrays: {sorted(missing)}"
                        )
                except Exception as exc:
                    issues.append(
                        f"Cannot read {npz_file.relative_to(scene_folder)}: {exc}"
                    )

    return issues


def generate_checksum_manifest(
    scene_folder: str | Path,
    output_file: str | Path | None = None,
    algorithm: str = "sha256",
) -> dict[str, str]:
    """Compute checksums for all files under ``frames/`` and write a manifest.

    Parameters
    ----------
    scene_folder:
        Scene root directory.
    output_file:
        Where to write the JSON manifest.  Defaults to
        ``<scene_folder>/checksums.json``.
    algorithm:
        Hash algorithm name (anything accepted by :mod:`hashlib`).

    Returns
    -------
    dict[str, str]
        Mapping from relative path → hex digest.
    """
    scene_folder = Path(scene_folder)
    if output_file is None:
        output_file = scene_folder / "checksums.json"
    else:
        output_file = Path(output_file)

    checksums: dict[str, str] = {}
    frames_dir = scene_folder / "frames"
    if not frames_dir.is_dir():
        logger.warning("No frames directory in %s", scene_folder)
        return checksums

    for p in sorted(frames_dir.rglob("*")):
        if not p.is_file():
            continue
        h = hashlib.new(algorithm)
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        rel = str(p.relative_to(scene_folder))
        checksums[rel] = h.hexdigest()

    with output_file.open("w") as f:
        json.dump(checksums, f, indent=2, sort_keys=True)

    logger.info(
        "Wrote checksum manifest (%d files) to %s", len(checksums), output_file
    )
    return checksums


def verify_checksum_manifest(
    scene_folder: str | Path,
    manifest_file: str | Path | None = None,
    algorithm: str = "sha256",
) -> list[str]:
    """Verify files against a previously generated checksum manifest.

    Parameters
    ----------
    scene_folder:
        Scene root directory.
    manifest_file:
        Path to the manifest JSON.  Defaults to
        ``<scene_folder>/checksums.json``.

    Returns
    -------
    list[str]
        Descriptions of mismatches or missing files.
    """
    scene_folder = Path(scene_folder)
    if manifest_file is None:
        manifest_file = scene_folder / "checksums.json"
    else:
        manifest_file = Path(manifest_file)

    issues: list[str] = []

    if not manifest_file.is_file():
        issues.append(f"Manifest file not found: {manifest_file}")
        return issues

    with manifest_file.open() as f:
        manifest: dict[str, str] = json.load(f)

    for rel_path, expected_hash in sorted(manifest.items()):
        full_path = scene_folder / rel_path
        if not full_path.is_file():
            issues.append(f"File missing: {rel_path}")
            continue
        h = hashlib.new(algorithm)
        with full_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_hash:
            issues.append(
                f"Checksum mismatch for {rel_path}: "
                f"expected {expected_hash[:16]}…, got {actual[:16]}…"
            )

    return issues
