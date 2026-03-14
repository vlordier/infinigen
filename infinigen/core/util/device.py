# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Unified device selection for PyTorch workloads.

Supports CUDA, MPS (Apple Silicon M1/M2/M3/M4), and CPU fallback.
"""

import logging
import os
import platform

logger = logging.getLogger(__name__)


def get_torch_device(prefer: str | None = None):
    """Return the best available ``torch.device``.

    Parameters
    ----------
    prefer : str | None
        If given, try this device first (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).
        Falls back automatically if the requested backend is unavailable.

    Returns
    -------
    torch.device
    """
    import torch

    # Allow environment variable override
    env_device = os.environ.get("INFINIGEN_TORCH_DEVICE")
    if env_device:
        prefer = env_device

    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            logger.info("Using CUDA device")
            return torch.device("cuda")
        if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS device (Apple Silicon)")
            return torch.device("mps")
        if prefer == "cpu":
            logger.info("Using CPU device (explicitly requested)")
            return torch.device("cpu")

    # Auto-detect best available device
    if torch.cuda.is_available():
        logger.info("Auto-detected CUDA device")
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-detected MPS device (Apple Silicon)")
        return torch.device("mps")

    logger.info("Using CPU device (no GPU backend available)")
    return torch.device("cpu")


def is_apple_silicon() -> bool:
    """Return *True* if running on Apple Silicon (arm64 macOS)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
