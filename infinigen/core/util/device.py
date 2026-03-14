# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Unified device selection and performance tuning for PyTorch workloads.

Supports CUDA, MPS (Apple Silicon M1/M2/M3/M4), and CPU fallback.
Provides device-specific helpers for optimal dtype, batch size, and thread
configuration to maximise throughput on each target.
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


# ---------------------------------------------------------------------------
# Device capability helpers – dtype, batch size, and threading
# ---------------------------------------------------------------------------


def optimal_dtype(device=None):
    """Return the fastest floating-point dtype for *device*.

    * CUDA  → ``torch.float16``  (Tensor Cores on Volta+, 2× throughput)
    * MPS   → ``torch.float32``  (MPS has limited float16 support)
    * CPU   → ``torch.float32``  (best vectorised width on AVX2/512)

    Override via the ``INFINIGEN_TORCH_DTYPE`` environment variable
    (values: ``float16``, ``bfloat16``, ``float32``, ``float64``).
    """
    import torch

    env = os.environ.get("INFINIGEN_TORCH_DTYPE")
    if env:
        _map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if env in _map:
            return _map[env]
        logger.warning("Unknown INFINIGEN_TORCH_DTYPE=%s, using auto", env)

    if device is None:
        device = get_torch_device()

    dtype = device.type
    if dtype == "cuda":
        return torch.float16
    # MPS and CPU work best with float32
    return torch.float32


def optimal_batch_size(device=None, element_bytes: int = 4) -> int:
    """Heuristic batch size that keeps GPU utilisation high without OOM.

    * CUDA  → uses ~60 % of free VRAM divided by *element_bytes*.
    * MPS   → conservative 256 K elements (unified memory, shared with OS).
    * CPU   → 1 M elements (fits comfortably in L3 cache on most CPUs).
    """
    if device is None:
        device = get_torch_device()

    dtype = device.type
    if dtype == "cuda":
        import torch

        free, _total = torch.cuda.mem_get_info()
        return max(1, int(free * 0.6) // max(element_bytes, 1))
    if dtype == "mps":
        return 256_000
    # CPU
    return 1_000_000


def optimal_num_threads(device=None) -> int:
    """Return the recommended number of worker threads.

    * CUDA  → 1 (kernel launch is async; more host threads add overhead).
    * MPS   → 2 (one for feeding the GPU, one for post-processing).
    * CPU   → ``os.cpu_count()`` capped at 8 to avoid over-subscription.

    Override via the ``INFINIGEN_NUM_THREADS`` environment variable.
    """
    env = os.environ.get("INFINIGEN_NUM_THREADS")
    if env:
        return max(1, int(env))

    if device is None:
        device = get_torch_device()

    dtype = device.type
    if dtype == "cuda":
        return 1
    if dtype == "mps":
        return 2
    return min(os.cpu_count() or 4, 8)


def device_capabilities(device=None) -> dict:
    """Return a summary dict describing *device* capabilities.

    Keys always present: ``backend``, ``optimal_dtype``, ``batch_size``,
    ``num_threads``.

    For CUDA, additional keys:  ``name``, ``compute_capability``,
    ``total_memory_mb``, ``free_memory_mb``.
    """
    if device is None:
        device = get_torch_device()

    info: dict = {
        "backend": device.type,
        "optimal_dtype": str(optimal_dtype(device)),
        "batch_size": optimal_batch_size(device),
        "num_threads": optimal_num_threads(device),
    }

    if device.type == "cuda":
        import torch

        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info.update(
            {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": total // (1024 * 1024),
                "free_memory_mb": free // (1024 * 1024),
            }
        )

    return info
