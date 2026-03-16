# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Tensor/array interop helpers for backend integration.

This module centralizes zero-copy conversion helpers (when possible) between
NumPy and PyTorch, including DLPack entry points for Rust/Python backend glue.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np


def torch_available() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except Exception:
        return False


def to_torch_tensor(
    array: np.ndarray,
    *,
    device: str | None = None,
    dtype: Any | None = None,
) -> Any:
    """Convert numpy array to torch tensor.

    Uses ``torch.from_numpy`` (zero-copy on CPU) and optional device move.
    """
    import torch

    t = cast(Any, torch.from_numpy(array))
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def to_numpy_array(tensor: Any) -> np.ndarray[Any, Any]:
    """Convert tensor-like object to numpy array.

    For torch tensors on GPU, this performs a host transfer.
    """
    if isinstance(tensor, np.ndarray):
        return cast(np.ndarray[Any, Any], tensor)

    if hasattr(tensor, "detach") and hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        return tensor.detach().cpu().numpy()

    return np.asarray(tensor)


def torch_tensor_to_dlpack(tensor: Any) -> Any:
    """Export tensor to DLPack capsule using ``__dlpack__`` protocol."""
    if hasattr(tensor, "__dlpack__"):
        return tensor.__dlpack__()
    raise TypeError("Tensor does not implement __dlpack__")


def torch_tensor_from_dlpack(dlpack_capsule: Any) -> Any:
    """Import DLPack capsule into torch tensor."""
    import torch.utils.dlpack

    return torch.utils.dlpack.from_dlpack(dlpack_capsule)


def maybe_dlpack_from_tensor(tensor: Any) -> Any | None:
    """Return DLPack capsule when producer supports ``__dlpack__``.

    This is useful for Rust backends exposing ``__dlpack__`` from Python bridge
    objects. Returns ``None`` when unsupported.
    """
    if hasattr(tensor, "__dlpack__"):
        return tensor.__dlpack__()
    return None
