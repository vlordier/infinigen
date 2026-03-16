# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""OcMesher backend protocol helpers.

This module defines lightweight runtime validation and capability collection for
pluggable OcMesher backends (default Python/C++ backend or custom backends,
including Rust-backed implementations).
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Mapping, cast

import numpy as np

SELF_TEST_SCHEMA_KEYS = (
    "ok",
    "backend",
    "backend_version",
    "test_bounds",
    "vertices",
    "faces",
    "capabilities",
    "error",
)


def normalize_backend_capabilities(raw_caps: dict[str, Any]) -> dict[str, Any]:
    caps = dict(raw_caps)

    alias_map = {
        "cuda": "supports_cuda",
        "mps": "supports_mps",
        "cpu": "supports_cpu",
        "max_batch_size": "max_batch",
        "max_batch_size_hint": "max_batch",
        "dtype": "preferred_dtype",
    }
    for old_key, new_key in alias_map.items():
        if old_key in caps and new_key not in caps:
            caps[new_key] = caps[old_key]

    for key in ("supports_cuda", "supports_mps", "supports_cpu"):
        if key in caps:
            caps[key] = bool(caps[key])

    if "max_batch" in caps:
        try:
            caps["max_batch"] = int(caps["max_batch"])
        except Exception:
            pass

    return caps


def validate_ocmesher_backend_class(backend_cls: type, class_path: str) -> None:
    """Validate the minimum constructor/call contract for a backend class."""
    if not callable(backend_cls):
        raise TypeError(f"Configured OcMesher backend is not callable: {class_path}")

    try:
        params = list(signature(backend_cls).parameters.values())
    except (TypeError, ValueError):
        # Some extension classes do not expose signatures; allow them.
        return

    user_params = [p for p in params if p.name != "self"]
    required_positional = [
        p
        for p in user_params
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is Parameter.empty
    ]

    # Expected constructor shape is at least: (cameras, bounds, ...)
    if len(required_positional) < 2:
        raise TypeError(
            f"OcMesher backend {class_path} has incompatible constructor; "
            "expected at least two required positional args (cameras, bounds)"
        )


def collect_ocmesher_backend_capabilities(instance: Any) -> dict[str, Any]:
    """Collect backend capabilities if exposed by the backend instance.

    Backends may expose a ``capabilities() -> dict`` method. Missing method is
    treated as valid and yields only basic metadata.
    """
    caps: dict[str, Any] = {
        "backend_name": instance.__class__.__qualname__,
        "backend_module": instance.__class__.__module__,
    }

    capabilities_fn = getattr(instance, "capabilities", None)
    if not callable(capabilities_fn):
        capabilities_fn = getattr(instance, "get_capabilities", None)
    if not callable(capabilities_fn):
        return caps

    try:
        backend_caps = capabilities_fn()
    except Exception as exc:  # pragma: no cover - defensive runtime path
        caps["capabilities_error"] = str(exc)
        return caps

    if isinstance(backend_caps, dict):
        caps.update(normalize_backend_capabilities(cast(dict[str, Any], backend_caps)))
    else:
        caps["capabilities_error"] = (
            "capabilities() did not return a dict; ignoring backend-reported capabilities"
        )

    return caps


def collect_ocmesher_backend_class_capabilities(backend_cls: type) -> dict[str, Any]:
    """Collect static/backend-class capability hints when available."""
    caps: dict[str, Any] = {}

    for method_name in ("capabilities", "get_capabilities"):
        if caps:
            break
        method = getattr(backend_cls, method_name, None)
        if not callable(method):
            continue
        try:
            raw_caps = method()
        except TypeError:
            # Instance-bound methods require construction and are handled elsewhere.
            continue
        except Exception:
            continue
        if isinstance(raw_caps, dict):
            caps = normalize_backend_capabilities(cast(dict[str, Any], raw_caps))

    if not caps:
        for attr_name in ("DEFAULT_CAPABILITIES", "CAPABILITIES"):
            raw_caps = getattr(backend_cls, attr_name, None)
            if isinstance(raw_caps, dict):
                caps = normalize_backend_capabilities(cast(dict[str, Any], raw_caps))
                break

    return caps


def resolve_ocmesher_runtime_kwargs(
    backend_cls: type,
    kwargs: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    device_hint: str | None = None,
) -> dict[str, Any]:
    """Resolve runtime kwargs using env overrides, class capabilities, and hints."""
    resolved: dict[str, Any] = dict(kwargs)
    caps = collect_ocmesher_backend_class_capabilities(backend_cls)
    env_map = env if env is not None else {}

    def _class_accepts_kwarg(kwarg_name: str) -> bool:
        try:
            params = signature(backend_cls.__init__).parameters.values()
        except (TypeError, ValueError):
            return True
        if any(p.kind == Parameter.VAR_KEYWORD for p in params):
            return True
        return any(p.name == kwarg_name for p in params)

    def _filter_supported_kwargs(values: dict[str, Any]) -> dict[str, Any]:
        if not values:
            return {}
        try:
            params = signature(backend_cls.__init__).parameters.values()
        except (TypeError, ValueError):
            return dict(values)
        if any(p.kind == Parameter.VAR_KEYWORD for p in params):
            return dict(values)
        allowed = {p.name for p in params}
        return {k: v for k, v in values.items() if k in allowed}

    requested_device = env_map.get("INFINIGEN_OCMESHER_DEVICE") or device_hint
    if (
        requested_device is not None
        and "device" not in resolved
        and _class_accepts_kwarg("device")
    ):
        resolved["device"] = requested_device

    requested_dtype = env_map.get("INFINIGEN_OCMESHER_DTYPE")
    if (
        requested_dtype is not None
        and "dtype" not in resolved
        and _class_accepts_kwarg("dtype")
    ):
        resolved["dtype"] = requested_dtype
    elif (
        "dtype" not in resolved
        and _class_accepts_kwarg("dtype")
        and isinstance(caps.get("preferred_dtype"), str)
    ):
        resolved["dtype"] = caps["preferred_dtype"]

    requested_batch = env_map.get("INFINIGEN_OCMESHER_BATCH")
    if requested_batch is not None:
        for key in ("max_batch", "batch_size", "sdf_batch_size"):
            if key not in resolved and _class_accepts_kwarg(key):
                try:
                    resolved[key] = int(requested_batch)
                except ValueError:
                    pass
                break
    elif isinstance(caps.get("max_batch"), int) and caps["max_batch"] > 0:
        for key in ("max_batch", "batch_size", "sdf_batch_size"):
            if key not in resolved and _class_accepts_kwarg(key):
                resolved[key] = caps["max_batch"]
                break

    stream_policy = env_map.get("INFINIGEN_OCMESHER_STREAM_POLICY")
    if (
        stream_policy is not None
        and "stream_policy" not in resolved
        and _class_accepts_kwarg("stream_policy")
    ):
        resolved["stream_policy"] = stream_policy
    elif "stream_policy" not in resolved and _class_accepts_kwarg("stream_policy"):
        if isinstance(caps.get("default_stream_policy"), str):
            resolved["stream_policy"] = caps["default_stream_policy"]
        elif bool(caps.get("supports_async", False)):
            resolved["stream_policy"] = "auto"

    return _filter_supported_kwargs(resolved)


def serialize_ocmesher_self_test_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize self-test output into a stable, JSON-friendly schema."""
    payload: dict[str, Any] = {
        "ok": bool(result.get("ok", False)),
        "backend": result.get("backend"),
        "backend_version": result.get("backend_version"),
        "test_bounds": None,
        "vertices": result.get("vertices"),
        "faces": result.get("faces"),
        "capabilities": {},
        "error": result.get("error"),
    }

    test_bounds = result.get("test_bounds")
    if isinstance(test_bounds, (tuple, list)) and len(test_bounds) == 6:
        payload["test_bounds"] = [float(v) for v in test_bounds]

    caps = result.get("capabilities")
    if isinstance(caps, dict):
        payload["capabilities"] = normalize_backend_capabilities(cast(dict[str, Any], caps))

    return payload


def normalize_ocmesher_result(
    result: Any,
    class_path: str,
    expect_single_mesh: bool = False,
) -> tuple[list[Any], list[np.ndarray]]:
    """Normalize backend return value to ``(meshes, in_view_tags)`` lists.

    Expected backend return shape is a 2-tuple: ``(meshes, in_view_tags)``.
    ``meshes`` and ``in_view_tags`` may be either single values or sequences.
    """
    if not isinstance(result, (tuple, list)):
        raise TypeError(
            f"OcMesher backend {class_path} returned invalid payload shape; "
            "expected (meshes, in_view_tags)"
        )

    result_items: list[Any] = list(cast(tuple[Any, ...] | list[Any], result))
    if len(result_items) != 2:
        raise TypeError(
            f"OcMesher backend {class_path} returned invalid payload shape; "
            "expected (meshes, in_view_tags)"
        )

    meshes_raw, tags_raw = result_items[0], result_items[1]

    meshes: list[Any] = (
        list(cast(tuple[Any, ...] | list[Any], meshes_raw))
        if isinstance(meshes_raw, (tuple, list))
        else [meshes_raw]
    )
    tags: list[Any] = (
        list(cast(tuple[Any, ...] | list[Any], tags_raw))
        if isinstance(tags_raw, (tuple, list))
        else [tags_raw]
    )

    if expect_single_mesh and len(meshes) != 1:
        raise ValueError(
            f"OcMesher backend {class_path} expected a single mesh, got {len(meshes)}"
        )

    if len(meshes) != len(tags):
        raise ValueError(
            f"OcMesher backend {class_path} returned {len(meshes)} meshes "
            f"but {len(tags)} in_view tags"
        )

    for idx, mesh in enumerate(meshes):
        if not hasattr(mesh, "vertex_attributes"):
            raise TypeError(
                f"OcMesher backend {class_path} mesh[{idx}] is missing "
                "expected attribute 'vertex_attributes'"
            )

    normalized_tags: list[np.ndarray] = []
    for idx, tag in enumerate(tags):
        arr = np.asarray(tag, dtype=bool)
        if arr.ndim != 1:
            raise ValueError(
                f"OcMesher backend {class_path} in_view tag[{idx}] must be 1-D, got {arr.ndim}-D"
            )
        normalized_tags.append(arr)

    return meshes, normalized_tags
