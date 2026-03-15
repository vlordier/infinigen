# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Gin configuration validation utilities.

Infinigen uses `gin-config <https://github.com/google/gin-config>`_ for
configuring its procedural-generation pipeline.  Gin is flexible but
provides no built-in schema checking — a typo in a parameter name is
silently ignored, and type mismatches surface as cryptic errors deep
inside the pipeline.

This module provides lightweight pre-flight checks that catch common
configuration mistakes *before* any expensive Blender/rendering work
starts.

Usage::

    from infinigen.core.util.config_validation import (
        validate_gin_config,
        GinValidationError,
    )

    issues = validate_gin_config(
        required_keys=["execute_tasks.frame_range", "execute_tasks.camera_id"],
        type_rules={
            "execute_tasks.frame_range": list,
            "execute_tasks.camera_id": list,
        },
        range_rules={
            "execute_tasks.fps": (1, 240),
        },
    )
    if issues:
        raise GinValidationError(issues)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class GinValidationError(Exception):
    """Raised when gin configuration fails validation.

    Attributes
    ----------
    issues : list[str]
        Human-readable descriptions of each detected problem.
    """

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        bullet_list = "\n  • ".join(self.issues)
        super().__init__(f"Gin configuration has {len(self.issues)} issue(s):\n  • {bullet_list}")


def _query_gin_value(key: str):
    """Attempt to resolve *key* from the current gin operative config.

    Returns ``(True, value)`` if found, or ``(False, None)`` otherwise.
    """
    import gin

    # gin stores bindings internally as scope/selector.param
    # The public API for querying a specific binding is via gin.query_parameter
    try:
        value = gin.query_parameter(key)
        return True, value
    except ValueError:
        return False, None


def validate_gin_config(
    required_keys: list[str] | None = None,
    type_rules: dict[str, type | tuple[type, ...]] | None = None,
    range_rules: dict[str, tuple[float | int | None, float | int | None]] | None = None,
    custom_rules: dict[str, Callable[[object], bool]] | None = None,
) -> list[str]:
    """Validate the currently-bound gin configuration.

    Parameters
    ----------
    required_keys:
        Keys that must be present (e.g. ``"execute_tasks.frame_range"``).
    type_rules:
        Mapping from key → expected type(s). Values resolved from gin are
        checked with ``isinstance``.
    range_rules:
        Mapping from key → ``(min, max)``. ``None`` means unbounded on
        that side.  Only checked when the key is present *and* numeric.
    custom_rules:
        Mapping from key → predicate ``(value) -> bool``.  A ``False``
        return is treated as a validation failure.

    Returns
    -------
    list[str]
        Descriptions of detected problems (empty list means valid).
    """
    issues: list[str] = []
    resolved_values: dict[str, object] = {}

    # 1. Required-key checks
    for key in required_keys or []:
        found, value = _query_gin_value(key)
        if not found:
            issues.append(f"Required key {key!r} is not bound in gin config")
        else:
            resolved_values[key] = value

    # Pre-resolve keys needed by later checks
    for mapping in (type_rules, range_rules, custom_rules):
        for key in mapping or {}:
            if key not in resolved_values:
                found, value = _query_gin_value(key)
                if found:
                    resolved_values[key] = value

    # 2. Type checks
    for key, expected in (type_rules or {}).items():
        if key in resolved_values:
            value = resolved_values[key]
            if not isinstance(value, expected):
                issues.append(
                    f"Key {key!r} has type {type(value).__name__}, expected {expected}"
                )

    # 3. Range checks
    for key, (lo, hi) in (range_rules or {}).items():
        if key in resolved_values:
            value = resolved_values[key]
            if isinstance(value, (int, float)):
                if lo is not None and value < lo:
                    issues.append(
                        f"Key {key!r} value {value} is below minimum {lo}"
                    )
                if hi is not None and value > hi:
                    issues.append(
                        f"Key {key!r} value {value} exceeds maximum {hi}"
                    )

    # 4. Custom predicate checks
    for key, predicate in (custom_rules or {}).items():
        if key in resolved_values:
            value = resolved_values[key]
            try:
                if not predicate(value):
                    issues.append(
                        f"Key {key!r} failed custom validation (value={value!r})"
                    )
            except Exception as exc:
                issues.append(
                    f"Key {key!r} custom validator raised {type(exc).__name__}: {exc}"
                )

    if issues:
        logger.warning("Gin validation found %d issue(s)", len(issues))
        for issue in issues:
            logger.warning("  • %s", issue)

    return issues
