# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Gin-configurable season presets."""

import gin


@gin.configurable
def spring_preset():
    return {
        "season": "spring",
        "temperature": 15.0,
        "precipitation": 0.5,
        "snow_cover": 0.0,
        "vegetation_phase": 0.7,
        "ground_is_frozen": False,
        "water_is_frozen": False,
    }


@gin.configurable
def summer_preset():
    return {
        "season": "summer",
        "temperature": 28.0,
        "precipitation": 0.3,
        "snow_cover": 0.0,
        "vegetation_phase": 1.0,
        "ground_is_frozen": False,
        "water_is_frozen": False,
    }


@gin.configurable
def autumn_preset():
    return {
        "season": "autumn",
        "temperature": 12.0,
        "precipitation": 0.4,
        "snow_cover": 0.1,
        "vegetation_phase": 0.4,
        "ground_is_frozen": False,
        "water_is_frozen": False,
    }


@gin.configurable
def winter_preset():
    return {
        "season": "winter",
        "temperature": -5.0,
        "precipitation": 0.6,
        "snow_cover": 0.8,
        "vegetation_phase": 0.1,
        "ground_is_frozen": True,
        "water_is_frozen": True,
    }
