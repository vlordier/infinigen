# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Gin-configurable time-of-day presets."""

import gin


@gin.configurable
def dawn_params():
    return {
        "sun_elevation": -3.0,
        "sun_rotation": 90.0,
        "sun_strength": 0.3,
        "sky_params": {"dust": 1.5, "air": 1.0, "ozone": 1.0},
    }


@gin.configurable
def morning_params():
    return {
        "sun_elevation": 15.0,
        "sun_rotation": 110.0,
        "sun_strength": 2.0,
        "sky_params": {"dust": 1.0, "air": 1.0, "ozone": 1.0},
    }


@gin.configurable
def noon_params():
    return {
        "sun_elevation": 60.0,
        "sun_rotation": 180.0,
        "sun_strength": 5.0,
        "sky_params": {"dust": 0.5, "air": 1.0, "ozone": 1.0},
    }


@gin.configurable
def afternoon_params():
    return {
        "sun_elevation": 30.0,
        "sun_rotation": 250.0,
        "sun_strength": 3.0,
        "sky_params": {"dust": 1.0, "air": 1.0, "ozone": 1.0},
    }


@gin.configurable
def dusk_params():
    return {
        "sun_elevation": -3.0,
        "sun_rotation": 270.0,
        "sun_strength": 0.3,
        "sky_params": {"dust": 1.5, "air": 1.0, "ozone": 1.0},
    }


@gin.configurable
def night_params():
    return {
        "sun_elevation": -30.0,
        "sun_rotation": 0.0,
        "sun_strength": 0.01,
        "sky_params": {"dust": 0.5, "air": 1.0, "ozone": 1.0},
    }
