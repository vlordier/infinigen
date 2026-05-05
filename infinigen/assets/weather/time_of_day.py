# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Time-of-day presets and sun path computation for Nishita sky lighting."""

from enum import Enum

import gin
import numpy as np


class TimeOfDay(Enum):
    DAWN = "dawn"
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    NIGHT = "night"


_TOD_RANGES = {
    TimeOfDay.DAWN: {
        "sun_elevation": (-6.0, 0.0),
        "sun_rotation": (0, 360),
        "strength": (0.05, 0.12),
        "dust_density": (0.3, 1.0),
        "air_density": (0.8, 1.2),
        "ozone_density": (1.0, 3.0),
    },
    TimeOfDay.MORNING: {
        "sun_elevation": (5.0, 25.0),
        "sun_rotation": (0, 360),
        "strength": (0.12, 0.20),
        "dust_density": (0.1, 0.5),
        "air_density": (0.7, 1.1),
        "ozone_density": (0.5, 2.0),
    },
    TimeOfDay.NOON: {
        "sun_elevation": "max",  # resolved per season
        "sun_rotation": (0, 360),
        "strength": (0.15, 0.25),
        "dust_density": (0.1, 0.3),
        "air_density": (0.7, 1.0),
        "ozone_density": (0.1, 1.0),
    },
    TimeOfDay.AFTERNOON: {
        "sun_elevation": (25.0, 60.0),
        "sun_rotation": (0, 360),
        "strength": (0.10, 0.20),
        "dust_density": (0.1, 0.4),
        "air_density": (0.7, 1.1),
        "ozone_density": (0.5, 2.0),
    },
    TimeOfDay.DUSK: {
        "sun_elevation": (-6.0, 0.0),
        "sun_rotation": (0, 360),
        "strength": (0.05, 0.12),
        "dust_density": (0.3, 1.0),
        "air_density": (0.8, 1.2),
        "ozone_density": (1.0, 3.0),
    },
    TimeOfDay.NIGHT: {
        "sun_elevation": (-90.0, -6.0),
        "sun_rotation": (0, 360),
        "strength": (0.0, 0.03),
        "dust_density": (0.1, 0.5),
        "air_density": (0.8, 1.2),
        "ozone_density": (0.5, 2.0),
    },
}

_SEASONAL_SUN_MAX = {
    "spring": 55.0,
    "summer": 67.5,
    "autumn": 37.5,
    "winter": 20.0,
}


@gin.configurable
def get_tod_sun_params(
    tod: TimeOfDay,
    season: str = "summer",
):
    """Return sun elevation/azimuth ranges and sky params for a given TOD and season.

    Returns a dict compatible with nishita_lighting() kwargs:
    sun_elevation, sun_rotation, strength, dust_density, air_density, ozone_density
    """
    ranges = _TOD_RANGES[tod].copy()

    if ranges["sun_elevation"] == "max":
        max_el = _SEASONAL_SUN_MAX.get(season, 67.5)
        ranges["sun_elevation"] = (max_el - 5, max_el)

    return {
        "sun_elevation": (
            "uniform",
            ranges["sun_elevation"][0],
            ranges["sun_elevation"][1],
        ),
        "sun_rotation": (
            "uniform",
            np.radians(ranges["sun_rotation"][0]),
            np.radians(ranges["sun_rotation"][1]),
        ),
        "strength": (
            "uniform",
            ranges["strength"][0],
            ranges["strength"][1],
        ),
        "dust_density": (
            "clip_gaussian",
            (ranges["dust_density"][0] + ranges["dust_density"][1]) / 2,
            (ranges["dust_density"][1] - ranges["dust_density"][0]) / 3,
            ranges["dust_density"][0],
            ranges["dust_density"][1],
        ),
        "air_density": (
            "clip_gaussian",
            (ranges["air_density"][0] + ranges["air_density"][1]) / 2,
            (ranges["air_density"][1] - ranges["air_density"][0]) / 3,
            ranges["air_density"][0],
            ranges["air_density"][1],
        ),
        "ozone_density": (
            "clip_gaussian",
            (ranges["ozone_density"][0] + ranges["ozone_density"][1]) / 2,
            (ranges["ozone_density"][1] - ranges["ozone_density"][0]) / 3,
            ranges["ozone_density"][0],
            ranges["ozone_density"][1],
        ),
    }
