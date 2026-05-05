# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Season-specific sun paths and sky parameters that integrate with nishita_lighting()."""

import gin
import numpy as np

from .season_system import SeasonState
from .time_of_day import TimeOfDay, get_tod_sun_params

_SEASON_SUN_MAX = {
    "spring": 55.0,
    "summer": 67.5,
    "autumn": 37.5,
    "winter": 20.0,
}

_SEASON_COLOR_TEMP = {
    "spring": (5000, 6000),
    "summer": (5500, 6500),
    "autumn": (4500, 5000),
    "winter": (4000, 5500),
}

_SEASON_DAYLENGTH = {
    "spring": (12, 14),
    "summer": (14, 16),
    "autumn": (10, 12),
    "winter": (8, 10),
}

_SEASON_TURBIDITY = {
    "spring": {"dust_density": (0.3, 0.8), "air_density": (0.8, 1.1)},
    "summer": {"dust_density": (0.05, 0.3), "air_density": (0.7, 1.0)},
    "autumn": {"dust_density": (0.2, 0.6), "air_density": (0.7, 1.1)},
    "winter": {"dust_density": (0.1, 0.5), "air_density": (0.8, 1.2)},
}


@gin.configurable
def get_seasonal_sun_params(
    season: str = "summer",
    time_of_day: TimeOfDay = TimeOfDay.NOON,
):
    """Return (sun_elevation, sun_azimuth_range, sky_params) for season + TOD.

    sky_params is a dict compatible with nishita_lighting() kwargs including
    seasonal turbidity adjustments.
    """
    sky_params = get_tod_sun_params(tod=time_of_day, season=season)

    turbidity = _SEASON_TURBIDITY.get(season, _SEASON_TURBIDITY["summer"])
    sky_params["dust_density"] = (
        "uniform",
        turbidity["dust_density"][0],
        turbidity["dust_density"][1],
    )
    sky_params["air_density"] = (
        "uniform",
        turbidity["air_density"][0],
        turbidity["air_density"][1],
    )

    return sky_params


def apply_seasonal_lighting(season_state: SeasonState, time_of_day: TimeOfDay):
    """Resolve to a dict of kwargs for nishita_lighting().

    When season_state is None, uses defaults that match existing behavior.
    """
    if season_state is None:
        return get_seasonal_sun_params(season="summer", time_of_day=time_of_day)

    return get_seasonal_sun_params(
        season=season_state.season, time_of_day=time_of_day
    )
