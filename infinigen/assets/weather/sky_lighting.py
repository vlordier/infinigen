# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Season-aware sky lighting wrapper for Nishita sky."""

from infinigen.assets.lighting.sky_lighting import nishita_lighting


def apply_seasonal_sky(season_state=None, time_of_day=None):
    """Apply Nishita sky with season and TOD parameters."""
    from .season_system import SeasonState
    from .time_of_day import TimeOfDay, get_tod_sun_params
    from .seasonal_lighting import get_seasonal_sun_params

    if season_state is None:
        season_state = SeasonState()
    if time_of_day is None:
        time_of_day = TimeOfDay.NOON

    sun_params = get_seasonal_sun_params(season_state.season, time_of_day)
    return nishita_lighting(**sun_params)
