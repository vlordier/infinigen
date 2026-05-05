# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Central season state object that flows through the entire scene generation pipeline."""

from dataclasses import dataclass

import gin


@gin.configurable
@dataclass
class SeasonState:
    season: str = "summer"
    temperature: float = 28.0
    precipitation: float = 0.2
    snow_cover: float = 0.0
    vegetation_phase: float = 1.0
    day_length_hours: float = 15.0
    ground_is_frozen: bool = False
    water_is_frozen: bool = False


@gin.configurable
def get_or_create_season_state(season_state=None, season_name=None):
    """Return season_state if provided, otherwise create from season_name or gin defaults.

    This is the primary entry point for the pipeline.
    If both are None, returns None (backwards compatible — no seasonal effects).
    """
    if season_state is not None:
        return season_state
    if season_name is not None:
        return create_season_state(season_name)
    return None


def create_season_state(season_name: str) -> SeasonState:
    """Return a SeasonState with sensible defaults for the named season."""
    if season_name == "spring":
        return SeasonState(
            season="spring",
            temperature=12.0,
            precipitation=0.5,
            snow_cover=0.0,
            vegetation_phase=0.6,
            day_length_hours=13.0,
            ground_is_frozen=False,
            water_is_frozen=False,
        )
    elif season_name == "summer":
        return SeasonState(
            season="summer",
            temperature=28.0,
            precipitation=0.2,
            snow_cover=0.0,
            vegetation_phase=1.0,
            day_length_hours=15.0,
            ground_is_frozen=False,
            water_is_frozen=False,
        )
    elif season_name == "autumn":
        return SeasonState(
            season="autumn",
            temperature=10.0,
            precipitation=0.4,
            snow_cover=0.0,
            vegetation_phase=0.3,
            day_length_hours=11.0,
            ground_is_frozen=False,
            water_is_frozen=False,
        )
    elif season_name == "winter":
        return SeasonState(
            season="winter",
            temperature=-5.0,
            precipitation=0.3,
            snow_cover=0.7,
            vegetation_phase=0.0,
            day_length_hours=9.0,
            ground_is_frozen=True,
            water_is_frozen=True,
        )
    else:
        raise ValueError(f"Unknown season: {season_name}. Use spring/summer/autumn/winter.")
