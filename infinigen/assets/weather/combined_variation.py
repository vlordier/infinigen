# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Systematic generation across the season x time-of-day matrix."""

import itertools

import gin
import numpy as np

from .time_of_day import TimeOfDay


@gin.configurable
def generate_variation_matrix(
    seasons: list[str] | None = None,
    times_of_day: list[str] | None = None,
    scene_count: int = 10,
) -> list[tuple[str, str]]:
    """Return list of (season, tod) pairs to generate.

    If scene_count < full matrix size, samples randomly.
    If scene_count >= full matrix, cycles through all combinations.
    """
    if seasons is None:
        seasons = ["spring", "summer", "autumn", "winter"]
    if times_of_day is None:
        times_of_day = ["dawn", "morning", "noon", "afternoon", "dusk", "night"]

    full_combinations = list(itertools.product(seasons, times_of_day))
    full_count = len(full_combinations)

    if scene_count >= full_count:
        cycles = scene_count // full_count
        remainder = scene_count % full_count
        result = full_combinations * cycles
        idxs = np.random.choice(full_count, size=remainder, replace=False)
        result += [full_combinations[i] for i in idxs]
    else:
        idxs = np.random.choice(full_count, size=scene_count, replace=False)
        result = [full_combinations[i] for i in idxs]

    return result
