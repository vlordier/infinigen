# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Quantitative scene complexity metrics for synthetic data generation.

Computes measurable statistics about generated scenes that can be used to:
  - Verify that a scene matches its target complexity level.
  - Provide auxiliary features to RL agents alongside rendered images.
  - Track dataset diversity across a generated corpus.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SceneMetrics:
    """Container for quantitative scene statistics.

    All counts default to ``0``; populate via :func:`compute_metrics` or
    manually in tests.
    """

    object_count: int = 0
    unique_material_count: int = 0
    total_polygon_count: int = 0
    total_vertex_count: int = 0
    scatter_instance_count: int = 0
    light_count: int = 0
    max_object_depth: int = 0

    # Derived aggregate score (computed lazily)
    _complexity_score: float | None = field(default=None, repr=False)

    @property
    def complexity_score(self) -> float:
        """Return a normalised scalar summarising scene complexity.

        The score is a weighted sum of the individual metrics, each normalised
        by a domain-specific reference maximum so that every component
        contributes on a comparable 0-1 scale.

        Returns
        -------
        float
            Non-negative complexity score.
        """
        if self._complexity_score is not None:
            return self._complexity_score
        self._complexity_score = self._compute_score()
        return self._complexity_score

    def _compute_score(self) -> float:
        """Weighted sum of normalised metrics."""
        # Reference maximums – rough order-of-magnitude values for a
        # fully-featured Infinigen nature scene.
        weights = {
            "object_count": (0.20, 500),
            "unique_material_count": (0.15, 50),
            "total_polygon_count": (0.25, 5_000_000),
            "total_vertex_count": (0.10, 2_500_000),
            "scatter_instance_count": (0.15, 100_000),
            "light_count": (0.05, 10),
            "max_object_depth": (0.10, 10),
        }
        score = 0.0
        for attr, (w, ref_max) in weights.items():
            val = getattr(self, attr, 0)
            score += w * min(val / max(ref_max, 1), 1.0)
        return score

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (excluding private fields)."""
        d = asdict(self)
        d.pop("_complexity_score", None)
        d["complexity_score"] = self.complexity_score
        return d


@dataclass
class DatasetDiversityStats:
    """Aggregate statistics across a collection of scene metrics.

    Useful for monitoring how diverse the generated dataset is.
    """

    scene_count: int = 0
    mean_object_count: float = 0.0
    std_object_count: float = 0.0
    mean_polygon_count: float = 0.0
    std_polygon_count: float = 0.0
    mean_material_count: float = 0.0
    std_material_count: float = 0.0
    mean_complexity_score: float = 0.0
    std_complexity_score: float = 0.0
    min_complexity_score: float = 0.0
    max_complexity_score: float = 0.0

    @classmethod
    def from_metrics_list(cls, metrics: list[SceneMetrics]) -> DatasetDiversityStats:
        """Compute diversity stats from a list of per-scene metrics."""
        if not metrics:
            return cls()

        n = len(metrics)
        objs = [m.object_count for m in metrics]
        polys = [m.total_polygon_count for m in metrics]
        mats = [m.unique_material_count for m in metrics]
        scores = [m.complexity_score for m in metrics]

        def _mean(xs: list[float]) -> float:
            return sum(xs) / max(len(xs), 1)

        def _std(xs: list[float]) -> float:
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

        return cls(
            scene_count=n,
            mean_object_count=_mean(objs),
            std_object_count=_std(objs),
            mean_polygon_count=_mean(polys),
            std_polygon_count=_std(polys),
            mean_material_count=_mean(mats),
            std_material_count=_std(mats),
            mean_complexity_score=_mean(scores),
            std_complexity_score=_std(scores),
            min_complexity_score=min(scores),
            max_complexity_score=max(scores),
        )
