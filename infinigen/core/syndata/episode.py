# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Episode configuration for Infinigen rendering pipeline.

Defines temporal structure for rendering: frame range, FPS, and trajectory
type.  These map directly to ``execute_tasks.frame_range`` and
``execute_tasks.fps`` gin bindings for the Infinigen pipeline.

All helpers are pure Python — no ``bpy`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Trajectory modes that Infinigen supports via camera animation policies
TRAJECTORY_STATIC = "static"  # single viewpoint
TRAJECTORY_RANDOM_WALK = "random_walk"  # Brownian motion
TRAJECTORY_ORBIT = "orbit"  # circular path around a point
TRAJECTORY_RRT = "rrt"  # rapidly-exploring random tree (obstacle-aware)
TRAJECTORY_FOLLOW = "follow"  # follow a target object

VALID_TRAJECTORIES = frozenset({
    TRAJECTORY_STATIC,
    TRAJECTORY_RANDOM_WALK,
    TRAJECTORY_ORBIT,
    TRAJECTORY_RRT,
    TRAJECTORY_FOLLOW,
})


@dataclass(frozen=True)
class EpisodeConfig:
    """Temporal structure of a single RL episode.

    Parameters
    ----------
    num_frames : int
        Number of frames per episode.  For single-frame training
        (e.g. object detection), use 1.
    fps : int
        Frames per second.  Lower FPS = more scene change per frame
        (useful for early curriculum stages).  Infinigen default is 24.
    start_frame : int
        First frame index (Blender uses 1-based indexing).
    trajectory : str
        Camera trajectory type.  Must be one of ``VALID_TRAJECTORIES``.
    """

    num_frames: int = 1
    fps: int = 24
    start_frame: int = 1
    trajectory: str = TRAJECTORY_STATIC

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            msg = f"num_frames must be >= 1, got {self.num_frames}"
            raise ValueError(msg)
        if self.fps < 1 or self.fps > 120:
            msg = f"fps must be in [1, 120], got {self.fps}"
            raise ValueError(msg)
        if self.start_frame < 1:
            msg = f"start_frame must be >= 1, got {self.start_frame}"
            raise ValueError(msg)
        if self.trajectory not in VALID_TRAJECTORIES:
            msg = f"trajectory must be one of {sorted(VALID_TRAJECTORIES)}, got {self.trajectory!r}"
            raise ValueError(msg)

    @property
    def end_frame(self) -> int:
        """Last frame index (inclusive, matching Blender convention)."""
        return self.start_frame + self.num_frames - 1

    @property
    def frame_range(self) -> tuple[int, int]:
        """``(start, end)`` tuple for ``execute_tasks.frame_range``."""
        return (self.start_frame, self.end_frame)

    @property
    def duration_seconds(self) -> float:
        """Episode duration in seconds."""
        return self.num_frames / self.fps

    def gin_overrides(self) -> dict[str, Any]:
        """Return gin bindings for episode timing."""
        return {
            "execute_tasks.frame_range": list(self.frame_range),
            "execute_tasks.fps": self.fps,
        }

    @staticmethod
    def single_frame() -> EpisodeConfig:
        """One-frame episode for static observation tasks."""
        return EpisodeConfig(num_frames=1, trajectory=TRAJECTORY_STATIC)

    @staticmethod
    def short_trajectory(num_frames: int = 30, fps: int = 10) -> EpisodeConfig:
        """Short trajectory for early curriculum stages.

        Lower FPS means more visual change per frame, which gives
        the agent more diverse training data per render.
        """
        return EpisodeConfig(
            num_frames=num_frames,
            fps=fps,
            trajectory=TRAJECTORY_RANDOM_WALK,
        )

    @staticmethod
    def navigation_episode(num_frames: int = 120, fps: int = 24) -> EpisodeConfig:
        """Full navigation episode with obstacle-aware trajectory."""
        return EpisodeConfig(
            num_frames=num_frames,
            fps=fps,
            trajectory=TRAJECTORY_RRT,
        )
