# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Bidirectional bridge between Infinigen syndata and GenesisDroneEnv.

Provides two-way data exchange for RL training in Infinigen-generated 3D
environments loaded into `Genesis World <https://genesis-world.readthedocs.io/>`_
via the `GenesisDroneEnv <https://github.com/KafuuChikai/GenesisDroneEnv>`_
framework.

**Infinigen → Genesis (export):**

* :func:`syndata_to_drone_env_config` converts Infinigen scene configuration
  (curriculum stage, camera, obstacles, randomisation) into GenesisDroneEnv-
  compatible YAML config dicts for the three config files:
  ``genesis_env.yaml``, ``flight.yaml``, and ``rl_env.yaml``.

* :func:`scene_to_drone_entities` converts :class:`GenesisSceneConfig` to
  GenesisDroneEnv entity add code lines for non-drone, non-plane entities.

**Genesis → Infinigen (feedback):**

* :class:`TrainingOutcome` structures the feedback from a GenesisDroneEnv
  training run: success rate, mean reward, crash rate, failure modes, and
  per-metric breakdowns.

* :func:`outcome_to_curriculum_params` maps training outcomes to
  recommended :class:`CurriculumConfig` adjustments — raising difficulty
  when the agent succeeds, lowering it (or holding) when it crashes.

All helpers are pure Python / NumPy — no ``bpy`` or ``genesis`` dependency
at import time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default drone physics timestep (seconds) — GenesisDroneEnv default.
_DEFAULT_DT: float = 0.01

#: Default FPV camera quaternion (FPV 70° look-down, GenesisDroneEnv convention).
_DEFAULT_CAM_QUAT: list[float] = [-0.455, -0.542, 0.542, 0.455]

#: Minimum success rate threshold to advance curriculum difficulty.
_ADVANCE_THRESHOLD: float = 0.7

#: Maximum crash rate threshold — above this, difficulty should regress.
_REGRESS_CRASH_THRESHOLD: float = 0.5

#: Difficulty step size for curriculum progression.
_DIFFICULTY_STEP: float = 0.1


# ---------------------------------------------------------------------------
# DroneEnvConfig — GenesisDroneEnv YAML config equivalent
# ---------------------------------------------------------------------------


@dataclass
class DroneEnvConfig:
    """Configuration bundle matching GenesisDroneEnv's 3-file YAML structure.

    Each field group maps to one of the three YAML config files used by
    GenesisDroneEnv: ``genesis_env.yaml``, ``flight.yaml``, ``rl_env.yaml``.

    This is an Infinigen-side *representation* — not a dependency on
    GenesisDroneEnv itself.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments (vectorised training).
    dt : float
        Physics timestep in seconds.
    cam_res : tuple[int, int]
        FPV camera resolution ``(width, height)``.
    cam_quat : tuple[float, float, float, float]
        Camera quaternion ``(x, y, z, w)`` for FPV mount orientation.
    cam_pos : tuple[float, float, float]
        Camera position offset on the drone body.
    drone_init_pos : tuple[float, float, float]
        Initial drone position.
    max_vis_fps : int
        Maximum visualisation FPS.
    render_cam : bool
        Whether to enable FPV camera rendering.
    show_viewer : bool
        Whether to show the Genesis viewer window.
    use_fpv_camera : bool
        Whether to set up FPV camera on drone.
    znear : float
        Depth camera near plane (metres).
    zfar : float
        Depth camera far plane (metres).
    init_pos_range : dict[str, tuple[float, float]]
        Initial position randomisation ranges: ``{"x": (lo, hi), ...}``.
    map_size : tuple[float, float]
        ``(width, length)`` of the environment map.
    obstacle_entities : list[dict[str, Any]]
        Scene entities to add (Infinigen-exported meshes, boxes, etc.).
    episode_length_s : float
        Maximum episode length in seconds.
    max_episode_length : int
        Maximum episode length in steps.
    reward_scales : dict[str, float]
        Reward function scale factors.
    num_obs : int
        Observation dimension.
    num_actions : int
        Action dimension.
    command_ranges : dict[str, tuple[float, float]]
        Target command randomisation ranges.
    termination_conditions : dict[str, float]
        Termination thresholds (roll, pitch, yaw, ground distance, etc.).
    """

    # -- genesis_env.yaml fields --
    num_envs: int = 4096
    dt: float = _DEFAULT_DT
    cam_res: tuple[int, int] = (256, 256)
    cam_quat: tuple[float, float, float, float] = (-0.455, -0.542, 0.542, 0.455)
    cam_pos: tuple[float, float, float] = (0.02, 0.0, 0.04)
    drone_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.4)
    max_vis_fps: int = 30
    render_cam: bool = False
    show_viewer: bool = False
    use_fpv_camera: bool = True
    znear: float = 0.1
    zfar: float = 5.0
    init_pos_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.6, 0.61)}
    )
    map_size: tuple[float, float] = (3.5, 3.5)
    obstacle_entities: list[dict[str, Any]] = field(default_factory=list)

    # -- rl_env.yaml fields --
    episode_length_s: float = 15.0
    max_episode_length: int = 1500
    reward_scales: dict[str, float] = field(
        default_factory=lambda: {
            "target": 10.0,
            "smooth": -0.0001,
            "yaw": 0.01,
            "angular": -0.0002,
            "crash": -10.0,
        }
    )
    num_obs: int = 23
    num_actions: int = 4
    command_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-1.2, 1.2),
            "y": (-1.2, 1.2),
            "z": (0.6, 1.0),
        }
    )
    termination_conditions: dict[str, float] = field(
        default_factory=lambda: {
            "roll_deg": 180.0,
            "pitch_deg": 180.0,
            "ground_m": 0.1,
            "x_m": 5.0,
            "y_m": 5.0,
            "z_m": 1.2,
        }
    )

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            msg = f"num_envs must be >= 1, got {self.num_envs}"
            raise ValueError(msg)
        if self.dt <= 0:
            msg = f"dt must be positive, got {self.dt}"
            raise ValueError(msg)
        if self.cam_res[0] < 1 or self.cam_res[1] < 1:
            msg = f"cam_res must be positive, got {self.cam_res}"
            raise ValueError(msg)

    def to_genesis_env_yaml(self) -> dict[str, Any]:
        """Return a dict matching GenesisDroneEnv ``genesis_env.yaml`` format.

        Returns
        -------
        dict[str, Any]
            YAML-serialisable config for ``genesis_env.yaml``.
        """
        d: dict[str, Any] = {
            "num_envs": self.num_envs,
            "drone_num": 1,
            "dt": self.dt,
            "max_vis_FPS": self.max_vis_fps,
            "use_FPV_camera": self.use_fpv_camera,
            "cam_quat": list(self.cam_quat),
            "cam_pos": list(self.cam_pos),
            "cam_res": list(self.cam_res),
            "drone_init_pos": list(self.drone_init_pos),
            "vis_waypoints": True,
            "viewer_follow_drone": False,
            "show_cam_GUI": False,
            "fixed_init_pos": False,
            "znear": self.znear,
            "zfar": self.zfar,
            "load_map": False,
            "use_rc": False,
            "show_viewer": self.show_viewer,
            "render_cam": self.render_cam,
            "controller": "angle",
            "min_dis": 0.8,
            "map_width": self.map_size[0],
            "map_length": self.map_size[1],
            "init_x_range": list(self.init_pos_range.get("x", (-0.05, 0.05))),
            "init_y_range": list(self.init_pos_range.get("y", (-0.05, 0.05))),
            "init_z_range": list(self.init_pos_range.get("z", (0.6, 0.61))),
            "target_thr": 0.1,
        }
        return d

    def to_rl_env_yaml(self) -> dict[str, Any]:
        """Return a dict matching GenesisDroneEnv ``rl_env.yaml`` format.

        Returns
        -------
        dict[str, Any]
            YAML-serialisable config for ``rl_env.yaml``.
        """
        cmd_cfg: dict[str, Any] = {}
        for axis, key in [("x", "pos_x_range"), ("y", "pos_y_range"), ("z", "pos_z_range")]:
            rng = self.command_ranges.get(axis, (-1.0, 1.0))
            cmd_cfg[key] = list(rng)

        task: dict[str, Any] = {
            "reward_scales": dict(self.reward_scales),
            "episode_length_s": self.episode_length_s,
            "max_episode_length": self.max_episode_length,
            "num_actions": self.num_actions,
            "num_commands": 3,
            "num_obs": self.num_obs,
            "clip_actions": 1.0,
            "target_thr": 0.1,
            "command_cfg": cmd_cfg,
            "termination_if_roll_greater_than": self.termination_conditions.get("roll_deg", 180.0),
            "termination_if_pitch_greater_than": self.termination_conditions.get("pitch_deg", 180.0),
            "termination_if_close_to_ground": self.termination_conditions.get("ground_m", 0.1),
            "termination_if_x_greater_than": self.termination_conditions.get("x_m", 5.0),
            "termination_if_y_greater_than": self.termination_conditions.get("y_m", 5.0),
            "termination_if_z_greater_than": self.termination_conditions.get("z_m", 1.2),
        }
        return {"task": task}

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict for persistence."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_json(path: str | Path) -> DroneEnvConfig:
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text())
        # Normalize tuple fields from lists
        for key in ("cam_res", "cam_quat", "cam_pos", "drone_init_pos", "map_size"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        # Normalize nested tuple dicts
        for key in ("init_pos_range", "command_ranges"):
            if key in data:
                data[key] = {
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in data[key].items()
                }
        return DroneEnvConfig(**data)


# ---------------------------------------------------------------------------
# Training outcome — structured feedback from GenesisDroneEnv
# ---------------------------------------------------------------------------


@dataclass
class TrainingOutcome:
    """Structured feedback from a GenesisDroneEnv training run.

    Captures the key metrics that drive curriculum progression decisions:
    how well the agent performs, where it fails, and what conditions
    caused failures.

    Parameters
    ----------
    success_rate : float
        Fraction of episodes reaching the target (0.0–1.0).
    mean_reward : float
        Mean cumulative reward across episodes.
    crash_rate : float
        Fraction of episodes ending in a crash termination.
    timeout_rate : float
        Fraction of episodes ending due to time limit.
    mean_episode_length : float
        Mean episode length in steps.
    reward_breakdown : dict[str, float]
        Per-component mean rewards (matching ``reward_scales`` keys).
    failure_modes : dict[str, float]
        Fraction of failures by cause: ``"roll"``, ``"pitch"``,
        ``"out_of_bounds"``, ``"ground_collision"``, ``"timeout"``.
    mean_position_error : float
        Mean final position error (metres).
    difficulty : float
        The difficulty level this outcome was recorded at (0.0–1.0).
    num_episodes : int
        Total episodes in this evaluation.
    metadata : dict[str, Any]
        Arbitrary additional metadata (scene ID, seed, etc.).
    """

    success_rate: float = 0.0
    mean_reward: float = 0.0
    crash_rate: float = 0.0
    timeout_rate: float = 0.0
    mean_episode_length: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    failure_modes: dict[str, float] = field(default_factory=dict)
    mean_position_error: float = 0.0
    difficulty: float = 0.0
    num_episodes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_rate <= 1.0:
            msg = f"success_rate must be in [0, 1], got {self.success_rate}"
            raise ValueError(msg)
        if not 0.0 <= self.crash_rate <= 1.0:
            msg = f"crash_rate must be in [0, 1], got {self.crash_rate}"
            raise ValueError(msg)
        if not 0.0 <= self.timeout_rate <= 1.0:
            msg = f"timeout_rate must be in [0, 1], got {self.timeout_rate}"
            raise ValueError(msg)
        if not 0.0 <= self.difficulty <= 1.0:
            msg = f"difficulty must be in [0, 1], got {self.difficulty}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write outcome to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_json(path: str | Path) -> TrainingOutcome:
        """Load outcome from a JSON file."""
        data = json.loads(Path(path).read_text())
        return TrainingOutcome(**data)


# ---------------------------------------------------------------------------
# Infinigen → Genesis (export direction)
# ---------------------------------------------------------------------------


def syndata_to_drone_env_config(
    *,
    curriculum: Any | None = None,
    drone_camera: Any | None = None,
    camera_rig: Any | None = None,
    observation: Any | None = None,
    randomiser: Any | None = None,
    frame_metadata: Any | None = None,
    scene_config: Any | None = None,
    num_envs: int = 4096,
    dt: float = _DEFAULT_DT,
) -> DroneEnvConfig:
    """Convert Infinigen syndata components to a DroneEnvConfig.

    Maps curriculum complexity, camera settings, obstacle layout, and
    domain randomisation parameters to GenesisDroneEnv's config format.

    Parameters
    ----------
    curriculum : CurriculumConfig | None
        Curriculum stage — controls scene complexity.
    drone_camera : DroneCamera | None
        Camera optics (FoV, aspect ratio).
    camera_rig : CameraRigConfig | None
        Camera rig layout.
    observation : ObservationConfig | None
        Observation space config.
    randomiser : DomainRandomiser | None
        Domain randomisation parameters.
    frame_metadata : FrameMetadata | None
        Per-frame metadata with obstacles.
    scene_config : GenesisSceneConfig | None
        Pre-built Genesis scene config for entity extraction.
    num_envs : int
        Number of parallel environments.
    dt : float
        Physics timestep.

    Returns
    -------
    DroneEnvConfig
        GenesisDroneEnv-compatible configuration.
    """
    cfg = DroneEnvConfig(num_envs=num_envs, dt=dt)

    # ---- Camera from syndata ----
    if drone_camera is not None:
        # Map DroneCamera FoV to cam_res based on aspect ratio
        ar = getattr(drone_camera, "aspect_ratio", 4 / 3)
        h = cfg.cam_res[1]
        w = max(1, int(round(h * ar)))
        cfg = DroneEnvConfig(
            **{**asdict(cfg), "cam_res": (w, h)},
        )

    if observation is not None:
        cfg = DroneEnvConfig(
            **{**asdict(cfg), "render_cam": True, "use_fpv_camera": True},
        )

    # ---- Obstacle entities from metadata ----
    obstacle_entities: list[dict[str, Any]] = []
    if frame_metadata is not None:
        for i, obs in enumerate(getattr(frame_metadata, "obstacles", [])):
            cx, cy, cz = obs.center
            dx, dy, dz = obs.extent
            obstacle_entities.append({
                "name": f"obstacle_{i}_{obs.label}",
                "morph_type": "Box",
                "pos": (cx, cy, cz),
                "size": (dx * 2, dy * 2, dz * 2),
                "fixed": True,
                "collision": True,
            })
        # Derive init position ranges from metadata spatial extent
        if frame_metadata.obstacles:
            all_cx = [o.center[0] for o in frame_metadata.obstacles]
            all_cy = [o.center[1] for o in frame_metadata.obstacles]
            margin = 0.5
            if all_cx and all_cy:
                x_range = (min(all_cx) - margin, max(all_cx) + margin)
                y_range = (min(all_cy) - margin, max(all_cy) + margin)
                cfg = DroneEnvConfig(
                    **{
                        **asdict(cfg),
                        "init_pos_range": {"x": x_range, "y": y_range, "z": (0.6, 0.61)},
                        "map_size": (
                            max(3.5, abs(x_range[1] - x_range[0]) + 2 * margin),
                            max(3.5, abs(y_range[1] - y_range[0]) + 2 * margin),
                        ),
                    },
                )

    # ---- Entities from GenesisSceneConfig ----
    if scene_config is not None:
        for ent in getattr(scene_config, "entities", []):
            # Skip ground plane and drone — those are added by GenesisDroneEnv
            if ent.morph_type in ("Plane",):
                continue
            obstacle_entities.append(ent.to_dict())

    cfg = DroneEnvConfig(**{**asdict(cfg), "obstacle_entities": obstacle_entities})

    # ---- Difficulty-driven reward scaling ----
    if curriculum is not None:
        progress = getattr(curriculum, "progress", 0.0)
        # Scale crash penalty with difficulty (harsher at higher levels)
        scales = dict(cfg.reward_scales)
        scales["crash"] = -10.0 * (1.0 + progress)
        cfg = DroneEnvConfig(**{**asdict(cfg), "reward_scales": scales})

    # ---- Map domain randomisation to init ranges ----
    if randomiser is not None:
        sample = randomiser.sample()
        wind = sample.get("wind_strength", 0.0)
        # Wider init ranges at higher difficulty to match randomised conditions
        difficulty = getattr(randomiser, "difficulty", 0.5)
        spread = 0.05 + difficulty * 0.5
        pos_range = cfg.init_pos_range
        pos_range["x"] = (-spread, spread)
        pos_range["y"] = (-spread, spread)
        cfg = DroneEnvConfig(**{**asdict(cfg), "init_pos_range": pos_range})

    return cfg


def scene_to_drone_entities(
    scene_config: Any,
) -> list[str]:
    """Generate GenesisDroneEnv entity-add code lines from a GenesisSceneConfig.

    Produces Python code snippets compatible with the Genesis_env
    ``__init__`` pattern where entities are added via ``scene.add_entity()``.

    Parameters
    ----------
    scene_config : GenesisSceneConfig
        Scene configuration with entities.

    Returns
    -------
    list[str]
        Python code lines for adding entities.
    """
    lines: list[str] = []
    for ent in scene_config.entities:
        if ent.morph_type in ("Plane",):
            continue  # Ground plane handled by Genesis_env
        var = ent.name.replace("-", "_").replace(" ", "_").replace(".", "_")

        morph_args: list[str] = []
        if ent.file_path:
            morph_args.append(f'file="{ent.file_path}"')
        if ent.pos != (0.0, 0.0, 0.0):
            morph_args.append(f"pos={ent.pos!r}")
        if ent.is_fixed:
            morph_args.append("fixed=True")
        if not ent.collision:
            morph_args.append("collision=False")
        for k, v in ent.extra.items():
            morph_args.append(f"{k}={v!r}")

        morph_str = ", ".join(morph_args)
        lines.append(
            f"self.{var} = self.scene.add_entity("
            f"gs.morphs.{ent.morph_type}({morph_str}))"
        )
    return lines


# ---------------------------------------------------------------------------
# Genesis → Infinigen (feedback direction)
# ---------------------------------------------------------------------------


def outcome_to_curriculum_params(
    outcome: TrainingOutcome,
    current_stage: int,
    total_stages: int,
    *,
    advance_threshold: float = _ADVANCE_THRESHOLD,
    regress_crash_threshold: float = _REGRESS_CRASH_THRESHOLD,
) -> dict[str, Any]:
    """Map GenesisDroneEnv training outcomes to curriculum adjustment parameters.

    Implements a simple curriculum progression policy:

    * **Advance** (increase stage): success rate ≥ *advance_threshold* AND
      crash rate < *regress_crash_threshold*.
    * **Regress** (decrease stage): crash rate ≥ *regress_crash_threshold*.
    * **Hold**: Otherwise.

    Also returns recommended adjustments to scene generation parameters
    based on which failure modes dominate.

    Parameters
    ----------
    outcome : TrainingOutcome
        Training feedback from a GenesisDroneEnv run.
    current_stage : int
        Current curriculum stage (0-indexed).
    total_stages : int
        Total number of curriculum stages.
    advance_threshold : float
        Minimum success rate to advance.
    regress_crash_threshold : float
        Maximum crash rate before regression.

    Returns
    -------
    dict[str, Any]
        Curriculum adjustment parameters:

        - ``"recommended_stage"`` — int: next stage to use.
        - ``"action"`` — str: ``"advance"``, ``"regress"``, or ``"hold"``.
        - ``"difficulty"`` — float: recommended difficulty in [0, 1].
        - ``"scene_adjustments"`` — dict with hints for scene generation:
          ``"increase_obstacles"``, ``"widen_corridors"``,
          ``"reduce_clutter"``, ``"add_edge_cases"``.

    Raises
    ------
    ValueError
        If *current_stage* or *total_stages* are invalid.
    """
    if total_stages < 1:
        msg = f"total_stages must be >= 1, got {total_stages}"
        raise ValueError(msg)
    if not 0 <= current_stage < total_stages:
        msg = f"current_stage must be in [0, {total_stages}), got {current_stage}"
        raise ValueError(msg)

    # Determine progression action
    action: str
    next_stage: int

    if (
        outcome.success_rate >= advance_threshold
        and outcome.crash_rate < regress_crash_threshold
    ):
        action = "advance"
        next_stage = min(current_stage + 1, total_stages - 1)
    elif outcome.crash_rate >= regress_crash_threshold:
        action = "regress"
        next_stage = max(current_stage - 1, 0)
    else:
        action = "hold"
        next_stage = current_stage

    difficulty = next_stage / max(1, total_stages - 1)

    # Analyse failure modes for scene generation hints
    fm = outcome.failure_modes
    scene_adjustments: dict[str, bool] = {
        # If ground collisions dominate → widen corridors / raise min altitude
        "widen_corridors": fm.get("ground_collision", 0.0) > 0.3,
        # If out-of-bounds dominate → the map may be too cluttered
        "reduce_clutter": fm.get("out_of_bounds", 0.0) > 0.3,
        # If success rate is very high → add more obstacles for challenge
        "increase_obstacles": outcome.success_rate > 0.9,
        # If crash rate from roll/pitch → add edge cases with tight spaces
        "add_edge_cases": (
            fm.get("roll", 0.0) + fm.get("pitch", 0.0) > 0.3
        ),
    }

    return {
        "recommended_stage": next_stage,
        "action": action,
        "difficulty": difficulty,
        "scene_adjustments": scene_adjustments,
    }


def apply_curriculum_adjustment(
    base_config: DroneEnvConfig,
    adjustment: dict[str, Any],
) -> DroneEnvConfig:
    """Apply curriculum adjustment hints to a DroneEnvConfig.

    Modifies the base config according to the ``"scene_adjustments"``
    from :func:`outcome_to_curriculum_params`.

    Parameters
    ----------
    base_config : DroneEnvConfig
        Starting configuration.
    adjustment : dict[str, Any]
        Adjustment parameters from :func:`outcome_to_curriculum_params`.

    Returns
    -------
    DroneEnvConfig
        Updated configuration.
    """
    d = asdict(base_config)
    hints = adjustment.get("scene_adjustments", {})
    difficulty = adjustment.get("difficulty", 0.0)

    # Widen corridors → increase map size, wider init ranges
    if hints.get("widen_corridors", False):
        w, l = d["map_size"]
        d["map_size"] = (w * 1.2, l * 1.2)

    # Reduce clutter → tighten command ranges to keep agent in bounds
    if hints.get("reduce_clutter", False):
        for axis in ("x", "y"):
            lo, hi = d["command_ranges"].get(axis, (-1.2, 1.2))
            shrink = 0.8
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * shrink
            d["command_ranges"][axis] = (mid - half, mid + half)

    # Increase obstacles → harsher crash penalty to incentivise avoidance
    if hints.get("increase_obstacles", False):
        d["reward_scales"]["crash"] = d["reward_scales"].get("crash", -10.0) * 1.5

    # Scale init position spread with difficulty
    spread = 0.05 + difficulty * 0.5
    d["init_pos_range"]["x"] = (-spread, spread)
    d["init_pos_range"]["y"] = (-spread, spread)

    return DroneEnvConfig(**d)
