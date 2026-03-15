# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Genesis World integration — bridge Infinigen scene data to Genesis physics engine.

Provides pure-Python converters and configuration helpers to load
Infinigen-generated 3D scenes into `Genesis World
<https://genesis-world.readthedocs.io/>`_ for physical simulation,
differentiable rendering, and RL agent training.

Supported import paths:

* **OBJ / PLY / STL** meshes → ``gs.morphs.Mesh``
* **MJCF** articulated assets → ``gs.morphs.MJCF``
* **URDF** robot descriptions → ``gs.morphs.URDF``

Key features:

* :func:`scene_manifest_from_dir` discovers all importable assets in an
  Infinigen export directory and produces a :class:`GenesisSceneManifest`.
* :class:`GenesisSceneConfig` aggregates entities, cameras, lights, and
  renderer settings into a single configuration object.
* :func:`to_genesis_script` generates a self-contained Python script that
  reconstructs the Infinigen scene inside Genesis for physics simulation.

All helpers are pure Python — no ``bpy`` or ``genesis`` dependency at import
time, so CI and pre-flight tooling stay lightweight.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum sun elevation (degrees) to avoid degenerate light placement.
_MIN_ELEVATION_DEG: float = 5.0
#: Maximum sun elevation (degrees).
_MAX_ELEVATION_DEG: float = 85.0
#: Multiplier mapping Infinigen sun intensity to Genesis HDR light color.
_GENESIS_LIGHT_INTENSITY_SCALE: float = 10.0
#: Default number of simulation steps in generated scripts.
_DEFAULT_SIMULATION_STEPS: int = 1000

# ---------------------------------------------------------------------------
# Supported file extensions (lowered) → Genesis morph type
# ---------------------------------------------------------------------------

_MESH_EXTENSIONS: dict[str, str] = {
    ".obj": "Mesh",
    ".ply": "Mesh",
    ".stl": "Mesh",
    ".glb": "Mesh",
    ".gltf": "Mesh",
}

_SCENE_EXTENSIONS: dict[str, str] = {
    ".xml": "MJCF",
    ".urdf": "URDF",
    ".usd": "USD",
    ".usda": "USD",
    ".usdc": "USD",
}

_ALL_EXTENSIONS: dict[str, str] = {**_MESH_EXTENSIONS, **_SCENE_EXTENSIONS}


def _morph_type_for(path: str | Path) -> str | None:
    """Return the Genesis morph type string for a file, or *None* if unknown."""
    ext = Path(path).suffix.lower()
    return _ALL_EXTENSIONS.get(ext)


# ---------------------------------------------------------------------------
# Entity configuration
# ---------------------------------------------------------------------------


@dataclass
class GenesisEntityConfig:
    """Descriptor for one entity to load into a Genesis scene.

    Parameters
    ----------
    name : str
        Human-readable entity name.
    file_path : str
        Path to the asset file (OBJ, PLY, STL, MJCF, URDF, USD, glTF).
    morph_type : str
        Genesis morph class name: ``"Mesh"``, ``"MJCF"``, ``"URDF"``,
        ``"USD"``, ``"Box"``, ``"Plane"``.
    pos : tuple[float, float, float]
        World position ``(x, y, z)``.
    euler : tuple[float, float, float]
        Euler rotation ``(rx, ry, rz)`` in **degrees**, applied in XYZ
        order (Genesis convention).
    scale : float | tuple[float, float, float]
        Uniform or per-axis scale factor.
    is_fixed : bool
        If *True* the entity is immovable (infinite mass).
    surface : str
        Genesis surface preset name (e.g. ``"Rough"``, ``"Aluminium"``).
        Empty string means use the asset's own materials.
    surface_color : tuple[float, float, float] | None
        RGB color override (each channel 0-1).
    collision : bool
        Enable collision geometry.
    extra : dict[str, Any]
        Additional morph/surface keyword arguments passed through verbatim.
    """

    name: str = "entity"
    file_path: str = ""
    morph_type: str = "Mesh"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float | tuple[float, float, float] = 1.0
    is_fixed: bool = False
    surface: str = ""
    surface_color: tuple[float, float, float] | None = None
    collision: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    _VALID_MORPH_TYPES = frozenset({
        "Mesh", "MJCF", "URDF", "USD", "Box", "Plane", "Sphere", "Cylinder",
        "Terrain",
    })

    def __post_init__(self) -> None:
        if self.morph_type not in self._VALID_MORPH_TYPES:
            msg = (
                f"morph_type must be one of {sorted(self._VALID_MORPH_TYPES)}, "
                f"got {self.morph_type!r}"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @staticmethod
    def from_file(
        file_path: str | Path,
        *,
        name: str = "",
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        **kwargs: Any,
    ) -> GenesisEntityConfig:
        """Create an entity config by auto-detecting the morph type from file extension.

        Raises
        ------
        ValueError
            If the file extension is not recognised.
        """
        p = Path(file_path)
        mt = _morph_type_for(p)
        if mt is None:
            msg = f"Unsupported asset file extension: {p.suffix!r}"
            raise ValueError(msg)
        return GenesisEntityConfig(
            name=name or p.stem,
            file_path=str(p),
            morph_type=mt,
            pos=pos,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------


@dataclass
class GenesisCamera:
    """Camera configuration matching Genesis ``gs.Camera`` parameters.

    Parameters
    ----------
    name : str
        Camera identifier.
    res : tuple[int, int]
        Resolution ``(width, height)`` in pixels.
    pos : tuple[float, float, float]
        World position ``(x, y, z)``.
    lookat : tuple[float, float, float]
        Look-at target ``(x, y, z)``.
    fov : float
        Vertical field-of-view in degrees.
    near : float
        Near clipping plane (metres).
    far : float
        Far clipping plane (metres).
    spp : int
        Samples per pixel (ray tracer quality).
    """

    name: str = "cam_0"
    res: tuple[int, int] = (640, 480)
    pos: tuple[float, float, float] = (3.0, 0.0, 2.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 60.0
    near: float = 0.01
    far: float = 100.0
    spp: int = 256

    def __post_init__(self) -> None:
        if self.res[0] < 1 or self.res[1] < 1:
            msg = f"Resolution must be positive, got {self.res}"
            raise ValueError(msg)
        if not 1.0 <= self.fov <= 179.0:
            msg = f"fov must be in [1, 179], got {self.fov}"
            raise ValueError(msg)
        if self.near <= 0 or self.far <= self.near:
            msg = f"Invalid clip planes: near={self.near}, far={self.far}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Light configuration
# ---------------------------------------------------------------------------


@dataclass
class GenesisLight:
    """Point or area light for Genesis scenes.

    Parameters
    ----------
    pos : tuple[float, float, float]
        World position.
    color : tuple[float, float, float]
        Light color (intensity-weighted RGB, Genesis uses HDR values).
    radius : float
        Light radius (area light size; 0 for point light).
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 10.0)
    color: tuple[float, float, float] = (10.0, 10.0, 10.0)
    radius: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (matches Genesis light spec)."""
        return {"pos": self.pos, "color": self.color, "radius": self.radius}


# ---------------------------------------------------------------------------
# Full scene configuration
# ---------------------------------------------------------------------------


@dataclass
class GenesisSceneConfig:
    """Complete Genesis scene description assembled from Infinigen outputs.

    Parameters
    ----------
    entities : list[GenesisEntityConfig]
        All scene entities (meshes, robots, terrain).
    cameras : list[GenesisCamera]
        Camera(s) in the scene.
    lights : list[GenesisLight]
        Light sources.
    renderer : str
        ``"RayTracer"`` or ``"Rasterizer"``.
    backend : str
        Genesis compute backend: ``"cpu"``, ``"cuda"``, ``"metal"``.
    gravity : tuple[float, float, float]
        Gravity vector (m/s²).  Default: Earth.
    dt : float
        Simulation timestep (seconds).
    """

    entities: list[GenesisEntityConfig] = field(default_factory=list)
    cameras: list[GenesisCamera] = field(default_factory=list)
    lights: list[GenesisLight] = field(default_factory=list)
    renderer: str = "RayTracer"
    backend: str = "cuda"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 0.01

    def __post_init__(self) -> None:
        if self.renderer not in ("RayTracer", "Rasterizer"):
            msg = f"renderer must be 'RayTracer' or 'Rasterizer', got {self.renderer!r}"
            raise ValueError(msg)
        if self.dt <= 0:
            msg = f"dt must be positive, got {self.dt}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "cameras": [c.to_dict() for c in self.cameras],
            "lights": [l.to_dict() for l in self.lights],
            "renderer": self.renderer,
            "backend": self.backend,
            "gravity": self.gravity,
            "dt": self.dt,
        }

    def save_json(self, path: str | Path) -> None:
        """Write scene config to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_json(path: str | Path) -> GenesisSceneConfig:
        """Load scene config from a JSON file."""
        data = json.loads(Path(path).read_text())
        return GenesisSceneConfig(
            entities=[GenesisEntityConfig(**e) for e in data.get("entities", [])],
            cameras=[GenesisCamera(**c) for c in data.get("cameras", [])],
            lights=[GenesisLight(**l) for l in data.get("lights", [])],
            renderer=data.get("renderer", "RayTracer"),
            backend=data.get("backend", "cuda"),
            gravity=tuple(data.get("gravity", (0.0, 0.0, -9.81))),
            dt=data.get("dt", 0.01),
        )


# ---------------------------------------------------------------------------
# Scene manifest — discover assets in an Infinigen export directory
# ---------------------------------------------------------------------------


@dataclass
class GenesisSceneManifest:
    """Inventory of importable assets discovered in an export directory.

    Attributes
    ----------
    root_dir : str
        Absolute path to the export directory.
    mesh_files : list[str]
        Paths to mesh files (OBJ, PLY, STL, glTF) relative to *root_dir*.
    mjcf_files : list[str]
        Paths to MJCF ``.xml`` files.
    urdf_files : list[str]
        Paths to URDF files.
    usd_files : list[str]
        Paths to USD/USDA/USDC files.
    metadata_files : list[str]
        Paths to JSON metadata files.
    """

    root_dir: str = ""
    mesh_files: list[str] = field(default_factory=list)
    mjcf_files: list[str] = field(default_factory=list)
    urdf_files: list[str] = field(default_factory=list)
    usd_files: list[str] = field(default_factory=list)
    metadata_files: list[str] = field(default_factory=list)

    @property
    def total_assets(self) -> int:
        """Total number of importable asset files."""
        return (
            len(self.mesh_files)
            + len(self.mjcf_files)
            + len(self.urdf_files)
            + len(self.usd_files)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write manifest to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_json(path: str | Path) -> GenesisSceneManifest:
        """Load manifest from a JSON file."""
        data = json.loads(Path(path).read_text())
        return GenesisSceneManifest(**data)


def scene_manifest_from_dir(export_dir: str | Path) -> GenesisSceneManifest:
    """Discover all importable assets in an Infinigen export directory.

    Recursively scans *export_dir* for files with recognised extensions
    and classifies them by format.

    Parameters
    ----------
    export_dir : str | Path
        Root of the Infinigen export tree (e.g. ``output/scene_001/``).

    Returns
    -------
    GenesisSceneManifest
        Populated manifest with relative paths.

    Raises
    ------
    FileNotFoundError
        If *export_dir* does not exist.
    """
    root = Path(export_dir)
    if not root.is_dir():
        msg = f"Export directory does not exist: {root}"
        raise FileNotFoundError(msg)

    manifest = GenesisSceneManifest(root_dir=str(root.resolve()))
    mesh_exts = frozenset(_MESH_EXTENSIONS)
    usd_exts = frozenset({".usd", ".usda", ".usdc"})

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        ext = p.suffix.lower()
        if ext in mesh_exts:
            manifest.mesh_files.append(rel)
        elif ext == ".xml":
            manifest.mjcf_files.append(rel)
        elif ext == ".urdf":
            manifest.urdf_files.append(rel)
        elif ext in usd_exts:
            manifest.usd_files.append(rel)
        elif ext == ".json":
            manifest.metadata_files.append(rel)
    return manifest


# ---------------------------------------------------------------------------
# Converters: syndata types → Genesis configuration
# ---------------------------------------------------------------------------


def camera_from_syndata(
    drone_cam: Any,
    rig: Any,
    *,
    resolution: tuple[int, int] = (640, 480),
) -> list[GenesisCamera]:
    """Convert syndata :class:`DroneCamera` + :class:`CameraRigConfig` to Genesis cameras.

    Parameters
    ----------
    drone_cam : DroneCamera
        Camera optics (FoV, aspect ratio).
    rig : CameraRigConfig
        Camera rig layout (positions, stereo baseline, multi-drone count).
    resolution : tuple[int, int]
        Render resolution ``(width, height)``.

    Returns
    -------
    list[GenesisCamera]
        One :class:`GenesisCamera` per effective camera in the rig,
        repeated for each rig instance (``n_rigs``).
    """
    cameras: list[GenesisCamera] = []
    effective = rig.effective_cameras
    rig_idx = 0
    for _rig_i in range(rig.n_rigs):
        for cam_dict in effective:
            loc = cam_dict.get("loc", (0, 0, 0))
            # Genesis camera looks down -Z by default; we set lookat to (0,0,0)
            # and use pos from the rig
            cameras.append(GenesisCamera(
                name=f"drone_{rig_idx}",
                res=resolution,
                pos=loc,
                lookat=(0.0, 0.0, 0.0),
                fov=drone_cam.fov_deg,
            ))
            rig_idx += 1
    return cameras


def observation_to_render_kwargs(obs: Any) -> dict[str, bool]:
    """Convert syndata :class:`ObservationConfig` to Genesis ``cam.render()`` kwargs.

    Parameters
    ----------
    obs : ObservationConfig
        Observation space configuration.

    Returns
    -------
    dict[str, bool]
        Keyword arguments for ``camera.render()``, e.g.
        ``{"rgb": True, "depth": True, "segmentation": True, "normal": False}``.
    """
    # Import locally to avoid circular deps at module level
    from infinigen.core.syndata.observation import (
        PASS_DEPTH,
        PASS_NORMAL,
        PASS_OBJECT_INDEX,
    )

    return {
        "rgb": obs.include_rgb,
        "depth": PASS_DEPTH in obs.passes,
        "segmentation": PASS_OBJECT_INDEX in obs.passes,
        "normal": PASS_NORMAL in obs.passes,
    }


def randomisation_to_genesis_lights(
    randomiser: Any,
    *,
    base_height: float = 10.0,
) -> list[GenesisLight]:
    """Convert syndata :class:`DomainRandomiser` ranges to Genesis light configs.

    Samples one concrete configuration from the randomiser's current
    difficulty using its seed, then maps lighting parameters to Genesis
    light format.

    Parameters
    ----------
    randomiser : DomainRandomiser
        Domain randomisation source (uses ``sample()``).
    base_height : float
        Default height for the primary light source.

    Returns
    -------
    list[GenesisLight]
        Light(s) configured from the sampled parameters.
    """
    sample = randomiser.sample()
    intensity = sample.get("sun_intensity", 1.0)
    elevation_deg = sample.get("sun_elevation", 45.0)

    # Place the sun light at an angle matching the sampled elevation
    elev_rad = math.radians(max(_MIN_ELEVATION_DEG, min(_MAX_ELEVATION_DEG, elevation_deg)))
    distance = base_height / max(math.sin(elev_rad), 0.01)
    lx = distance * math.cos(elev_rad)
    lz = base_height

    # Map intensity to Genesis HDR light color
    color_val = _GENESIS_LIGHT_INTENSITY_SCALE * intensity

    return [
        GenesisLight(
            pos=(lx, 0.0, lz),
            color=(color_val, color_val, color_val),
            radius=3.0,
        ),
    ]


def metadata_to_entities(
    frame_meta: Any,
    *,
    obstacle_surface: str = "Rough",
    obstacle_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> list[GenesisEntityConfig]:
    """Convert :class:`FrameMetadata` obstacle bboxes to Genesis box entities.

    This is useful for reconstructing an approximate collision scene from
    Infinigen metadata without the full mesh data — e.g., for fast
    physics-only simulation of drone navigation.

    Parameters
    ----------
    frame_meta : FrameMetadata
        Per-frame metadata with obstacle bounding boxes.
    obstacle_surface : str
        Genesis surface preset for obstacles.
    obstacle_color : tuple[float, float, float]
        Default obstacle color (RGB 0-1).

    Returns
    -------
    list[GenesisEntityConfig]
        One ``Box`` entity per obstacle bounding box.
    """
    entities: list[GenesisEntityConfig] = []
    for i, obs in enumerate(frame_meta.obstacles):
        # BBox3D.extent stores half-extents (dx, dy, dz); Genesis Box
        # ``size`` parameter expects full dimensions, so we double them.
        cx, cy, cz = obs.center
        dx, dy, dz = obs.extent
        entities.append(GenesisEntityConfig(
            name=f"obstacle_{i}_{obs.label}",
            morph_type="Box",
            pos=(cx, cy, cz),
            is_fixed=True,
            surface=obstacle_surface,
            surface_color=obstacle_color,
            extra={"size": (dx * 2, dy * 2, dz * 2)},
        ))
    return entities


def manifest_to_entities(
    manifest: GenesisSceneManifest,
    *,
    include_meshes: bool = True,
    include_mjcf: bool = True,
    include_urdf: bool = True,
) -> list[GenesisEntityConfig]:
    """Convert a :class:`GenesisSceneManifest` to entity configs.

    Each discovered asset file becomes a :class:`GenesisEntityConfig` with
    auto-detected morph type and the absolute file path.

    Parameters
    ----------
    manifest : GenesisSceneManifest
        Asset manifest from :func:`scene_manifest_from_dir`.
    include_meshes, include_mjcf, include_urdf : bool
        Which asset types to include.

    Returns
    -------
    list[GenesisEntityConfig]
        One entity config per included asset file.
    """
    root = Path(manifest.root_dir)
    entities: list[GenesisEntityConfig] = []

    def _add(files: list[str], morph: str) -> None:
        for rel in files:
            p = root / rel
            entities.append(GenesisEntityConfig(
                name=p.stem,
                file_path=str(p),
                morph_type=morph,
            ))

    if include_meshes:
        _add(manifest.mesh_files, "Mesh")
    if include_mjcf:
        _add(manifest.mjcf_files, "MJCF")
    if include_urdf:
        _add(manifest.urdf_files, "URDF")
    return entities


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def _format_tuple(t: tuple) -> str:
    """Format a tuple as a Python literal."""
    return f"({', '.join(repr(v) for v in t)})"


def to_genesis_script(config: GenesisSceneConfig) -> str:
    """Generate a self-contained Python script that builds a Genesis scene.

    The generated script imports ``genesis``, initialises the backend,
    creates a scene with all entities/cameras/lights, builds it, and
    runs a basic simulation loop.

    Parameters
    ----------
    config : GenesisSceneConfig
        Full scene configuration.

    Returns
    -------
    str
        Python source code string.  Write to a ``.py`` file and run with
        ``python generated_scene.py``.
    """
    lines: list[str] = [
        '"""Genesis scene generated from Infinigen syndata export."""',
        "",
        "import genesis as gs",
        "",
        f"gs.init(backend=gs.{config.backend})",
        "",
    ]

    # ---- Scene constructor ----
    renderer_cls = config.renderer
    light_dicts = [l.to_dict() for l in config.lights]
    lines.append("scene = gs.Scene(")
    lines.append(f"    rigid_options=gs.options.RigidOptions(gravity={_format_tuple(config.gravity)}),")
    lines.append(f"    sim_options=gs.options.SimOptions(dt={config.dt!r}),")
    if light_dicts:
        lines.append(f"    renderer=gs.renderers.{renderer_cls}(")
        lines.append(f"        lights={light_dicts!r},")
        lines.append("    ),")
    else:
        lines.append(f"    renderer=gs.renderers.{renderer_cls}(),")
    lines.append(")")
    lines.append("")

    # ---- Entities ----
    for ent in config.entities:
        var = ent.name.replace("-", "_").replace(" ", "_").replace(".", "_")
        morph_args: list[str] = []
        if ent.file_path:
            morph_args.append(f'file="{ent.file_path}"')
        if ent.pos != (0.0, 0.0, 0.0):
            morph_args.append(f"pos={_format_tuple(ent.pos)}")
        if ent.euler != (0.0, 0.0, 0.0):
            morph_args.append(f"euler={_format_tuple(ent.euler)}")
        if ent.scale != 1.0:
            if isinstance(ent.scale, tuple):
                morph_args.append(f"scale={_format_tuple(ent.scale)}")
            else:
                morph_args.append(f"scale={ent.scale!r}")
        if ent.is_fixed:
            morph_args.append("fixed=True")
        # Extra morph kwargs (e.g. size for Box)
        for k, v in ent.extra.items():
            if isinstance(v, tuple):
                morph_args.append(f"{k}={_format_tuple(v)}")
            else:
                morph_args.append(f"{k}={v!r}")
        morph_str = ", ".join(morph_args)

        surface_str = ""
        if ent.surface:
            sargs: list[str] = []
            if ent.surface_color is not None:
                sargs.append(f"color={_format_tuple(ent.surface_color)}")
            surface_str = f", surface=gs.surfaces.{ent.surface}({', '.join(sargs)})"

        lines.append(
            f"{var} = scene.add_entity("
            f"gs.morphs.{ent.morph_type}({morph_str}){surface_str})"
        )
    lines.append("")

    # ---- Cameras ----
    for cam in config.cameras:
        var = cam.name.replace("-", "_").replace(" ", "_")
        lines.append(f"{var} = scene.add_camera(")
        lines.append(f"    res={_format_tuple(cam.res)},")
        lines.append(f"    pos={_format_tuple(cam.pos)},")
        lines.append(f"    lookat={_format_tuple(cam.lookat)},")
        lines.append(f"    fov={cam.fov!r},")
        lines.append(f"    near={cam.near!r},")
        lines.append(f"    far={cam.far!r},")
        lines.append(f"    spp={cam.spp!r},")
        lines.append("    GUI=False,")
        lines.append(")")
    lines.append("")

    # ---- Build & simulate ----
    lines.append("scene.build()")
    lines.append("")
    lines.append("# ---- Simulation loop ----")
    lines.append(f"for step in range({_DEFAULT_SIMULATION_STEPS}):")
    lines.append("    scene.step()")
    if config.cameras:
        first_cam = config.cameras[0].name.replace("-", "_").replace(" ", "_")
        lines.append(f"    rgb, depth, seg, normal = {first_cam}.render(")
        lines.append("        rgb=True, depth=True, segmentation=True, normal=True,")
        lines.append("    )")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level assembly
# ---------------------------------------------------------------------------


def build_genesis_config(
    *,
    export_dir: str | Path | None = None,
    frame_metadata: Any | None = None,
    drone_camera: Any | None = None,
    camera_rig: Any | None = None,
    observation: Any | None = None,
    randomiser: Any | None = None,
    resolution: tuple[int, int] = (640, 480),
    backend: str = "cuda",
    renderer: str = "RayTracer",
) -> GenesisSceneConfig:
    """Assemble a complete :class:`GenesisSceneConfig` from syndata components.

    This is the main entry-point for converting an Infinigen scene
    (export directory + syndata metadata/config) into a Genesis-ready
    configuration.

    Parameters
    ----------
    export_dir : str | Path | None
        If provided, scan for mesh/MJCF/URDF assets.
    frame_metadata : FrameMetadata | None
        If provided, generate box entities for obstacle bounding boxes.
    drone_camera : DroneCamera | None
        Camera optics.
    camera_rig : CameraRigConfig | None
        Camera rig layout.
    observation : ObservationConfig | None
        Observation space config.
    randomiser : DomainRandomiser | None
        Domain randomisation source for lighting.
    resolution : tuple[int, int]
        Render resolution ``(width, height)``.
    backend : str
        Genesis compute backend.
    renderer : str
        Renderer type.

    Returns
    -------
    GenesisSceneConfig
        Ready-to-use scene configuration.
    """
    entities: list[GenesisEntityConfig] = []
    cameras: list[GenesisCamera] = []
    lights: list[GenesisLight] = []

    # ---- Discover assets from export directory ----
    if export_dir is not None:
        manifest = scene_manifest_from_dir(export_dir)
        entities.extend(manifest_to_entities(manifest))

    # ---- Add obstacle boxes from metadata ----
    if frame_metadata is not None:
        entities.extend(metadata_to_entities(frame_metadata))

    # ---- Always add a ground plane ----
    if not any(e.morph_type == "Plane" for e in entities):
        entities.insert(0, GenesisEntityConfig(
            name="ground",
            morph_type="Plane",
            is_fixed=True,
        ))

    # ---- Cameras ----
    if drone_camera is not None and camera_rig is not None:
        cameras = camera_from_syndata(drone_camera, camera_rig, resolution=resolution)
    elif not cameras:
        cameras = [GenesisCamera(name="cam_0", res=resolution)]

    # ---- Lights ----
    if randomiser is not None:
        lights = randomisation_to_genesis_lights(randomiser)
    else:
        lights = [GenesisLight()]  # default overhead light

    return GenesisSceneConfig(
        entities=entities,
        cameras=cameras,
        lights=lights,
        renderer=renderer,
        backend=backend,
    )
