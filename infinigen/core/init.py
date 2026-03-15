# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.
import ast
import logging
import os
import random
import sys
from pathlib import Path

import bpy

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.
# See https://github.com/opencv/opencv/issues/21326#issuecomment-1008517425

import addon_utils
import gin
import numpy as np
from numpy.random import randint

import infinigen
from infinigen.core.util.logging import LogLevel
from infinigen.core.util.math import int_hash
from infinigen.core.util.organization import Task

logger = logging.getLogger(__name__)

CYCLES_GPUTYPES_PREFERENCE = [
    # key must be a valid cycles device_type
    # ordering indicate preference - earlier device types will be used over later if both are available
    #  - e.g most OPTIX gpus will also show up as a CUDA gpu, but we will prefer to use OPTIX due to this list's ordering
    "OPTIX",
    "CUDA",
    "METAL",  # Apple Silicon (M1/M2/M3/M4)
    "HIP",  # untested
    "ONEAPI",  # untested
    "CPU",
]

# Cycles denoiser selection priority: OPTIX (NVIDIA GPU), then OIDN (bundled,
# CPU-only, significantly improved in Blender 5.0), then disabled with a warning.
CYCLES_DENOISER_PRIORITY = ["OPTIX", "OPENIMAGEDENOISE"]

# Cycles volume rendering defaults (Blender 5.0: step_rate controls quality
# vs. speed trade-off; max_steps caps ray march iterations; bounces controls
# how many times a volume ray can scatter before being terminated).
CYCLES_VOLUME_STEP_RATE = 0.1
CYCLES_VOLUME_MAX_STEPS = 32
CYCLES_VOLUME_BOUNCES = 4

# Atmosphere quality tiers used by configure_volume_rendering():
#   "none"    – no volume scattering (default, fastest)
#   "fog"     – ground-level fog / low-visibility conditions
#   "haze"    – light atmospheric haze suitable for outdoor scenes
#   "dense"   – heavy industrial haze or storm atmosphere
ATMOSPHERE_QUALITY_PRESETS: dict[str, dict] = {
    "none": {
        "volume_step_rate": 0.1,
        "volume_max_steps": 32,
        "volume_bounces": 4,
        "use_world_volume": False,
    },
    "fog": {
        "volume_step_rate": 0.05,
        "volume_max_steps": 64,
        "volume_bounces": 8,
        "use_world_volume": True,
        "world_volume_density": 0.04,
        "world_volume_anisotropy": 0.3,
        "world_volume_color": (0.85, 0.87, 0.9, 1.0),
    },
    "haze": {
        "volume_step_rate": 0.08,
        "volume_max_steps": 48,
        "volume_bounces": 6,
        "use_world_volume": True,
        "world_volume_density": 0.015,
        "world_volume_anisotropy": 0.15,
        "world_volume_color": (0.9, 0.88, 0.82, 1.0),
    },
    "dense": {
        "volume_step_rate": 0.02,
        "volume_max_steps": 128,
        "volume_bounces": 12,
        "use_world_volume": True,
        "world_volume_density": 0.12,
        "world_volume_anisotropy": 0.5,
        "world_volume_color": (0.7, 0.68, 0.65, 1.0),
    },
}

# Cached device enumeration result to avoid repeated Blender API calls
_cached_devices: list | None = None

# Upper bound for randomly chosen scene seeds
MAX_RANDOM_SEED = int(1e7)


def parse_args_blender(parser):
    if "--" in sys.argv:
        # Running using a blender commandline python.
        # args before '--' are intended for blender not infinigen
        argvs = sys.argv[sys.argv.index("--") + 1 :]
        return parser.parse_args(argvs)
    else:
        return parser.parse_args()


def parse_seed(seed, task=None):
    if seed is None:
        if task is not None and Task.Coarse not in task:
            raise ValueError(
                "Running tasks on an already generated scene, you need to specify --seed or results will"
                " not be view-consistent"
            )
        return randint(MAX_RANDOM_SEED), "chosen at random"

    # WARNING: Do not add support for decimal numbers here, it will cause ambiguity, as some hex numbers are valid decimals

    try:
        return int(seed, 16), "parsed as hexadecimal"
    except ValueError:
        pass

    return int_hash(seed), "hashed string to integer"


def apply_scene_seed(seed, task=None):
    scene_seed, reason = parse_seed(seed, task)
    logger.info(f"Converted {seed=} to {scene_seed=}, {reason}")
    gin.constant("OVERALL_SEED", scene_seed)
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    return scene_seed


def sanitize_override(override: list):
    if ("=" in override) and not any((c in override) for c in "\"'[]"):
        k, v = override.split("=")
        try:
            ast.literal_eval(v)
        except (ValueError, SyntaxError):
            if "@" not in v:
                override = f'{k}="{v}"'

    return override


def contained_stems(filenames: list[str], folder: Path):
    assert folder.exists()
    names = [p.stem for p in folder.iterdir()]
    return {s.stem in names or s.name in names for s in map(Path, filenames)}


def resolve_folder_maybe_relative(folder, root):
    folder = Path(folder)
    if folder.exists():
        return folder
    folder_rel = root / folder
    if folder_rel.exists():
        return folder_rel
    raise FileNotFoundError(f"Could not find {folder} or {folder_rel}")


@gin.configurable
def apply_gin_configs(
    config_folders: Path | list[Path],
    configs: list[str] = None,
    overrides: list[str] = None,
    skip_unknown: bool = False,
    finalize_config=False,
    mandatory_folders: list[Path] = None,
    mutually_exclusive_folders: list[Path] = None,
):
    """
    Apply gin configuration files and bindings.

    Parameters
    ----------
    configs_folder : Path
        The path to the toplevel folder containing the gin configuration files.
    configs : list[str]
        A list of filenames to find within the configs_folder.
    overrides : list[str]
        A list of gin-formatted pairs to override the configs with.
    skip_unknown : bool
        If True, ignore errors for configs that were set by the user but not used anywhere
    mandatory_folders : list[Path]
        For each folder in the list, at least one config file must be loaded from that folder.
    mutually_exclusive_folders : list[Path]
        For each folder in the list, at most one config file must be loaded from that folder.

    """

    if configs is None:
        configs = []
    if overrides is None:
        overrides = []
    if mandatory_folders is None:
        mandatory_folders = []
    if mutually_exclusive_folders is None:
        mutually_exclusive_folders = []

    if not isinstance(config_folders, list):
        config_folders = [config_folders]

    root = infinigen.repo_root()

    def find_config(p):
        p = Path(p)
        for folder_rel in config_folders:
            folder = root / folder_rel

            if not folder.exists():
                raise ValueError(
                    f"{apply_gin_configs.__name__} got bad {folder_rel=}, {folder=} did not exist"
                )

            for file in folder.glob("**/*.gin"):
                if file.stem == p.stem:
                    logger.debug(f"Resolved {p} to file {file}")
                    return file
            logger.debug(f"Could not find {p} in {folder}")

        raise FileNotFoundError(
            f"Could not find {p} or {p.stem} in any of {config_folders}"
        )

    configs = [find_config(g) for g in ["base.gin"] + configs]
    overrides = [sanitize_override(o) for o in overrides]

    for mandatory_folder in mandatory_folders:
        mandatory_folder = resolve_folder_maybe_relative(mandatory_folder, root)
        if not contained_stems(configs, mandatory_folder):
            raise FileNotFoundError(
                f"At least one config file must be loaded from {mandatory_folder} to avoid unexpected behavior"
            )

    for mutex_folder in mutually_exclusive_folders:
        mutex_folder = resolve_folder_maybe_relative(mutex_folder, root)
        stems = {s.stem for s in mutex_folder.iterdir()}
        config_stems = {s.stem for s in configs}
        both = stems.intersection(config_stems)
        if len(both) > 1:
            raise ValueError(
                f"At most one config file must be loaded from {mutex_folder} to avoid unexpected behavior, instead got {both=}"
            )

    with LogLevel(logger=logging.getLogger(), level=logging.WARNING):
        gin.parse_config_files_and_bindings(
            configs,
            bindings=overrides,
            skip_unknown=skip_unknown,
            finalize_config=finalize_config,
        )


@gin.configurable
def configure_render_cycles(
    # supplied by gin.config
    min_samples,
    num_samples,
    time_limit,
    adaptive_threshold,
    exposure,
    denoise,
):
    bpy.context.scene.render.engine = "CYCLES"

    bpy.context.scene.cycles.use_denoising = denoise
    if denoise:
        _configure_denoiser()

    bpy.context.scene.cycles.samples = num_samples  # i.e. infinity
    bpy.context.scene.cycles.adaptive_min_samples = min_samples
    bpy.context.scene.cycles.adaptive_threshold = (
        adaptive_threshold  # i.e. noise threshold
    )
    bpy.context.scene.cycles.time_limit = time_limit
    bpy.context.scene.cycles.film_exposure = exposure

    # Enable persistent data when rendering multiple frames
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    if frame_end > frame_start:
        bpy.context.scene.render.use_persistent_data = True


def _configure_denoiser():
    """Select the best available Cycles denoiser with a hardened fallback chain.

    Blender 5.0 ships a significantly improved OIDN (Open Image Denoise) model
    that approaches OptiX quality without requiring an NVIDIA GPU.  The selection
    order is:

    1. **OPTIX** – best quality, NVIDIA GPU required.
    2. **OPENIMAGEDENOISE** (OIDN) – excellent quality, CPU-only, bundled in
       Blender 5.0.  This is the preferred fallback when OptiX is unavailable.
    3. **Disabled** – if neither denoiser is accessible, denoising is turned off
       with a clear warning so the operator knows to lower sample counts manually.

    The old code only tried OPTIX and left ``use_denoising=True`` with no denoiser
    configured when it failed (Blender would silently ignore the setting).  This
    function always leaves the scene in a defined state.
    """
    for denoiser in CYCLES_DENOISER_PRIORITY:
        try:
            bpy.context.scene.cycles.denoiser = denoiser
            logger.info(f"Cycles denoiser set to {denoiser}")
            return
        except (RuntimeError, TypeError) as e:
            logger.debug(f"Denoiser {denoiser} not available: {e}")

    # No supported denoiser — disable to avoid undefined Blender behavior.
    bpy.context.scene.cycles.use_denoising = False
    logger.warning(
        "No supported Cycles denoiser found (tried OPTIX, OPENIMAGEDENOISE). "
        "Denoising disabled — consider increasing num_samples manually."
    )


@gin.configurable
def configure_eevee_next(
    use_shadows: bool = False,
    use_gtao: bool = False,
    use_bloom: bool = False,
    taa_render_samples: int = 1,
    use_taa_reprojection: bool = False,
    use_high_quality_normals: bool = True,
):
    """Configure EEVEE Next for fast annotation / ground-truth rendering.

    EEVEE Next renders flat-shaded annotation passes (depth, normals, object
    index, instance segmentation) 10–50× faster than Cycles, because it uses
    hardware rasterisation rather than path tracing.  Since the flat-shading
    pass replaces all scene materials with simple random-color shaders and
    removes all volume and world lighting, the EEVEE output is pixel-identical
    to the Cycles output for every annotation pass Infinigen saves.

    This function is intended to be called instead of
    ``configure_cycles_devices()`` when ``flat_shading=True``.

    Args:
        use_shadows: Enable shadow rendering.  Disabled by default because
            annotation passes do not need accurate shadows, and disabling them
            further reduces render time.
        use_gtao: Enable ground-truth ambient occlusion.  Disabled by default
            for the same reason.
        use_bloom: Enable bloom post-processing.  Disabled by default.
        taa_render_samples: Temporal anti-aliasing sample count for the final
            render.  1 sample is sufficient for flat annotation passes.
        use_taa_reprojection: Use temporal reprojection in TAA.
        use_high_quality_normals: Use high-quality normal computation.
            Recommended for accurate normal annotation passes.
    """
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"

    eevee = bpy.context.scene.eevee
    eevee.use_shadows = use_shadows
    eevee.use_gtao = use_gtao
    eevee.use_bloom = use_bloom
    eevee.taa_render_samples = taa_render_samples
    eevee.use_taa_reprojection = use_taa_reprojection
    eevee.use_high_quality_normals = use_high_quality_normals

    logger.info(
        "Configured EEVEE Next for annotation rendering "
        f"({taa_render_samples} TAA sample(s), "
        f"{use_shadows=}, {use_gtao=})"
    )


@gin.configurable
def configure_cycles_devices(use_gpu=True):
    global _cached_devices

    if use_gpu is False:
        logger.info(f"Job will use CPU-only due to {use_gpu=}")
        bpy.context.scene.cycles.device = "CPU"
        return

    assert bpy.context.scene.render.engine == "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences

    if _cached_devices is None:
        # Enumerate devices once and cache the result
        for dt in prefs.get_device_types(bpy.context):
            prefs.get_devices_for_type(dt[0])
        _cached_devices = list(prefs.devices)

    assert len(_cached_devices) != 0, _cached_devices

    types = list(d.type for d in _cached_devices)

    types = sorted(types, key=CYCLES_GPUTYPES_PREFERENCE.index)
    logger.info(f"Available devices have {types=}")
    use_device_type = types[0]

    if use_device_type == "CPU":
        logger.warning(f"Job will use CPU-only, only found {types=}")
        bpy.context.scene.cycles.device = "CPU"
        return

    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = use_device_type
    use_devices = [d for d in _cached_devices if d.type == use_device_type]

    logger.info(f"Cycles will use {use_device_type=}, {len(use_devices)=}")

    for d in _cached_devices:
        d.use = False
    for d in use_devices:
        d.use = True

    return use_devices


def require_blender_addon(addon: str, fail: str = "fatal", allow_online=False):
    def report_fail(msg):
        if fail == "warn":
            logger.warning(
                msg + ". Generation may crash at runtime if certain assets are used."
            )
        elif fail == "fatal":
            raise ValueError(msg)
        else:
            raise ValueError(
                f"{require_blender_addon.__name__} got unrecognized {fail=}"
            )

    long = f"bl_ext.blender_org.{addon}"

    addon_present = long in bpy.context.preferences.addons.keys()

    if addon_present:
        logger.debug(f"Addon {addon} already present.")
        return True

    builtin_local_addons = set(a.__name__ for a in addon_utils.modules(refresh=True))

    if (
        (addon not in builtin_local_addons)
        and (long not in builtin_local_addons)
        and (not allow_online)
    ):
        report_fail(f"{addon=} not found and online install is disabled")

    try:
        if long in builtin_local_addons:
            logger.info(
                f"Addon {addon} already in blender local addons, attempt to enable it."
            )
            bpy.ops.preferences.addon_enable(module=long)
        else:
            bpy.ops.extensions.userpref_allow_online()
            logger.info(f"Installing Add-on {addon}.")
            bpy.ops.extensions.repo_sync(repo_index=0)
            bpy.ops.extensions.package_install(
                repo_index=0, pkg_id=addon, enable_on_install=True
            )
            bpy.ops.preferences.addon_enable(module=long)
    except (RuntimeError, TypeError) as e:
        report_fail(f"Failed to install {addon=} due to {e=}")

    if long not in bpy.context.preferences.addons.keys():
        report_fail(f"Attempted to install {addon=} but wasnt found after install")

    return True


_VALID_VIEW_TRANSFORMS = frozenset(
    {
        "Filmic",  # Blender default prior to 4.0
        "AgX",  # Blender 4.0+
        "ACES 2.0",  # Blender 5.0+ — industry-standard wide-gamut pipeline
        "None",  # Linear / no tone-mapping (for technical EXR passes)
        "Standard",  # Legacy sRGB direct
        "Filmic Log",
        "False Color",
    }
)


@gin.configurable
def configure_color_management(
    view_transform: str = "AgX",
    look: str = "None",
    exposure: float = 0.0,
    gamma: float = 1.0,
    display_device: str = "sRGB",
    sequencer_color_space: str = "sRGB",
):
    """Configure Blender's OCIO color-management pipeline.

    Blender 5.0 ships native ACES 1.3 and ACES 2.0 view transforms, enabling
    an industry-standard wide-gamut, scene-linear workflow without external OCIO
    configs.  Setting ``view_transform="ACES 2.0"`` activates the full ACES
    pipeline: ACEScg as the working space, proper HDR sky radiance, and
    color-accurate EXR ground-truth output that downstream depth/normal/flow
    networks can rely on.

    Args:
        view_transform: OCIO view transform. One of ``"Filmic"`` (legacy
            default), ``"AgX"`` (Blender 4.0+), ``"ACES 2.0"`` (Blender 5.0+,
            recommended for ML training data), ``"None"`` (linear, for raw
            technical passes).
        look: Optional contrast/look preset applied on top of the transform,
            e.g. ``"Medium High Contrast"`` or ``"None"``.
        exposure: Scene linear exposure adjustment in stops (0.0 = no change).
        gamma: Display gamma (1.0 = no change).
        display_device: Target display color space, e.g. ``"sRGB"`` or
            ``"Rec.2020"``.
        sequencer_color_space: Colour space for the video sequencer, e.g.
            ``"sRGB"`` or ``"Linear Rec.2020"``.
    """
    if view_transform not in _VALID_VIEW_TRANSFORMS:
        logger.warning(
            f"configure_color_management: unknown {view_transform=}. "
            f"Known values: {sorted(_VALID_VIEW_TRANSFORMS)}. "
            "Blender may reject this at runtime."
        )

    scene = bpy.context.scene
    scene.display_settings.display_device = display_device
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = look
    scene.view_settings.exposure = exposure
    scene.view_settings.gamma = gamma
    scene.sequencer_colorspace_settings.name = sequencer_color_space

    logger.info(
        f"Color management: {view_transform=}, {look=}, "
        f"{exposure=:.2f} stops, {display_device=}"
    )


@gin.configurable
def configure_render_time_pass(
    enabled: bool = False,
    log_on_enable: bool = True,
) -> bool:
    """Enable the Blender 5.0 Render Time pass on the active view layer.

    The **Render Time** pass (introduced in Blender 5.0) records the per-pixel
    render time in seconds as a greyscale EXR layer (socket name ``RenderTime``
    in the compositor ``Render Layers`` node).  This is the pass that Blender
    uses internally to build its render-cost heatmap in the viewport.

    Infinigen's compositor output already routes any pass listed in
    ``passes_to_save`` to a file slot, so enabling this pass automatically
    produces a ``RenderTime<frame>.exr`` file alongside depth, normals, and
    other GT data.  The resulting heatmap can be used to:

    - Identify expensive scene regions (complex foliage, volumetric fog,
      subsurface scattering) during dataset inspection.
    - Drive per-scene adaptive sample budgeting: if the measured render time
      for a given region exceeds a threshold, reduce ``num_samples`` for that
      quality tier.
    - Feed complexity signals back to the ``SceneBudget`` scheduler to stay
      within a per-frame wall-clock budget.

    The pass is disabled by default so that it has **zero overhead** for
    existing pipelines.  Activate it by adding ``render_time_pass.gin`` to any
    run's config stack, or by overriding ``configure_render_time_pass.enabled``
    in a custom gin binding.

    Args:
        enabled: When ``True``, sets
            ``bpy.context.scene.view_layers["ViewLayer"].cycles.use_pass_render_time = True``
            so that Cycles populates the ``RenderTime`` socket.  When ``False``
            (default) the pass is not enabled and no extra render overhead is
            incurred.
        log_on_enable: Emit an ``INFO``-level log message when the pass is
            activated.  Useful for confirming gin configuration is applied.

    Returns:
        Whether the render time pass is now enabled on the active view layer.
    """
    if not enabled:
        return False

    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    viewlayer.cycles.use_pass_render_time = True

    if log_on_enable:
        logger.info(
            "Render Time pass enabled — RenderTime EXR will be saved alongside "
            "other render passes.  Useful for per-pixel cost budgeting."
        )

    return True


# Pass descriptor used when the render time pass is wired into the compositor.
# Format: (viewlayer_pass_name, compositor_socket_name) — matches the
# passes_to_save list format consumed by render.configure_compositor_output().
RENDER_TIME_PASS_DESCRIPTOR = ("render_time", "RenderTime")


@gin.configurable
def configure_volume_rendering(
    volume_step_rate: float = CYCLES_VOLUME_STEP_RATE,
    volume_max_steps: int = CYCLES_VOLUME_MAX_STEPS,
    volume_bounces: int = CYCLES_VOLUME_BOUNCES,
    atmosphere_preset: str | None = None,
    use_world_volume: bool = False,
    world_volume_density: float = 0.02,
    world_volume_anisotropy: float = 0.2,
    world_volume_color: tuple[float, float, float, float] = (0.85, 0.87, 0.9, 1.0),
) -> None:
    """Configure Cycles volume rendering quality and optional world-space atmosphere.

    Blender 5.0 ships native volumetric data (OpenVDB grids) as first-class
    geometry-node citizens.  The Python-accessible rendering parameters
    (``volume_step_rate``, ``volume_max_steps``, ``volume_bounces``) control
    how accurately Cycles ray-marches through any volume present in the scene,
    whether authored via Geometry Nodes or via a World shader.

    This function also supports injecting a *World Volume* shader — a uniform
    homogeneous participating medium applied to the entire scene — which is the
    simplest way to add atmospheric fog, haze, or dense smog without authoring
    per-object volumes.

    **Atmosphere presets** (``atmosphere_preset``):

    ``"none"``   – no volume scattering (default; identical to baseline Infinigen)
    ``"fog"``    – ground-level fog / low-visibility with high anisotropy
    ``"haze"``   – light atmospheric haze for sunny outdoor scenes
    ``"dense"``  – heavy industrial smog or storm conditions

    When ``atmosphere_preset`` is set, it overrides all other keyword arguments
    so that a single gin binding activates a complete atmospheric profile:

    .. code-block:: python

        # Activate via gin
        configure_volume_rendering.atmosphere_preset = "fog"

    Individual parameters can still be overridden on top of a preset.

    Args:
        volume_step_rate: Cycles ray-march step size (relative to scene scale).
            Smaller values → higher quality, more render time.
        volume_max_steps: Maximum number of Cycles volume integration steps per
            ray.  Increase for very deep volumes (e.g. storm clouds).
        volume_bounces: Maximum number of volume-scattering events per ray.
        atmosphere_preset: When set to one of ``"none"``, ``"fog"``, ``"haze"``,
            or ``"dense"``, this overrides the individual parameters above with
            a balanced preset tuned for that condition.
        use_world_volume: When ``True``, injects a homogeneous participating
            medium (Principled Volume shader) into the scene World material.
        world_volume_density: Extinction coefficient of the world volume (higher
            = thicker fog/haze).
        world_volume_anisotropy: Henyey-Greenstein phase function anisotropy
            (``-1`` = back-scatter, ``0`` = isotropic, ``1`` = forward-scatter).
            Forward-scatter values (``0.2``–``0.5``) simulate realistic haze.
        world_volume_color: RGBA scattering colour of the world volume.
    """
    # Apply preset if requested — overrides individual parameters.
    if atmosphere_preset is not None:
        if atmosphere_preset not in ATMOSPHERE_QUALITY_PRESETS:
            known = list(ATMOSPHERE_QUALITY_PRESETS)
            logger.warning(
                f"Unknown atmosphere_preset={atmosphere_preset!r}; "
                f"known presets: {known}.  Ignoring preset and using "
                "explicit parameters."
            )
        else:
            cfg = ATMOSPHERE_QUALITY_PRESETS[atmosphere_preset]
            volume_step_rate = cfg.get("volume_step_rate", volume_step_rate)
            volume_max_steps = cfg.get("volume_max_steps", volume_max_steps)
            volume_bounces = cfg.get("volume_bounces", volume_bounces)
            use_world_volume = cfg.get("use_world_volume", use_world_volume)
            world_volume_density = cfg.get("world_volume_density", world_volume_density)
            world_volume_anisotropy = cfg.get(
                "world_volume_anisotropy", world_volume_anisotropy
            )
            world_volume_color = cfg.get("world_volume_color", world_volume_color)

    # Apply Cycles volume rendering quality settings.
    bpy.context.scene.cycles.volume_step_rate = volume_step_rate
    bpy.context.scene.cycles.volume_preview_step_rate = volume_step_rate
    bpy.context.scene.cycles.volume_max_steps = volume_max_steps
    bpy.context.scene.cycles.volume_bounces = volume_bounces

    logger.info(
        f"Volume rendering: {volume_step_rate=}, {volume_max_steps=}, "
        f"{volume_bounces=}, {use_world_volume=}"
    )

    if not use_world_volume:
        return

    # Inject a Principled Volume shader into the World material.
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    ntree = world.node_tree
    nodes = ntree.nodes
    links = ntree.links

    # Remove any existing volume socket connection to avoid duplicates.
    world_out = next(
        (n for n in nodes if n.type == "OUTPUT_WORLD"), None
    )
    if world_out is None:
        world_out = nodes.new("ShaderNodeOutputWorld")
        world_out.location = (300, 0)

    # Clear existing volume link if any.
    for link in list(links):
        if link.to_node == world_out and link.to_socket.name == "Volume":
            links.remove(link)

    # Create Principled Volume shader node.
    vol = nodes.new("ShaderNodeVolumePrincipled")
    vol.location = (0, 0)
    vol.inputs["Density"].default_value = world_volume_density
    vol.inputs["Anisotropy"].default_value = world_volume_anisotropy
    if "Color" in vol.inputs:
        vol.inputs["Color"].default_value = world_volume_color

    links.new(vol.outputs["Volume"], world_out.inputs["Volume"])
    logger.info(
        f"World volume atmosphere injected: "
        f"{world_volume_density=:.4f}, {world_volume_anisotropy=:.3f}"
    )


@gin.configurable
def configure_blender(
    render_engine="CYCLES",
    motion_blur=False,
    motion_blur_shutter=0.5,
):
    bpy.context.preferences.system.scrollback = 0
    bpy.context.preferences.edit.undo_steps = 0

    if render_engine == "CYCLES":
        configure_render_cycles()
        configure_cycles_devices()
    else:
        raise ValueError(f"Unrecognized {render_engine=}")

    configure_color_management()
    configure_volume_rendering()

    bpy.context.scene.render.use_motion_blur = motion_blur
    if motion_blur:
        bpy.context.scene.cycles.motion_blur_position = "START"
        bpy.context.scene.render.motion_blur_shutter = motion_blur_shutter
