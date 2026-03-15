import os

import numpy as np

try:
    import mcubes
except ImportError:
    mcubes = None

try:
    from skimage.measure import marching_cubes as skimage_marching_cubes
except ImportError:
    skimage_marching_cubes = None


def _resolve_backend(backend: str | None = None) -> str:
    backend = backend or os.getenv("INFINIGEN_ISOSURFACE_BACKEND", "auto")
    if backend == "auto":
        if skimage_marching_cubes is not None:
            return "skimage"
        if mcubes is not None:
            return "mcubes"
        raise ImportError(
            "No marching cubes backend available. Install `mcubes` or `scikit-image`."
        )
    if backend == "mcubes":
        if mcubes is None:
            raise ImportError("Requested marching cubes backend `mcubes`, but it is not installed.")
        return backend
    if backend == "skimage":
        if skimage_marching_cubes is None:
            raise ImportError(
                "Requested marching cubes backend `skimage`, but scikit-image is not installed."
            )
        return backend
    raise ValueError(f"Unknown marching cubes backend: {backend}")


def marching_cubes_vertices_faces(
    volume: np.ndarray, level: float, backend: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return only vertices/faces using the configured marching-cubes backend."""
    backend = _resolve_backend(backend)
    volume = np.ascontiguousarray(volume, dtype=np.float32)

    if backend == "mcubes":
        vertices, faces = mcubes.marching_cubes(volume, level)
        return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)

    vertices, faces, _, _ = skimage_marching_cubes(volume, level)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)
