import os, math, random
import numpy as np


class TerrainProvider:
    """Provides elevation queries from real or archetype DEM data.

    For OSM cities: load from lat/lon bounds.
    For procedural cities: load from archetype + procedural noise.
    """

    def __init__(self, elevation_array=None, bounds=None, transform=None):
        self._arr = elevation_array
        self._bounds = bounds
        self._transform = transform
        self._min_x = self._max_x = self._min_y = self._max_y = None

    @classmethod
    def from_archetype(cls, name, noise_scale=0.0, noise_amplitude=0.0, seed=0):
        """Load an archetype DEM and optionally add Perlin noise."""
        path = os.path.join(os.path.dirname(__file__), "terrain_archetypes", f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archetype DEM not found: {path}")
        data = np.load(path)
        arr = data["elevation"].astype(np.float64)
        left, bottom, right, top = data["left"], data["bottom"], data["right"], data["top"]
        h, w = arr.shape
        if noise_amplitude > 0 and noise_scale > 0:
            noise = _perlin_noise_2d(h, w, scale=noise_scale, seed=seed) * noise_amplitude
            arr = arr + noise
        bounds = (left, bottom, right, top)
        transform = ((right - left) / w, 0, left, 0, (bottom - top) / h, top)
        return cls(arr, bounds, transform)

    @classmethod
    def from_rasterio(cls, path):
        """Load from a GeoTIFF file using rasterio."""
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float64)
            bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
            transform = (ds.transform.a, ds.transform.b, ds.transform.c,
                         ds.transform.d, ds.transform.e, ds.transform.f)
        return cls(arr, bounds, transform)

    @classmethod
    def flat(cls, elevation=0.0):
        """A flat terrain at a constant elevation."""
        return cls(elevation_array=np.array([[elevation]]),
                   bounds=(0, 0, 1, 1),
                   transform=(1, 0, 0, 0, 1, 0))

    def _map_to_utm(self, xs, ys):
        """Convert UTM coordinates to raster indices (for already-aligned data).

        For OSM data in UTM, we store UTM bounds directly.
        For lat/lon archetypes, we store lat/lon bounds.
        This method is overridden by UTM-mapped providers.
        """
        return self._map_to_raster(xs, ys)

    def _map_to_raster(self, xs, ys):
        """Convert world coordinates to raster pixel indices."""
        left, bottom, right, top = self._bounds
        h, w = self._arr.shape
        px = ((np.asarray(xs) - left) / (right - left) * w).astype(int)
        py = ((np.asarray(ys) - top) / (bottom - top) * h).astype(int)
        # Py is inverted because raster rows go top-to-bottom but world Y goes bottom-to-top
        if bottom < top:
            py = ((top - np.asarray(ys)) / (top - bottom) * h).astype(int)
        else:
            py = ((np.asarray(ys) - bottom) / (top - bottom) * h).astype(int)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        return px, py

    def set_utm_bounds(self, utm_min_x, utm_max_x, utm_min_y, utm_max_y):
        self._bounds_utm = (utm_min_x, utm_min_y, utm_max_x, utm_max_y)
        return self

    def get_elevation(self, x, y):
        """Return elevation at a single (x, y) coordinate."""
        px, py = self._map_to_raster([x], [y])
        return float(self._arr[py[0], px[0]])

    def get_elevation_batch(self, xs, ys):
        """Return elevations at multiple coordinates."""
        px, py = self._map_to_raster(xs, ys)
        return self._arr[py, px]

    def get_elevation_footprint(self, boundary):
        """Average elevation over a building footprint polygon."""
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        return float(np.mean(self.get_elevation_batch(xs, ys)))


class TerrainModifier:
    """Applies terrain to road/building geometry.

    Modes: gentle, moderate, aggressive
    """

    def __init__(self, terrain_provider, mode="moderate"):
        self.terrain = terrain_provider
        self.mode = mode

    def road_vertex_z(self, x, y):
        z = self.terrain.get_elevation(x, y)
        return z

    def building_z(self, boundary):
        avg_z = self.terrain.get_elevation_footprint(boundary)
        if self.mode == "aggressive":
            return avg_z - 0.5
        if self.mode == "moderate":
            return avg_z
        return avg_z

    def terrain_mesh(self, utm_min_x, utm_max_x, utm_min_y, utm_max_y, resolution=2.0):
        """Generate a subdivided ground mesh with terrain height displacement."""
        import bpy
        w = utm_max_x - utm_min_x
        h = utm_max_y - utm_min_y
        nx = max(2, int(w / resolution))
        ny = max(2, int(h / resolution))
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=nx, y_subdivisions=ny,
                                        size=max(w, h), location=((utm_min_x+utm_max_x)/2,
                                                                   (utm_min_y+utm_max_y)/2, 0))
        obj = bpy.context.active_object
        mesh = obj.data
        verts = mesh.vertices
        for v in verts:
            v.co.z = self.terrain.get_elevation(v.co.x, v.co.y)
        mesh.update()
        obj.name = "terrain"
        return obj


# Archetype presets per regional style
ARCHETYPE_MAP = {
    "mediterranean": "stpaul",
    "soviet": "narva",
    "baltic": "narva",
    "generic": "warsaw",
    "european_old": "monaco",
}


def terrain_from_preset(preset_name, city_size=200, noise_scale=0.05, noise_amplitude=2.0, seed=0):
    """Create a TerrainProvider for a procedural preset (non-OSM)."""
    from infinigen.assets.urban.city_presets import load_preset
    try:
        preset = load_preset(preset_name)
        style = preset.get("regional_style", "generic")
    except ValueError:
        style = "generic"
    arch = ARCHETYPE_MAP.get(style, "warsaw")
    return TerrainProvider.from_archetype(
        arch, noise_scale=noise_scale * city_size,
        noise_amplitude=noise_amplitude, seed=seed,
    )


def terrain_from_osm(road_segments, cache_dir="~/.cache/infinigen/dem", seed=0):
    """Download and create TerrainProvider from OSM road segment bounds.

    Falls back to flat terrain if download fails.
    """
    xs = [p[0] for seg in road_segments for p in (seg.source, seg.target)]
    ys = [p[1] for seg in road_segments for p in (seg.source, seg.target)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    from infinigen.assets.urban.osmnx_skeleton import _utm_to_latlon
    try:
        lat_min, lon_min = _utm_to_latlon(min_x, min_y)
        lat_max, lon_max = _utm_to_latlon(max_x, max_y)
        bounds = (min(lon_min, lon_max), min(lat_min, lat_max),
                  max(lon_min, lon_max), max(lat_min, lat_max))
        import tempfile, os
        cdir = os.path.expanduser(cache_dir)
        os.makedirs(cdir, exist_ok=True)
        cache_key = f"{bounds[0]:.2f}_{bounds[1]:.2f}_{bounds[2]:.2f}_{bounds[3]:.2f}"
        cache_path = os.path.join(cdir, f"{cache_key}.tif")
        if os.path.exists(cache_path):
            return TerrainProvider.from_rasterio(cache_path)
        import subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_path = tmp.name
        tmp.close()
        import elevation
        elevation.clip(bounds=bounds, output=tmp_path)
        provider = TerrainProvider.from_rasterio(tmp_path)
        import shutil
        shutil.move(tmp_path, cache_path)
        return provider
    except Exception as e:
        print(f"DEM download failed: {e}, using flat terrain")
        return TerrainProvider.flat()


def _perlin_noise_2d(h, w, scale=10.0, seed=0):
    """Simple Perlin-like noise for terrain variation."""
    rng = np.random.RandomState(seed)
    noise = np.zeros((h, w))
    angle = rng.uniform(0, 2 * np.pi, size=(int(h / scale) + 2, int(w / scale) + 2))
    for i in range(h):
        for j in range(w):
            gi = i / scale
            gj = j / scale
            si, sj = int(gi), int(gj)
            fi, fj = gi - si, gj - sj
            fi, fj = fi * fi * (3 - 2 * fi), fj * fj * (3 - 2 * fj)
            n00 = angle[si, sj]
            n10 = angle[si + 1, sj]
            n01 = angle[si, sj + 1]
            n11 = angle[si + 1, sj + 1]
            v00 = (gi - si) * math.cos(n00) + (gj - sj) * math.sin(n00)
            v10 = (gi - (si + 1)) * math.cos(n10) + (gj - sj) * math.sin(n10)
            v01 = (gi - si) * math.cos(n01) + (gj - (sj + 1)) * math.sin(n01)
            v11 = (gi - (si + 1)) * math.cos(n11) + (gj - (sj + 1)) * math.sin(n11)
            v0 = v00 + fi * (v10 - v00)
            v1 = v01 + fi * (v11 - v01)
            noise[i, j] = v0 + fj * (v1 - v0)
    return noise