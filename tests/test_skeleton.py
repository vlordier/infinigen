from infinigen.assets.urban.skeleton import (
    GridGenerator, OrganicSpineGenerator, RadialGenerator, SingleSpineGenerator,
)


def test_radial_skeleton_basic():
    result = RadialGenerator.generate(size=200, n_radials=8, n_rings=3, seed=42)
    assert len(result.road_segments) >= 8 * 3
    assert len(result.blocks) >= 8 * 3
    for block in result.blocks:
        assert len(block.boundary) >= 3
        assert isinstance(block.zone_id, str)


def test_radial_deterministic():
    a = RadialGenerator.generate(200, seed=42)
    b = RadialGenerator.generate(200, seed=42)
    assert len(a.road_segments) == len(b.road_segments)
    assert len(a.blocks) == len(b.blocks)


def test_radial_different_seed():
    a = RadialGenerator.generate(200, seed=42)
    b = RadialGenerator.generate(200, seed=99)
    assert len(a.road_segments) == len(b.road_segments)


def test_grid_skeleton():
    result = GridGenerator.generate(size=200, rows=5, cols=5, seed=42)
    assert len(result.blocks) == 5 * 5


def test_organic_spine():
    result = OrganicSpineGenerator.generate(size=200, seed=42)
    assert len(result.road_segments) >= 3
    assert len(result.blocks) >= 1


def test_single_spine():
    result = SingleSpineGenerator.generate(size=200, seed=42)
    assert len(result.road_segments) >= 2
    assert len(result.blocks) >= 1
