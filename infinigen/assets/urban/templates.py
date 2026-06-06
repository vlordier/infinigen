from dataclasses import dataclass, field
import random
from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.block_subdivision import BuildingLot


@dataclass
class DistrictTemplateConfig:
    internal_road_width: float = 8.0
    internal_sidewalk: bool = False
    lot_depth: float = 30.0
    lot_width: float = 20.0
    lot_min_area: float = 20.0
    irregularity: float = 0.0
    dead_end_chance: float = 0.0
    density: float = 0.5


@dataclass
class DistrictFill:
    road_segments: list[RoadSegment] = field(default_factory=list)
    building_lots: list[BuildingLot] = field(default_factory=list)


class BaseTemplate:
    name = "base"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        raise NotImplementedError


def register_template(cls):
    _TEMPLATE_REGISTRY[cls.name] = cls
    return cls


_TEMPLATE_REGISTRY = {}


def get_template(name: str):
    return _TEMPLATE_REGISTRY.get(name)


from infinigen.assets.urban.template_utils import make_grid_segments, bbox_lots


@register_template
class RectangularGridTemplate(BaseTemplate):
    name = "rectangular_grid"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs = make_grid_segments(
            (x0, y0), (x1, y1),
            spacing=max(config.lot_depth, config.lot_width) * 2,
            road_type="local", width=config.internal_road_width,
            sidewalk=config.internal_sidewalk, rng=rng,
            irregularity=config.irregularity,
        )
        lots = bbox_lots(
            (x0 + 2, y0 + 2), (x1 - 2, y1 - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)


@register_template
class OrganicGridTemplate(BaseTemplate):
    name = "organic_grid"

    @staticmethod
    def fill(boundary, config: DistrictTemplateConfig, rng: random.Random) -> DistrictFill:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        spacing = max(config.lot_depth, config.lot_width) * 2
        segs = make_grid_segments(
            (x0, y0), (x1, y1), spacing=spacing,
            road_type="local", width=config.internal_road_width,
            sidewalk=config.internal_sidewalk,
            irregularity=config.irregularity, rng=rng,
        )
        from infinigen.assets.urban.template_utils import clip_segments_to_boundary
        segs = clip_segments_to_boundary(segs, boundary)
        dither = spacing * config.irregularity * 0.5 if config.irregularity else 0
        lots = bbox_lots(
            (x0 + dither + 2, y0 + dither + 2),
            (x1 - dither - 2, y1 - dither - 2),
            lot_width=config.lot_width, lot_depth=config.lot_depth,
        )
        lots = [l for l in lots if l.area >= config.lot_min_area]
        return DistrictFill(road_segments=segs, building_lots=lots)
