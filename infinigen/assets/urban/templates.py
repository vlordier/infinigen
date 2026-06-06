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
