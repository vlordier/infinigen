from infinigen.assets.urban.graph_parser import RoadSegment
from infinigen.assets.urban.skeleton import CitySkeleton, BlockFace
from infinigen.assets.urban.road_to_dcel import RoadToDCEL


def _utm_to_latlon(easting, northing, zone=None, hem=None):
    """Convert UTM easting/northing to (lat, lon) using pyproj."""
    try:
        import pyproj
    except ImportError:
        return (0.0, 0.0)
    if zone is None:
        zone = int((easting / 100000 + 31) % 60) or 60
    hem = "south" if northing < 0 else "north"
    crs_utm = f"+proj=utm +zone={int(zone)} +{hem} +datum=WGS84 +units=m"
    crs_wgs = "+proj=longlat +datum=WGS84"
    transformer = pyproj.Transformer.from_crs(crs_utm, crs_wgs, always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return (lat, lon)


_ROAD_TYPE_MAP = {
    'motorway': 'arterial',
    'motorway_link': 'arterial',
    'trunk': 'arterial',
    'trunk_link': 'arterial',
    'primary': 'arterial',
    'primary_link': 'collector',
    'secondary': 'collector',
    'secondary_link': 'collector',
    'tertiary': 'local',
    'tertiary_link': 'local',
    'residential': 'local',
    'unclassified': 'local',
    'living_street': 'alley',
    'service': 'alley',
}

_LANE_ESTIMATE = {
    'motorway': 4, 'trunk': 4, 'primary': 3, 'secondary': 2,
    'tertiary': 2, 'residential': 2, 'unclassified': 1,
    'living_street': 1, 'service': 1,
}

_WIDTH_ESTIMATE = {
    'motorway': 24.0, 'trunk': 24.0, 'primary': 20.0, 'secondary': 14.0,
    'tertiary': 10.0, 'residential': 8.0, 'unclassified': 6.0,
    'living_street': 5.0, 'service': 4.0,
}


class OsmnxSkeleton:
    @staticmethod
    def generate(place=None, point=None, dist=None, network_type="drive",
                 simplify=True, retain_all=False, truncate_by_edge=False,
                 crop_size=None, size=200, seed=0) -> CitySkeleton:
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("Install osmnx: pip install osmnx")
        ox.settings.log_console = False
        ox.settings.use_cache = True
        if place:
            G = ox.graph_from_place(place, network_type=network_type,
                                    simplify=simplify, retain_all=retain_all,
                                    truncate_by_edge=truncate_by_edge)
        elif point and dist:
            G = ox.graph_from_point(point, dist=dist, network_type=network_type,
                                    simplify=simplify)
        else:
            raise ValueError("Provide place= or point= + dist=")
        G_proj = ox.project_graph(G)
        skeleton = OsmnxSkeleton.from_graph(G_proj)
        if crop_size and len(skeleton.road_segments) > 1000:
            xs = [p[0] for seg in skeleton.road_segments for p in (seg.source, seg.target)]
            ys = [p[1] for seg in skeleton.road_segments for p in (seg.source, seg.target)]
            cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
            half = crop_size / 2
            keep = []
            for seg in skeleton.road_segments:
                if (abs(seg.source[0]-cx) < half and abs(seg.source[1]-cy) < half) or \
                   (abs(seg.target[0]-cx) < half and abs(seg.target[1]-cy) < half):
                    keep.append(seg)
            dcel = RoadToDCEL.build(keep)
            blocks = OsmnxSkeleton._extract_blocks(dcel)
            skeleton = CitySkeleton(road_segments=keep, blocks=blocks)
        return skeleton

    @staticmethod
    def from_graph(G) -> CitySkeleton:
        segs = []
        nodes_dict = dict(G.nodes(data=True))
        for u, v, k, data in G.edges(keys=True, data=True):
            geo = data.get('geometry', None)
            if geo:
                coords = list(geo.coords)
                for i in range(len(coords) - 1):
                    segs.append(OsmnxSkeleton._edge_to_segment(
                        coords[i][0], coords[i][1],
                        coords[i + 1][0], coords[i + 1][1],
                        data,
                    ))
            else:
                s_x = nodes_dict[u].get('x', 0)
                s_y = nodes_dict[u].get('y', 0)
                t_x = nodes_dict[v].get('x', 0)
                t_y = nodes_dict[v].get('y', 0)
                segs.append(OsmnxSkeleton._edge_to_segment(
                    s_x, s_y, t_x, t_y, data,
                ))
        dcel = RoadToDCEL.build(segs)
        blocks = OsmnxSkeleton._extract_blocks(dcel)
        return CitySkeleton(road_segments=segs, blocks=blocks)

    @staticmethod
    def _edge_to_segment(s_x, s_y, t_x, t_y, data):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0]
        road_type = _ROAD_TYPE_MAP.get(highway, 'local')
        lanes = data.get('lanes', _LANE_ESTIMATE.get(highway, 2))
        if isinstance(lanes, str):
            try:
                lanes = int(lanes)
            except ValueError:
                lanes = _LANE_ESTIMATE.get(highway, 2)
        width = data.get('width', _WIDTH_ESTIMATE.get(highway, 8.0))
        if isinstance(width, str):
            try:
                width = float(width)
            except ValueError:
                width = _WIDTH_ESTIMATE.get(highway, 8.0)
        sidewalk = road_type in ('arterial', 'collector', 'local')
        return RoadSegment(
            source=(s_x, s_y), target=(t_x, t_y),
            road_type=road_type, lane_count=lanes,
            width=width, sidewalk=sidewalk,
        )

    @staticmethod
    def _extract_blocks(dcel) -> list[BlockFace]:
        blocks = []
        for face in dcel.faces:
            if face.is_boundary:
                continue
            he = face.half_edge
            if he is None:
                continue
            boundary = []
            start = he
            while True:
                boundary.append(he.origin.position)
                he = he.next
                if he is start or he is None:
                    break
            if len(boundary) >= 3:
                blocks.append(BlockFace(boundary=boundary, zone_id="inner"))
        return blocks
