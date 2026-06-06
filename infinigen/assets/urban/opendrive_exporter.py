from datetime import datetime
import math
from infinigen.assets.urban.dcel import DCEL, DCELNode
from infinigen.assets.urban.graph_parser import RoadSegment


def _heading(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return math.atan2(dy, dx)


def _width_markings(lane_count, road_type):
    if road_type == "highway":
        return 3.5, "solid", "solid"
    if road_type == "arterial":
        return 3.5, "solid", "solid"
    if road_type == "collector":
        return 3.5, "solid", "solid"
    if road_type == "alley":
        return 2.5, "none", "none"
    return 3.0, "solid", "solid"


def _lane_type(seg):
    if seg.road_type == "highway":
        return "driving"
    if seg.road_type == "arterial":
        return "driving"
    if seg.road_type == "collector":
        return "driving"
    if seg.road_type == "alley":
        return "driving"
    return "driving"


def export_opendrive(road_segments: list[RoadSegment], filename: str):
    from xml.etree.ElementTree import Element, SubElement, tostring
    import xml.dom.minidom

    ns_xodr = "http://www.opendrive.org"

    roads_by_node: dict[tuple[float, float], list[RoadSegment]] = {}
    seg_map: dict[int, dict] = {}
    for seg in road_segments:
        src, tgt = seg.source, seg.target
        roads_by_node.setdefault(src, []).append(seg)
        roads_by_node.setdefault(tgt, []).append(seg)

    nodes_set = set()
    for seg in road_segments:
        nodes_set.add(seg.source)
        nodes_set.add(seg.target)
    node_list = sorted(nodes_set)

    node_degree = {n: len(roads_by_node.get(n, [])) for n in node_list}
    is_junction = {n: node_degree[n] >= 3 for n in node_list}

    odr = Element("OpenDRIVE")
    header = SubElement(odr, "header")
    header.set("revMajor", "1")
    header.set("revMinor", "4")
    header.set("name", "infinigen_city")
    header.set("version", "1.00")
    header.set("date", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    xs = [n[0] for n in node_list]
    ys = [n[1] for n in node_list]
    header.set("north", str(max(ys)))
    header.set("south", str(min(ys)))
    header.set("east", str(max(xs)))
    header.set("west", str(min(xs)))

    road_id_counter = [0]
    junction_id_counter = [0]
    road_to_seg: dict[int, RoadSegment] = {}

    junctions: dict[tuple[float, float], int] = {}
    for n in node_list:
        if is_junction[n]:
            jid = str(junction_id_counter[0] + 1)
            junction_id_counter[0] += 1
            junctions[n] = jid

    junction_roads: list[tuple[int, RoadSegment, tuple[float, float], tuple[float, float]]] = []

    for seg in road_segments:
        src, tgt = seg.source, seg.target
        src_j = junctions.get(src)
        tgt_j = junctions.get(tgt)
        junction_id = "-1"
        if src_j or tgt_j:
            junction_id = src_j if src_j else tgt_j

        road = SubElement(odr, "road")
        rid = road_id_counter[0] + 1
        road_id_counter[0] += 1
        road.set("name", f"road_{rid}")
        road.set("id", str(rid))
        road.set("junction", junction_id if junction_id != "-1" else "-1")

        dx, dy = tgt[0] - src[0], tgt[1] - src[1]
        length = math.sqrt(dx * dx + dy * dy)
        road.set("length", f"{length:.4f}")

        link = SubElement(road, "link")
        if src in junctions:
            SubElement(link, "predecessor", elementType="junction", elementId=junctions[src])
        elif node_degree[src] >= 2:
            for other in roads_by_node[src]:
                if other != seg:
                    pred_id = road_id_counter[0]
                    SubElement(link, "predecessor", elementType="road", elementId=str(pred_id))
                    break

        if tgt in junctions:
            SubElement(link, "successor", elementType="junction", elementId=junctions[tgt])
        elif node_degree[tgt] >= 2:
            for other in roads_by_node[tgt]:
                if other != seg:
                    succ_id = road_id_counter[0]
                    SubElement(link, "successor", elementType="road", elementId=str(succ_id))
                    break

        planview = SubElement(road, "planView")
        geo = SubElement(planview, "geometry")
        hdg = _heading(src, tgt)
        geo.set("s", "0.000")
        geo.set("x", f"{src[0]:.4f}")
        geo.set("y", f"{src[1]:.4f}")
        geo.set("hdg", f"{hdg:.6f}")
        geo.set("length", f"{length:.4f}")
        SubElement(geo, "line")

        lanes = SubElement(road, "lanes")
        lane_offset = SubElement(lanes, "laneOffset")
        lane_offset.set("s", "0")
        lane_offset.set("a", f"{seg.width / 2:.4f}")
        lane_offset.set("b", "0")
        lane_offset.set("c", "0")
        lane_offset.set("d", "0")

        ls = SubElement(lanes, "laneSection")
        ls.set("s", "0")

        center = SubElement(ls, "center")
        center_lane = SubElement(center, "lane")
        center_lane.set("id", "0")
        center_lane.set("type", "none")
        center_lane.set("level", "false")

        right = SubElement(ls, "right")
        lane_w, right_mark, left_mark = _width_markings(seg.lane_count, seg.road_type)
        for li in range(1, seg.lane_count + 1):
            l = SubElement(right, "lane")
            l.set("id", str(-li))
            l.set("type", _lane_type(seg))
            l.set("level", "false")
            lw = SubElement(l, "width")
            lw.set("s", "0")
            lw.set("a", f"{lane_w:.4f}")
            lw.set("b", "0")
            lw.set("c", "0")
            lw.set("d", "0")
            rm = SubElement(l, "roadMark")
            rm.set("s", "0")
            rm.set("type", right_mark if li == seg.lane_count else "broken")
            rm.set("weight", "standard")
            rm.set("color", "white")
            rm.set("width", "0.15")

        road_to_seg[rid] = seg

    for n, jid in junctions.items():
        junc = SubElement(odr, "junction")
        junc.set("name", f"junction_{jid}")
        junc.set("id", jid)
        conn_segs = roads_by_node.get(n, [])
        for ci, seg in enumerate(conn_segs):
            conn = SubElement(junc, "connection")
            conn.set("id", str(ci))
            src, tgt = seg.source, seg.target
            is_incoming = (abs(src[0] - n[0]) < 0.1 and abs(src[1] - n[1]) < 0.1)
            if is_incoming:
                conn.set("incomingRoad", str(road_id_counter[0]))
            else:
                conn.set("incomingRoad", str(road_id_counter[0]))
            conn.set("connectingRoad", str(road_id_counter[0] + 1))
            conn.set("contactPoint", "start")

    xml_str = xml.dom.minidom.parseString(tostring(odr, encoding="unicode")).toprettyxml(indent="  ")
    with open(filename, "w") as f:
        f.write(xml_str)
    return filename