# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import bmesh
import bpy
import mathutils
import numpy as np

from infinigen.core.nodes.node_wrangler import Nodes

from .blender import ViewportMode


def special_bounds(obj):
    inf = 1e5
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    points = co.reshape(-1, 3)
    mask = np.sum(points**2, axis=-1) ** 0.5 < 0.5 * inf
    return points[mask].min(axis=0), points[mask].max(axis=0)


def on_bound_edges(points, points_min, points_max):
    flags = [0, 0, 0]
    eps = 1e-4
    for i in range(3):
        if abs(points[i] - points_min[i]) < eps:
            flags[i] = -1
        elif abs(points[i] - points_max[i]) < eps:
            flags[i] = 1
    return flags


def get_bevel_edges(obj):
    inf = 1e5
    points_min, points_max = special_bounds(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n_edges = len(bm.edges)
    # Extract all edge vertex positions at once
    v0_co = np.array([e.verts[0].co[:] for e in bm.edges])
    v1_co = np.array([e.verts[1].co[:] for e in bm.edges])
    eps = 1e-4
    # Compute bound flags for both endpoints
    def _bound_flags(co, pmin, pmax):
        flags = np.zeros(co.shape, dtype=int)
        flags[np.abs(co - pmin) < eps] = -1
        flags[np.abs(co - pmax) < eps] = 1
        return flags
    f0 = _bound_flags(v0_co, points_min, points_max)
    f1 = _bound_flags(v1_co, points_min, points_max)
    on_bounds = np.sum((f0 != 0) & (f0 == f1), axis=1)
    mags0 = np.linalg.norm(v0_co, axis=1)
    mask = (on_bounds >= 2) | ((mags0 > 0.5 * inf) & (mags0 < 1.5 * inf))
    edges = np.where(mask)[0].tolist()
    return edges


def add_bevel(obj, edges, offset=0.03, segments=8):
    with ViewportMode(obj, mode="EDIT"):
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        for edge in bm.edges:
            if edge.index in edges:
                edge.select_set(True)
        bpy.ops.mesh.bevel(
            offset=offset, offset_pct=0, segments=segments, release_confirm=True
        )
    return obj


def complete_bevel(nw, geometry, preprocess):
    inf = 1e5
    geometry = nw.new_node(Nodes.RealizeInstances, [geometry])
    if not preprocess:
        return geometry
    return nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": (geometry, 0),
            "Offset": nw.new_node(
                Nodes.Vector, attrs={"vector": mathutils.Vector((inf, 0, 0))}
            ),
        },
    )


def complete_no_bevel(nw, geometry, preprocess):
    inf = 1e5
    geometry = nw.new_node(Nodes.RealizeInstances, [geometry])
    if not preprocess:
        return geometry
    return nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": (geometry, 0),
            "Offset": nw.new_node(
                Nodes.Vector, attrs={"vector": mathutils.Vector((2 * inf, 0, 0))}
            ),
        },
    )
