# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import math

import bmesh
import numpy as np
from numpy.random import normal, uniform


def reaction_diffusion(
    obj,
    weight_fn,
    steps=1000,
    dt=1.0,
    scale=0.5,
    diff_a=0.18,
    diff_b=0.09,
    feed_rate=0.055,
    kill_rate=0.062,
    perturb=0.05,
):
    diff_a = diff_a * scale
    diff_b = diff_b * scale
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    n = len(bm.verts)
    a = np.ones(n)
    coords = np.empty((n, 3))
    for i, v in enumerate(bm.verts):
        coords[i] = v.co
    b = weight_fn(coords)
    edge_from = np.array([e.verts[0].index for e in bm.edges])
    edge_to = np.array([e.verts[1].index for e in bm.edges])
    bm.free()
    kill_feed = kill_rate + feed_rate
    for _ in range(steps):
        a_msg = a[edge_to] - a[edge_from]
        b_msg = b[edge_to] - b[edge_from]
        lap_a = np.bincount(edge_from, a_msg, n) - np.bincount(edge_to, a_msg, n)
        lap_b = np.bincount(edge_from, b_msg, n) - np.bincount(edge_to, b_msg, n)
        ab2 = a * b * b
        a += (diff_a * lap_a - ab2 + feed_rate * (1 - a)) * dt
        b += (diff_b * lap_b + ab2 - kill_feed * b) * dt

    a_msg = a[edge_to] - a[edge_from]
    b_msg = b[edge_to] - b[edge_from]
    lap_a = np.bincount(edge_from, a_msg, n) - np.bincount(edge_to, a_msg, n)
    lap_b = np.bincount(edge_from, b_msg, n) - np.bincount(edge_to, b_msg, n)

    a *= 1 + normal(0, perturb, n)
    b *= 1 + normal(0, perturb, n)
    lap_a *= 1 + normal(0, perturb, n)
    lap_a *= 1 + normal(0, perturb, n)

    vg_a = obj.vertex_groups.new(name="A")
    vg_b = obj.vertex_groups.new(name="B")
    vg_la = obj.vertex_groups.new(name="LA")
    vg_lb = obj.vertex_groups.new(name="LB")
    deform_bm = bmesh.new()
    deform_bm.from_mesh(obj.data)
    deform_bm.verts.ensure_lookup_table()
    deform_layer = deform_bm.verts.layers.deform.verify()
    for vg, vals in ((vg_la, lap_a), (vg_lb, lap_b), (vg_a, a), (vg_b, b)):
        gi = vg.index
        for i in range(n):
            deform_bm.verts[i][deform_layer][gi] = float(vals[i])
    deform_bm.to_mesh(obj.data)
    deform_bm.free()
    obj.data.update()


def feed2kill(feed):
    return math.sqrt(feed) / 2 - feed


def make_periodic_weight_fn(n_instances, stride=0.1):
    def periodic_weight_fn(coords):
        multiplier = uniform(20, 100, (1, n_instances))
        center = coords[np.random.randint(0, len(coords) - 1, n_instances)]
        phi = (np.expand_dims(coords, 1) * np.expand_dims(center, 0)).sum(
            -1
        ) * multiplier
        measure = np.cos(phi).sum(-1) / math.sqrt(n_instances)
        return (np.abs(measure) < stride).astype(float)

    return periodic_weight_fn
