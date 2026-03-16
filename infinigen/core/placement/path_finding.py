# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import mathutils
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def camera_rotation_matrix(pointing_direction, up_vector):
    forward = pointing_direction / np.linalg.norm(pointing_direction)
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return np.column_stack((right, up, forward))


def path_finding(
    bvhtree, bounding_box, start_pose, end_pose, resolution=100000, margin=0.1
):
    volume = np.prod(bounding_box[1] - bounding_box[0])
    N = np.floor(
        (bounding_box[1] - bounding_box[0]) * (resolution / volume) ** (1 / 3)
    ).astype(np.int32)
    NN = np.prod(N)
    # print(f"{N=}")
    start_location, _start_rotation = start_pose
    end_location, _end_rotation = end_pose
    margin_d = np.ceil((resolution / volume) ** (1 / 3) * margin)
    row = []
    col = []
    data = []
    pi2 = 2.0 * np.pi

    def freespace_ray_check(a, b, margin=0):
        v = b - a
        v_len = v.length
        location, *_ = bvhtree.ray_cast(a, v, v_len)
        if location is not None:
            return False
        if margin != 0:
            if v[0] != 0:
                perp = mathutils.Vector([v[1], -v[0], 0])
            else:
                perp = mathutils.Vector([0, v[2], -v[1]])
            offset = v.cross(perp)
            offset *= margin / offset.length
            check_N = 10
            angle = pi2 / check_N
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            for _i in range(check_N):
                location, *_ = bvhtree.ray_cast(a + offset, v, v_len)
                if location is not None:
                    return False
                tar_direction = offset.cross(v)
                tar_direction *= margin / tar_direction.length
                offset = offset * cos_a + tar_direction * sin_a
        return True

    def index(i, j, k):
        return i * N[1] * N[2] + j * N[2] + k

    x, y, z = np.meshgrid(
        np.arange(N[0]), np.arange(N[1]), np.arange(N[2]), indexing="ij"
    )
    x = (
        bounding_box[0][0]
        + (bounding_box[1][0] - bounding_box[0][0]) * (x + 0.5) / N[0]
    )
    y = (
        bounding_box[0][1]
        + (bounding_box[1][1] - bounding_box[0][1]) * (y + 0.5) / N[1]
    )
    z = (
        bounding_box[0][2]
        + (bounding_box[1][2] - bounding_box[0][2]) * (z + 0.5) / N[2]
    )
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

    start_index = index(
        *np.floor(
            (np.array(start_location) - bounding_box[0])
            / (bounding_box[1] - bounding_box[0])
            * N
        ).astype(np.int32)
    )
    end_index = index(
        *np.floor(
            (np.array(end_location) - bounding_box[0])
            / (bounding_box[1] - bounding_box[0])
            * N
        ).astype(np.int32)
    )
    if end_index == start_index:
        return None

    x[start_index] = start_pose[0].x
    y[start_index] = start_pose[0].y
    z[start_index] = start_pose[0].z
    x[end_index] = end_pose[0].x
    y[end_index] = end_pose[0].y
    z[end_index] = end_pose[0].z

    penalty = 99
    # Pre-compute all valid neighbor pairs vectorized
    gi, gj, gk = np.meshgrid(
        np.arange(N[0]), np.arange(N[1]), np.arange(N[2]), indexing="ij"
    )
    gi, gj, gk = gi.ravel(), gj.ravel(), gk.ravel()
    offsets = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [0, 1, 1], [1, 0, 1],
        [1, -1, 0], [0, 1, -1], [1, 0, -1],
    ])
    for di, dj, dk in offsets:
        ni, nj, nk = gi + di, gj + dj, gk + dk
        valid = (ni >= 0) & (nj >= 0) & (nk >= 0) & (ni < N[0]) & (nj < N[1]) & (nk < N[2])
        src_idx = (gi[valid] * N[1] * N[2] + gj[valid] * N[2] + gk[valid]).astype(int)
        dst_idx = (ni[valid] * N[1] * N[2] + nj[valid] * N[2] + nk[valid]).astype(int)
        weight = penalty if dk != 0 else 1
        for s, d in zip(src_idx, dst_idx):
            pos_from = mathutils.Vector([x[s], y[s], z[s]])
            pos_to = mathutils.Vector([x[d], y[d], z[d]])
            if freespace_ray_check(pos_from, pos_to):
                row.append(s)
                col.append(d)
                data.append(weight)
                row.append(d)
                col.append(s)
                data.append(weight)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    A = csr_matrix((data, (row, col)), shape=(NN, NN))
    G = nx.from_scipy_sparse_array(A)

    n_neighbors = A.sum(axis=0).A1
    boundaries = np.where(n_neighbors != 8 + 10 * penalty)[0].tolist()

    lengths_dict = nx.multi_source_dijkstra_path_length(G, boundaries, weight="weight")
    lengths = np.full(NN, np.inf)
    idx = np.fromiter(lengths_dict.keys(), dtype=int, count=len(lengths_dict))
    vals = np.fromiter(lengths_dict.values(), dtype=float, count=len(lengths_dict))
    lengths[idx] = vals

    mask1 = lengths[row] >= margin_d
    mask2 = lengths[col] >= margin_d
    row = row[mask1 & mask2]
    col = col[mask1 & mask2]
    data = data[mask1 & mask2]

    A = csr_matrix((data, (row, col)), shape=(NN, NN))
    G = nx.from_scipy_sparse_array(A)

    try:
        path = nx.shortest_path(G, start_index, end_index, weight="weight")
    except Exception:
        return None

    stack = [start_index]

    for p in path[1:]:
        back = 0
        target = mathutils.Vector([x[p], y[p], z[p]])
        while freespace_ray_check(
            mathutils.Vector(
                [x[stack[-1 - back]], y[stack[-1 - back]], z[stack[-1 - back]]]
            ),
            target,
            margin=margin,
        ):
            back += 1
            if back == len(stack):
                break
        if back != 1:
            stack = stack[: 1 - back]
        stack.append(p)

    locations = []
    lengths = []
    for i, p in enumerate(stack):
        if i == 0:
            locations.append(start_pose[0])
        elif i == len(stack) - 1:
            locations.append(end_pose[0])
        else:
            locations.append(mathutils.Vector([x[p], y[p], z[p]]))
        if len(locations) >= 2:
            lengths.append((locations[-1] - locations[-2]).length)
    keyframed_poses = []
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(lengths)))

    for i in range(len(stack)):
        if i == 0:
            keyframed_poses.append((0, *start_pose))
        else:
            if i == len(stack) - 1:
                rotation_euler = end_pose[1]
            else:
                rotation_matrix = mathutils.Matrix(
                    camera_rotation_matrix(
                        np.array(locations[i] - locations[i - 1]), np.array([0, 0, 1])
                    )
                ) @ mathutils.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                rotation_euler = rotation_matrix.to_euler()
                if rotation_euler.y != 0:
                    rotation_euler.y = 0
                    rotation_euler.x += np.pi
                    rotation_euler.z += np.pi
            angle_differece = [
                abs(rotation_euler.z - pi2 - keyframed_poses[i - 1][2].z),
                abs(rotation_euler.z - keyframed_poses[i - 1][2].z),
                abs(rotation_euler.z + pi2 - keyframed_poses[i - 1][2].z),
            ]
            rotation_euler.z += (np.argmin(angle_differece) - 1) * pi2
            keyframed_poses.append((float(cumulative_lengths[i]), locations[i], rotation_euler))
    return keyframed_poses
