# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import logging

import bpy

from infinigen.core.nodes import Nodes

logger = logging.getLogger(__name__)


class CompatOutputProxy:
    """
    Wraps a Blender node returned by new_node() and transparently remaps
    deprecated output-socket names to their Blender 5.x equivalents.

    This lets callers continue using old socket names (e.g. R/G/B from the
    removed ShaderNodeSeparateRGB) even though the actual node now uses
    different names (X/Y/Z for ShaderNodeSeparateXYZ).

    The proxy wraps the node's ``outputs`` collection so that
    ``node.outputs["R"]`` is transparently remapped to ``node.outputs["X"]``.
    """

    def __init__(self, node, output_remap):
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_output_remap", output_remap)

    class _CompatSocketCollection:
        """Wraps bpy_prop_collection to remap socket names on lookup."""

        def __init__(self, collection, output_remap):
            object.__setattr__(self, "_collection", collection)
            object.__setattr__(self, "_output_remap", output_remap)

        def __getitem__(self, key):
            remap = object.__getattribute__(self, "_output_remap")
            key = remap.get(key, key)
            return object.__getattribute__(self, "_collection")[key]

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_collection"), name)

    @property
    def outputs(self):
        """Return a socket-collection wrapper that remaps deprecated names."""
        return self._CompatSocketCollection(
            object.__getattribute__(self, "_node").outputs,
            object.__getattribute__(self, "_output_remap"),
        )

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_node"), name)

    def __repr__(self):
        return f"CompatOutputProxy({object.__getattribute__(self, '_node')})"


def infer_output_socket(item):
    """
    Figure out if `item` somehow represents a node with an output we can use.
    If so, return that output socket
    """

    if isinstance(item, bpy.types.NodeSocket):
        res = item
    elif isnode(item):
        # take the first active socket
        try:
            res = next(o for o in item.outputs if o.enabled)
        except StopIteration as e:
            raise ValueError(
                f"Attempted to get output socket for {item} but none are enabled!"
            ) from e
    elif isinstance(item, tuple) and isnode(item[0]):
        node, socket_name = item
        if isinstance(socket_name, int):
            return node.outputs[socket_name]
        try:
            res = next(o for o in node.outputs if o.enabled and o.name == socket_name)
        except StopIteration as e:
            raise ValueError(
                f"Couldnt find an enabled socket on {node} corresponding to requested tuple {item}"
            ) from e
    else:
        return None

    if not res.enabled:
        raise ValueError(
            f"Attempted to use output socket {res} of node {res.node} which is not enabled. "
            "Please check your attrs are correct, or be more specific about the socket to use"
        )

    return res


def infer_input_socket(node, input_socket_name):
    if (
        node.bl_idname == Nodes.CaptureAttribute
        and input_socket_name != 0
        and input_socket_name != "Geometry"
    ):
        node.capture_items.new("FLOAT", name="Attribute")
    if isinstance(input_socket_name, str):
        try:
            input_socket = next(
                i for i in node.inputs if i.name == input_socket_name and i.enabled
            )
        except StopIteration:
            input_socket = node.inputs[input_socket_name]
    else:
        input_socket = node.inputs[input_socket_name]

    if not input_socket.enabled:
        logger.warning(
            f"Attempted to use ({input_socket.name=},{input_socket.type=}) of {node.name=}, but it was "
            f"disabled. Either change attrs={{...}}, "
            f"change the socket index, or specify the socket by name (assuming two enabled sockets don't "
            f"share a name)."
            f"The input sockets are "
            f"{[(i.name, i.type, ('ENABLED' if i.enabled else 'DISABLED')) for i in node.inputs]}.",
        )

    return input_socket


def isnode(x):
    return isinstance(
        x, bpy.types.ShaderNode | bpy.types.NodeInternal | bpy.types.GeometryNode
    )
