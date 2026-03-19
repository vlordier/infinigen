# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson - Render, flat shading, etc
# - Alex Raistrick - Compositing
# - Hei Law - Initial version


import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import bpy
import gin
import numpy as np
from imageio import imwrite

from infinigen.core import init
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement import camera as cam_util
from infinigen.core.rendering.post_render import (
    colorize_depth,
    colorize_flow,
    colorize_int_array,
    colorize_normals,
    load_depth,
    load_flow,
    load_normals,
    load_seg_mask,
    load_uniq_inst,
)
from infinigen.core.util.blender import set_geometry_option
from infinigen.core.util.logging import Timer
from infinigen.tools.datarelease_toolkit import reorganize_old_framesfolder
from infinigen.tools.suffixes import get_suffix

TRANSPARENT_SHADERS = {Nodes.TranslucentBSDF, Nodes.TransparentBSDF}

logger = logging.getLogger(__name__)


# Blender 5 renamed many Render Layers output sockets from short IDs
# (e.g. DiffDir) to human-readable labels (e.g. Diffuse Direct).
LEGACY_RENDER_SOCKET_REMAP = {
    "DiffDir": "Diffuse Direct",
    "DiffCol": "Diffuse Color",
    "DiffInd": "Diffuse Indirect",
    "GlossDir": "Glossy Direct",
    "GlossCol": "Glossy Color",
    "GlossInd": "Glossy Indirect",
    "TransDir": "Transmission Direct",
    "TransCol": "Transmission Color",
    "TransInd": "Transmission Indirect",
    "VolumeDir": "Volume Direct",
    "Emit": "Emission",
    "Env": "Environment",
    "AO": "Ambient Occlusion",
    "IndexOB": "Object Index",
    "IndexMA": "Material Index",
}

# Fallback socket names by pass identifier when configured socket names do not
# exist in the current Blender version.
PASS_TO_SOCKET_FALLBACKS = {
    "z": ["Depth"],
    "normal": ["Normal"],
    "vector": ["Vector"],
    "object_index": ["Object Index", "IndexOB"],
    "material_index": ["Material Index", "IndexMA"],
    "emit": ["Emission", "Emit"],
    "environment": ["Environment", "Env"],
    "ambient_occlusion": ["Ambient Occlusion", "AO"],
    "diffuse_direct": ["Diffuse Direct", "DiffDir"],
    "diffuse_color": ["Diffuse Color", "DiffCol"],
    "diffuse_indirect": ["Diffuse Indirect", "DiffInd"],
    "glossy_direct": ["Glossy Direct", "GlossDir"],
    "glossy_color": ["Glossy Color", "GlossCol"],
    "glossy_indirect": ["Glossy Indirect", "GlossInd"],
    "transmission_direct": ["Transmission Direct", "TransDir"],
    "transmission_color": ["Transmission Color", "TransCol"],
    "transmission_indirect": ["Transmission Indirect", "TransInd"],
    "volume_direct": ["Volume Direct", "VolumeDir"],
}


def _normalize_socket_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_render_socket(render_layers, viewlayer_pass: str, socket_name: str):
    # 1) direct lookup
    sock = render_layers.outputs.get(socket_name)
    if sock is not None:
        return sock, socket_name

    # 2) explicit legacy remap
    remapped = LEGACY_RENDER_SOCKET_REMAP.get(socket_name)
    if remapped is not None:
        sock = render_layers.outputs.get(remapped)
        if sock is not None:
            return sock, remapped

    # 3) pass-based candidate list
    for candidate in PASS_TO_SOCKET_FALLBACKS.get(viewlayer_pass, []):
        sock = render_layers.outputs.get(candidate)
        if sock is not None:
            return sock, candidate

    # 4) normalized-name match for robustness
    target_norms = {
        _normalize_socket_name(socket_name),
        _normalize_socket_name(remapped) if remapped is not None else "",
        *(_normalize_socket_name(x) for x in PASS_TO_SOCKET_FALLBACKS.get(viewlayer_pass, [])),
    }
    target_norms.discard("")

    for out in render_layers.outputs:
        if _normalize_socket_name(out.name) in target_norms:
            return out, out.name

    return None, None


def _visible_render_objects():
    return tuple(obj for obj in bpy.data.objects if not obj.hide_render)


def remove_translucency():
    # The asserts were added since these edge cases haven't appeared yet -Lahav
    for material in bpy.data.materials:
        nw = NodeWrangler(material.node_tree)
        for node in nw.nodes:
            if node.bl_idname == Nodes.MixShader:
                fac_soc, shader_1_soc, shader_2_soc = node.inputs
                assert shader_1_soc.is_linked and len(shader_1_soc.links) == 1
                assert shader_2_soc.is_linked and len(shader_2_soc.links) == 1
                shader_1_type = shader_1_soc.links[0].from_node.bl_idname
                shader_2_type = shader_2_soc.links[0].from_node.bl_idname
                assert not (
                    shader_1_type in TRANSPARENT_SHADERS
                    and shader_2_type in TRANSPARENT_SHADERS
                )
                if shader_1_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 1.0
                elif shader_2_type in TRANSPARENT_SHADERS:
                    assert not fac_soc.is_linked
                    fac_soc.default_value = 0.0


def set_pass_indices():
    tree_output = {}
    visible_objects = _visible_render_objects()
    visible_names = {obj.name for obj in visible_objects}
    index = 1

    for obj in visible_objects:
        if obj.pass_index == 0:
            obj.pass_index = index
            index += 1

    for obj in visible_objects:
        object_dict = {"type": obj.type, "object_index": obj.pass_index, "children": []}
        if obj.type == "MESH":
            object_dict["num_verts"] = len(obj.data.vertices)
            object_dict["num_faces"] = len(obj.data.polygons)
            object_dict["materials"] = obj.material_slots.keys()
            object_dict["unapplied_modifiers"] = obj.modifiers.keys()
        tree_output[obj.name] = object_dict
        for child_obj in obj.children:
            if child_obj.name in visible_names:
                object_dict["children"].append(child_obj.pass_index)
    return tree_output


def set_material_pass_indices():
    output_material_properties = {}
    mat_index = 1
    for mat in bpy.data.materials:
        if mat.pass_index == 0:
            mat.pass_index = mat_index
            mat_index += 1
        output_material_properties[mat.name] = {"pass_index": mat.pass_index}
    return output_material_properties


# Can be pasted directly into the blender console
def make_clay():
    clay_material = bpy.data.materials.new(name="clay")
    clay_material.diffuse_color = (0.2, 0.05, 0.01, 1)
    for obj in bpy.data.objects:
        if "atmosphere" not in obj.name.lower() and not obj.hide_render:
            if len(obj.material_slots) == 0:
                obj.active_material = clay_material
            else:
                for mat_slot in obj.material_slots:
                    mat_slot.material = clay_material


@gin.configurable
def compositor_postprocessing(
    nw,
    source,
    show=True,
    color_correct=True,
    distort=0,
    glare=False,
):
    if distort > 0:
        source = nw.new_node(
            Nodes.LensDistortion, input_kwargs={"Image": source, "Dispersion": distort}
        )

    if color_correct:
        source = nw.new_node(
            Nodes.BrightContrast,
            input_kwargs={"Image": source, "Bright": 1.0, "Contrast": 4.0},
        )

    if glare:
        source = nw.new_node(
            Nodes.Glare,
            input_kwargs={"Image": source},
            attrs={"glare_type": "GHOSTS", "threshold": 0.5, "mix": -0.99},
        )

    if show:
        nw.new_node(Nodes.Composite, input_kwargs={"Image": source})

    return source.outputs[0] if hasattr(source, "outputs") else source


@gin.configurable
def configure_compositor_output(
    nw,
    frames_folder,
    image_denoised,
    image_noisy,
    passes_to_save,
    saving_ground_truth,
    ground_truth_image_name="UniqueInstances",
):
    def _new_output_node(file_format: str):
        node = nw.new_node(Nodes.OutputFile)
        if hasattr(node, "base_path"):
            node.base_path = str(frames_folder)
        if hasattr(node, "directory"):
            node.directory = str(frames_folder)
        try:
            node.format.file_format = file_format
        except TypeError:
            supported_formats = {
                item.identifier
                for item in node.format.bl_rna.properties["file_format"].enum_items
            }
            if "OPEN_EXR_MULTILAYER" in supported_formats:
                node.format.file_format = "OPEN_EXR_MULTILAYER"
            elif "OPEN_EXR" in supported_formats:
                node.format.file_format = "OPEN_EXR"
            else:
                node.format.file_format = next(iter(supported_formats))
        node.format.color_mode = "RGB"
        return node

    file_output_node_png = _new_output_node("PNG")
    file_output_node_exr = _new_output_node("OPEN_EXR")
    default_file_output_node = (
        file_output_node_exr if saving_ground_truth else file_output_node_png
    )
    output_name_targets = []
    use_legacy_file_slots = hasattr(file_output_node_png, "file_slots")
    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    render_layers = nw.new_node(Nodes.RenderLayers)

    def _create_output_input(node, socket_name: str, socket_type: str | None = None):
        if use_legacy_file_slots:
            slot_input = node.file_slots.new(socket_name)
            output_name_targets.append((socket_name, node.file_slots[slot_input.name]))
            return slot_input

        socket_types = []
        if socket_type is not None:
            socket_types.append(socket_type)
        socket_types.extend(["RGBA", "VECTOR", "FLOAT"])
        for socket_type in socket_types:
            try:
                item = node.file_output_items.new(socket_type, socket_name)
                output_name_targets.append((socket_name, node))
                return node.inputs[item.name]
            except TypeError:
                continue
        raise RuntimeError(
            f"Failed creating compositor output item for {socket_name=} in Blender API"
        )

    for viewlayer_pass, socket_name in passes_to_save:
        if hasattr(viewlayer, f"use_pass_{viewlayer_pass}"):
            setattr(viewlayer, f"use_pass_{viewlayer_pass}", True)
        else:
            setattr(viewlayer.cycles, f"use_pass_{viewlayer_pass}", True)
        # must save the material pass index as EXR
        file_output_node = (
            default_file_output_node
            if viewlayer_pass != "material_index"
            else file_output_node_exr
        )

        if not use_legacy_file_slots:
            # Blender 5+ has node-level file naming (no per-slot path), so use one
            # output node per pass to preserve distinct pass file names.
            file_output_node = _new_output_node(
                "OPEN_EXR"
                if (saving_ground_truth or viewlayer_pass == "material_index")
                else "PNG"
            )

        render_socket, resolved_socket_name = _resolve_render_socket(
            render_layers, viewlayer_pass, socket_name
        )
        if render_socket is None:
            logger.warning(
                "Skipping unavailable render pass output socket %s for viewlayer pass %s",
                socket_name,
                viewlayer_pass,
            )
            continue

        if resolved_socket_name != socket_name:
            logger.info(
                "Remapped render pass socket %s -> %s for viewlayer pass %s",
                socket_name,
                resolved_socket_name,
                viewlayer_pass,
            )

        output_socket_type = render_socket.type
        output_item_type = {
            "RGBA": "RGBA",
            "VECTOR": "VECTOR",
            "VALUE": "FLOAT",
        }.get(output_socket_type, "FLOAT")

        if viewlayer_pass in {"object_index", "material_index"}:
            # Blender 5 EEVEE may not emit scalar index passes reliably through
            # modern OutputFile VALUE/FLOAT items. Pack them into RGB channels
            # so the compositor writes a concrete EXR, then decode from channel 0.
            output_item_type = "RGBA"

        slot_input = _create_output_input(
            file_output_node, socket_name, socket_type=output_item_type
        )
        match viewlayer_pass:
            case "vector":
                separate_color = nw.new_node(Nodes.CompSeparateColor, [render_socket])
                comnbine_color = nw.new_node(
                    Nodes.CompCombineColor,
                    [0, (separate_color, 3), (separate_color, 2), 0],
                )
                nw.links.new(comnbine_color.outputs[0], slot_input)
            case "normal":
                try:
                    color = nw.new_node(
                        Nodes.CompositorMixRGB,
                        [None, render_socket, (0, 0, 0, 0)],
                        attrs={"blend_type": "ADD"},
                    ).outputs[0]
                    nw.links.new(color, slot_input)
                except RuntimeError:
                    logger.debug(
                        "Compositor mix node unavailable; linking normal pass directly"
                    )
                    nw.links.new(render_socket, slot_input)
            case "object_index" | "material_index":
                packed_index = nw.new_node(
                    Nodes.CompCombineColor,
                    input_kwargs={"Alpha": 1.0},
                    attrs={"mode": "RGB"},
                )
                nw.links.new(render_socket, packed_index.inputs["Red"])
                nw.links.new(render_socket, packed_index.inputs["Green"])
                nw.links.new(render_socket, packed_index.inputs["Blue"])
                nw.links.new(packed_index.outputs[0], slot_input)
            case _:
                nw.links.new(render_socket, slot_input)

    image = image_denoised if image_denoised is not None else image_noisy
    if saving_ground_truth:
        if use_legacy_file_slots:
            nw.links.new(image, default_file_output_node.inputs["Image"])
            default_file_output_node.file_slots["Image"].path = ground_truth_image_name
            output_name_targets.append(
                (ground_truth_image_name, default_file_output_node.file_slots["Image"])
            )
        else:
            ground_truth_node = _new_output_node("OPEN_EXR")
            slot_input = _create_output_input(
                ground_truth_node,
                ground_truth_image_name,
                socket_type="RGBA",
            )
            nw.links.new(image, slot_input)
    else:
        if use_legacy_file_slots:
            nw.links.new(image, default_file_output_node.inputs["Image"])
            nw.links.new(image, file_output_node_exr.inputs["Image"])
            output_name_targets.append(("Image", file_output_node_exr.file_slots["Image"]))
            output_name_targets.append(
                ("Image", default_file_output_node.file_slots["Image"])
            )
        else:
            png_image_node = _new_output_node("PNG")
            exr_image_node = _new_output_node("OPEN_EXR")
            nw.links.new(image, _create_output_input(png_image_node, "Image"))
            nw.links.new(image, _create_output_input(exr_image_node, "Image"))

    return output_name_targets


def shader_random(nw: NodeWrangler):
    # Code generated using version 2.4.3 of the node_transpiler

    object_info = nw.new_node(Nodes.ObjectInfo_Shader)

    white_noise_texture = nw.new_node(
        Nodes.WhiteNoiseTexture, input_kwargs={"Vector": object_info.outputs["Random"]}
    )
    emission = nw.new_node(
        Nodes.Emission,
        input_kwargs={"Color": white_noise_texture.outputs["Color"]},
    )

    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": emission},
    )


def _replace_shader_with_randcolor(material: bpy.types.Material):
    nt = material.node_tree
    if nt is None:
        return
    logger.debug(f"Replacing shader with randcolor for {material.name}")
    nodes = nt.nodes
    object_info = nodes.new(type="ShaderNodeObjectInfo")
    emission = nodes.new(type="ShaderNodeEmission")
    material_output = nodes["Material Output"]
    # Use object display color directly so each object gets a stable flat color,
    # even when sharing materials.
    nt.links.new(object_info.outputs["Color"], emission.inputs["Color"])
    nt.links.new(emission.outputs[0], material_output.inputs["Surface"])


def _replace_shader_with_object_index(material: bpy.types.Material):
    """Replace material with a flat white emission.

    Object indices are no longer encoded via per-material shader emission.
    Instead, a dedicated EEVEE render pass uses compositor IDMask nodes
    to read pass_index values directly from the scene, bypassing the
    material layer entirely.
    """
    nt = material.node_tree
    if nt is None:
        return
    nodes = nt.nodes
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material_output = nodes["Material Output"]
    nt.links.new(emission.outputs[0], material_output.inputs["Surface"])


def _remove_volume_shading(material: bpy.types.Material):
    nt = material.node_tree
    if nt is None:
        return
    nw = NodeWrangler(nt)
    for output in nw.find(Nodes.MaterialOutput):
        if "Volume" not in output.inputs:
            continue
        vol_socket = output.inputs["Volume"]
        if len(vol_socket.links) > 0:
            nw.links.remove(vol_socket.links[0])


def _replace_materials_with_flat_shading(
    obj: bpy.types.Object, mode: Literal["random", "object_index"] = "random"
):
    if mode == "random":
        # Seed object display color from pass_index for deterministic, per-object
        # flat colors that do not depend on textures or shader randomness.
        rng = np.random.default_rng(int(obj.pass_index))
        obj.color = (*rng.uniform(0.1, 0.9, 3), 1.0)

    shader_replacer = {
        "random": _replace_shader_with_randcolor,
        "object_index": _replace_shader_with_object_index,
    }[mode]
    for i in range(len(obj.material_slots)):
        if obj.material_slots[i] is None or obj.material_slots[i].material is None:
            logger.debug(
                f"Skipping {obj.name} with empty material slot {i}/{len(obj.material_slots)}"
            )
            continue
        try:
            shader_replacer(obj.material_slots[i].material)
        except Exception as e:
            mat = obj.material_slots[i].material
            raise RuntimeError(
                f"Error in blendergt flat_shading {shader_replacer.__name__} for "
                f"{obj.name} with material slot {i} {mat.name}: {e}"
            ) from e


def _assign_atmosphere_flat_material(obj):
    """Assign a pure-white emission material to the given object for flat annotation."""
    mat = bpy.data.materials.new(name="flat_atmosphere")
    mat.use_nodes = True
    nw = NodeWrangler(mat.node_tree)
    emission = nw.new_node(
        Nodes.Emission,
        input_kwargs={"Color": (1.0, 1.0, 1.0, 1.0), "Strength": 1.0},
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": emission})
    obj.active_material = mat


def global_flat_shading(mode: Literal["random", "object_index"] = "random"):
    # Remove all volumes in the scene as they cause noisy depth and unstable
    # segmentation colors under EEVEE flat shading.
    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if "fire_system_type" in obj and obj["fire_system_type"] == "volume":
            continue
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META"}:
            continue
        if obj.active_material is None:
            continue
        try:
            _remove_volume_shading(obj.active_material)
        except Exception as e:
            mat = obj.active_material
            raise RuntimeError(
                f"Error in blendergt flat_shading {_remove_volume_shading.__name__} for "
                f"{obj.name} with material {mat.name}: {e}"
            ) from e

    bpy.context.view_layer.update()

    # Get rid of all nondiffuse materials. e.g. glass becomes solid, or else we get noisy depth (as of bl3.6 at least)
    for obj in bpy.context.scene.view_layers["ViewLayer"].objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META"}:
            continue

        if mode == "object_index" and obj.name.lower() in {"atmosphere", "atmosphere_fine"}:
            # Exclude atmosphere shell from object-index render; otherwise it can
            # occlude the full view and collapse IDs to a single label.
            obj.hide_render = True
            continue

        obj.hide_viewport = False
        if "fire_system_type" in obj and obj["fire_system_type"] == "gt_mesh":
            obj.hide_viewport = False
            obj.hide_render = False
        if (
            not hasattr(obj, "material_slots")
            or obj.material_slots is None
            or len(obj.material_slots) == 0
        ):
            # Objects with no material slots (e.g. atmosphere) get a white emission
            # material so they still contribute to segmentation.
            if mode == "object_index":
                _assign_atmosphere_flat_material(obj)
                logger.info("Assigned white emission to %s type=%s", obj.name, obj.type)
            continue
        _replace_materials_with_flat_shading(obj, mode=mode)

    nw = NodeWrangler(bpy.data.worlds["World"].node_tree)
    for link in nw.links:
        nw.links.remove(link)


def postprocess_blendergt_outputs(frames_folder, output_stem, camera):
    uniq_inst_array = None

    # Save flow visualization
    flow_dst_path = frames_folder / f"Vector{output_stem}.exr"
    if flow_dst_path.is_file():
        try:
            flow_array = load_flow(flow_dst_path)
            np.save(flow_dst_path.with_name(f"Flow{output_stem}.npy"), flow_array)

            flow_color = colorize_flow(flow_array)
            if flow_color is not None:
                imwrite(
                    flow_dst_path.with_name(f"Flow{output_stem}.png"),
                    flow_color,
                )
            flow_dst_path.unlink()
        except Exception as e:
            logger.warning("Skipping flow postprocess for %s: %s", flow_dst_path, e)
    else:
        logger.warning("Missing flow pass output: %s", flow_dst_path)

    # Save surface normal visualization
    normal_dst_path = frames_folder / f"Normal{output_stem}.exr"
    if normal_dst_path.is_file():
        try:
            normal_array = load_normals(normal_dst_path, camera)
            np.save(
                frames_folder / f"SurfaceNormal{output_stem}.npy", normal_array
            )
            imwrite(
                frames_folder / f"SurfaceNormal{output_stem}.png",
                colorize_normals(normal_array),
            )
            normal_dst_path.unlink()
        except Exception as e:
            logger.warning("Skipping normal postprocess for %s: %s", normal_dst_path, e)
    else:
        logger.warning("Missing normal pass output: %s", normal_dst_path)

    # Save depth visualization
    depth_dst_path = frames_folder / f"Depth{output_stem}.exr"
    if depth_dst_path.is_file():
        try:
            depth_array = load_depth(depth_dst_path)
            np.save(frames_folder / f"Depth{output_stem}.npy", depth_array)
            imwrite(
                depth_dst_path.with_name(f"Depth{output_stem}.png"),
                colorize_depth(depth_array),
            )
            depth_dst_path.unlink()
        except Exception as e:
            logger.warning("Skipping depth postprocess for %s: %s", depth_dst_path, e)
    else:
        logger.warning("Missing depth pass output: %s", depth_dst_path)

    # Save segmentation visualization
    seg_dst_path = frames_folder / f"IndexOB{output_stem}.exr"
    if seg_dst_path.is_file():
        try:
            seg_mask_array = load_seg_mask(seg_dst_path)
            np.save(
                frames_folder / f"ObjectSegmentation{output_stem}.npy", seg_mask_array
            )
            imwrite(
                seg_dst_path.with_name(f"ObjectSegmentation{output_stem}.png"),
                colorize_int_array(seg_mask_array),
            )
            seg_dst_path.unlink()
        except Exception as e:
            logger.warning(
                "Skipping object segmentation postprocess for %s: %s", seg_dst_path, e
            )
    else:
        logger.warning("Missing object index pass output: %s", seg_dst_path)

    # Save unique instances visualization
    uniq_inst_path = frames_folder / f"UniqueInstances{output_stem}.exr"
    if uniq_inst_path.is_file():
        try:
            if uniq_inst_array is None:
                uniq_inst_array = load_uniq_inst(uniq_inst_path)
            np.save(
                frames_folder / f"InstanceSegmentation{output_stem}.npy",
                uniq_inst_array,
            )
            imwrite(
                uniq_inst_path.with_name(f"InstanceSegmentation{output_stem}.png"),
                colorize_int_array(uniq_inst_array),
            )
            uniq_inst_path.unlink()
        except Exception as e:
            logger.warning(
                "Skipping instance segmentation postprocess for %s: %s",
                uniq_inst_path,
                e,
            )
    else:
        logger.warning("Missing unique instance pass output: %s", uniq_inst_path)


def postprocess_materialgt_output(frames_folder, output_stem):
    # Save material segmentation visualization if present
    ma_seg_dst_path = frames_folder / f"IndexMA{output_stem}.exr"
    if ma_seg_dst_path.is_file():
        ma_seg_mask_array = load_seg_mask(ma_seg_dst_path)
        np.save(
            ma_seg_dst_path.with_name(f"MaterialSegmentation{output_stem}.npy"),
            ma_seg_mask_array,
        )
        imwrite(
            ma_seg_dst_path.with_name(f"MaterialSegmentation{output_stem}.png"),
            colorize_int_array(ma_seg_mask_array),
        )
        ma_seg_dst_path.unlink()


def _get_compositor_node_tree():
    """Return the compositor node tree, compatible with Blender 5+ API."""
    scene = bpy.context.scene
    # Blender 5+: scene.node_tree removed; use compositing_node_group
    if hasattr(scene, "compositing_node_group"):
        ng = scene.compositing_node_group
        if ng is None:
            scene.use_nodes = True
            ng = scene.compositing_node_group
            if ng is None:
                ng = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
                scene.compositing_node_group = ng
        return ng
    # Blender 4.x fallback
    if hasattr(scene, "node_tree"):
        if scene.node_tree is None:
            scene.use_nodes = True
        return scene.node_tree
    raise AttributeError(
        "Scene has neither 'node_tree' nor 'compositing_node_group'; "
        "unsupported Blender version for compositor setup."
    )


def _reset_compositor_node_tree():
    compositor_node_tree = _get_compositor_node_tree()
    for node in list(compositor_node_tree.nodes):
        compositor_node_tree.nodes.remove(node)
    return compositor_node_tree


def configure_compositor(
    frames_folder: Path,
    passes_to_save: list,
    flat_shading: bool,
    ground_truth_image_name="UniqueInstances",
):
    compositor_node_tree = _reset_compositor_node_tree()
    nw = NodeWrangler(compositor_node_tree)

    render_layers = nw.new_node(Nodes.RenderLayers)
    if flat_shading:
        final_image_denoised = render_layers.outputs["Image"]
        final_image_noisy = None
    else:
        final_image_denoised = compositor_postprocessing(
            nw, source=render_layers.outputs["Image"]
        )

        final_image_noisy = (
            compositor_postprocessing(
                nw, source=render_layers.outputs["Noisy Image"], show=False
            )
            if bpy.context.scene.cycles.use_denoising
            else None
        )

    return configure_compositor_output(
        nw,
        frames_folder,
        image_denoised=final_image_denoised,
        image_noisy=final_image_noisy,
        passes_to_save=passes_to_save,
        saving_ground_truth=flat_shading,
        ground_truth_image_name=ground_truth_image_name,
    )


def _apply_output_names(output_name_targets, indices):
    fileslot_suffix = get_suffix({"frame": "####", **indices})
    for output_name, output_target in output_name_targets:
        target_name = f"{output_name}{fileslot_suffix}"
        if hasattr(output_target, "path"):
            output_target.path = target_name
        elif hasattr(output_target, "file_name"):
            output_target.file_name = target_name


def _render_eevee_object_index_pass(frames_folder: Path, indices: dict):
    with Timer("Object Index Flat Shading"):
        global_flat_shading(mode="object_index")

    # Enable the object-index viewlayer pass for EEVEE
    viewlayer = bpy.context.scene.view_layers["ViewLayer"]
    viewlayer.use_pass_object_index = True

    output_name_targets = configure_compositor(
        frames_folder,
        [],
        flat_shading=True,
        ground_truth_image_name="IndexOB",
    )
    _apply_output_names(output_name_targets, indices)

    with Timer("Object Index Rendering"):
        bpy.ops.render.render(animation=True)


def _unlink_material_displacement_output(material: bpy.types.Material):
    if material.node_tree is None:
        return
    nw = NodeWrangler(material.node_tree)
    material_outputs = nw.find(Nodes.MaterialOutput)
    for output_node in material_outputs:
        if "Displacement" not in output_node.inputs:
            continue
        displacement_input = output_node.inputs["Displacement"]
        for link in displacement_input.links:
            logger.debug(
                f"{_unlink_material_displacement_output.__name__} removing {link} to {output_node.name} in {material.name}"
            )
            nw.links.remove(link)


@gin.configurable
def set_displacement_mode(
    displacement_mode: Literal["DISPLACEMENT", "BUMP", "BOTH", "NONE"] = "DISPLACEMENT",
):
    match displacement_mode:
        case "NONE":
            for material in bpy.data.materials:
                _unlink_material_displacement_output(material)
        case "DISPLACEMENT" | "BUMP" | "BOTH":
            for material in bpy.data.materials:
                set_geometry_option(material, displacement_mode)
        case _:
            raise ValueError(f"Invalid displacement mode: {displacement_mode}")


@gin.configurable
def render_image(
    camera: bpy.types.Object,
    frames_folder,
    passes_to_save,
    render_resolution_override=None,
    excludes=[],
    use_dof=False,
    dof_aperture_fstop=2.8,
    flat_shading=False,
    override_num_samples=None,
    use_eevee_next_for_annotations=False,
    enable_render_time_pass=False,
    force_eevee=False,
):
    tic = time.time()

    for exclude in excludes:
        bpy.data.objects[exclude].hide_render = True

    using_eevee = force_eevee or (flat_shading and use_eevee_next_for_annotations)
    if using_eevee:
        init.configure_eevee_next()
        logger.info("Render engine: BLENDER_EEVEE_NEXT (force_eevee=%s, flat_shading=%s)", force_eevee, flat_shading)
    else:
        init.configure_cycles_devices()
        logger.info("Render engine: CYCLES (force_eevee=%s, flat_shading=%s)", force_eevee, flat_shading)
    set_displacement_mode()

    # Inject the Blender 5.0 Render Time pass when requested.  This is a
    # Cycles-only pass (not available under EEVEE) that records per-pixel
    # render cost as a greyscale EXR layer.
    active_passes = list(passes_to_save)
    if enable_render_time_pass and not using_eevee:
        # Enable the viewlayer pass so Cycles populates the RenderTime socket
        init.configure_render_time_pass(enabled=True)
        # Inject the descriptor only if the pass is not already listed,
        # comparing by pass name (first element) to handle both list and
        # tuple entries robustly.
        rt_pass_name = init.RENDER_TIME_PASS_DESCRIPTOR[0]
        if rt_pass_name not in (p[0] for p in active_passes):
            active_passes.append(list(init.RENDER_TIME_PASS_DESCRIPTOR))

    main_active_passes = list(active_passes)
    if using_eevee:
        main_active_passes = [p for p in active_passes if p[0] != "object_index"]

    tmp_dir = frames_folder.parent.resolve() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    bpy.context.scene.render.filepath = f"{tmp_dir}{os.sep}"

    camrig_id, subcam_id = cam_util.get_id(camera)

    if override_num_samples is not None and not using_eevee:
        # override_num_samples is a Cycles-only property; skip for EEVEE
        bpy.context.scene.cycles.samples = override_num_samples

    if flat_shading:
        with Timer("Set object indices"):
            object_data = set_pass_indices()
            json_object = json.dumps(object_data, indent=4)
            first_frame = bpy.context.scene.frame_start
            suffix = get_suffix(
                dict(
                    cam_rig=camrig_id,
                    resample=0,
                    frame=first_frame,
                    subcam=subcam_id,
                )
            )
            (frames_folder / f"Objects{suffix}.json").write_text(json_object)

        with Timer("Flat Shading"):
            global_flat_shading(mode="random")
    else:
        segment_materials = "material_index" in (x[0] for x in active_passes)
        if segment_materials:
            with Timer("Set material indices"):
                material_data = set_material_pass_indices()
                json_object = json.dumps(material_data, indent=4)
                first_frame = bpy.context.scene.frame_start
                suffix = get_suffix(
                    dict(
                        cam_rig=camrig_id,
                        resample=0,
                        frame=first_frame,
                        subcam=subcam_id,
                    )
                )
                (frames_folder / f"Materials{suffix}.json").write_text(json_object)

    if not bpy.context.scene.use_nodes:
        bpy.context.scene.use_nodes = True
    output_name_targets = configure_compositor(
        frames_folder,
        main_active_passes,
        flat_shading,
        ground_truth_image_name="UniqueInstances",
    )

    indices = dict(cam_rig=camrig_id, resample=0, subcam=subcam_id)

    ## Update output names
    _apply_output_names(output_name_targets, indices)

    if use_dof == "IF_TARGET_SET":
        use_dof = camera.data.dof.focus_object is not None
    elif use_dof is not None:
        camera.data.dof.use_dof = use_dof
        camera.data.dof.aperture_fstop = dof_aperture_fstop

    if render_resolution_override is not None:
        bpy.context.scene.render.resolution_x = render_resolution_override[0]
        bpy.context.scene.render.resolution_y = render_resolution_override[1]

    # Render the scene
    bpy.context.scene.camera = camera
    with Timer("Actual rendering"):
        bpy.ops.render.render(animation=True)

    if using_eevee and any(p[0] == "object_index" for p in active_passes):
        _render_eevee_object_index_pass(frames_folder, indices)

    with Timer("Post Processing"):
        for frame in range(
            bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1
        ):
            if flat_shading:
                bpy.context.scene.frame_set(frame)
                suffix = get_suffix(dict(frame=frame, **indices))
                postprocess_blendergt_outputs(frames_folder, suffix, camera)
            else:
                cam_util.save_camera_parameters(
                    camera,
                    output_folder=frames_folder,
                    frame=frame,
                )
                bpy.context.scene.frame_set(frame)
                suffix = get_suffix(dict(frame=frame, **indices))
                postprocess_materialgt_output(frames_folder, suffix)

    for file in tmp_dir.glob("*.png"):
        file.unlink()

    reorganize_old_framesfolder(frames_folder)

    logger.info(f"rendering time: {time.time() - tic}")
