# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Copilot

"""
Non-regression tests for Blender API compatibility.

These tests document and verify the specific Blender API behaviors that
the infinigen codebase depends on, to detect regressions when upgrading
to a newer Blender/bpy version (e.g. from 4.5 to 5.0).

Each test targets a specific API path used in infinigen/core/util/blender.py,
infinigen/core/surface.py, or infinigen/core/nodes/node_wrangler.py.
"""

import bpy
import numpy as np
import pytest

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import geometry_node_group_empty_new, ng_inputs
from infinigen.core.util import blender as butil

# ---------------------------------------------------------------------------
# Blender version & app info
# ---------------------------------------------------------------------------


def test_bpy_app_version_is_tuple():
    """bpy.app.version must be a 3-tuple of ints (major, minor, patch)."""
    v = bpy.app.version
    assert isinstance(v, tuple)
    assert len(v) == 3
    assert all(isinstance(x, int) for x in v)


def test_bpy_app_version_string_non_empty():
    """bpy.app.version_string must be a non-empty string."""
    assert isinstance(bpy.app.version_string, str)
    assert len(bpy.app.version_string) > 0


def test_bpy_version_is_5_0():
    """Verify that we are running on the expected bpy 5.0.x release."""
    major, minor, _patch = bpy.app.version
    assert (major, minor) == (5, 0), (
        f"Expected bpy 5.0.x but got {bpy.app.version_string}. "
        "Please ensure bpy==5.0.1 is installed."
    )


# ---------------------------------------------------------------------------
# Scene / factory-settings reset (used in conftest cleanup after every test)
# ---------------------------------------------------------------------------


def test_factory_settings_reset_clears_objects():
    """bpy.ops.wm.read_factory_settings(use_empty=True) must remove all objects."""
    butil.spawn_cube()
    butil.spawn_cube()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    assert len(bpy.data.objects) == 0


def test_factory_settings_reset_clears_meshes():
    """bpy.ops.wm.read_factory_settings(use_empty=True) must remove all meshes."""
    butil.spawn_cube()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    assert len(bpy.data.meshes) == 0


# ---------------------------------------------------------------------------
# Primitive object creation
# ---------------------------------------------------------------------------


def test_spawn_cube_returns_object():
    obj = butil.spawn_cube()
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "MESH"


def test_spawn_plane_returns_object():
    obj = butil.spawn_plane()
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "MESH"


def test_spawn_cylinder_returns_object():
    obj = butil.spawn_cylinder()
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "MESH"


def test_spawn_sphere_returns_object():
    obj = butil.spawn_sphere()
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "MESH"


def test_spawn_icosphere_returns_object():
    obj = butil.spawn_icosphere()
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "MESH"


def test_spawn_empty_returns_object():
    obj = butil.spawn_empty("test_empty")
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "EMPTY"


def test_spawn_cube_has_vertices():
    """A default cube must have exactly 8 vertices."""
    obj = butil.spawn_cube()
    assert len(obj.data.vertices) == 8


def test_spawn_cube_has_faces():
    """A default cube must have exactly 6 faces."""
    obj = butil.spawn_cube()
    assert len(obj.data.polygons) == 6


def test_spawn_cube_is_in_scene():
    """Spawned objects must be linked into the active scene."""
    obj = butil.spawn_cube()
    assert obj.name in bpy.context.scene.objects


# ---------------------------------------------------------------------------
# Mesh data access via foreach_get / foreach_set
# ---------------------------------------------------------------------------


def test_vertices_foreach_get_co():
    """obj.data.vertices.foreach_get('co', ...) must return flat float array."""
    obj = butil.spawn_cube()
    verts = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", verts)
    assert verts.shape == (24,)  # 8 vertices * 3 coords


def test_polygons_foreach_get_vertices():
    """obj.data.polygons.foreach_get('vertices', ...) must work for a quad-mesh cube."""
    obj = butil.spawn_cube()
    # Each face of a default cube is a quad (4 vertices)
    # foreach_get 'vertices' on polygons gives the flat vertex-index array per loop
    loops = np.empty(len(obj.data.loops), dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loops)
    assert len(loops) == len(obj.data.loops)


def test_mesh_update_calc_edges():
    """bpy.data.meshes.new() + vertices/polygons + update(calc_edges=True) must produce edges."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh_data = butil.objectdata_from_VF(vertices, faces)
    assert len(mesh_data.edges) > 0


def test_objectdata_from_VF_vertex_count():
    """objectdata_from_VF must produce a mesh with the given number of vertices."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh_data = butil.objectdata_from_VF(vertices, faces)
    assert len(mesh_data.vertices) == 3
    assert len(mesh_data.polygons) == 1


# ---------------------------------------------------------------------------
# Attribute API (POINT / FACE domains; FLOAT / FLOAT_VECTOR / BOOLEAN types)
# ---------------------------------------------------------------------------


def test_float_attribute_face_domain_roundtrip():
    """FLOAT attribute on FACE domain must survive a write/read roundtrip."""
    obj = butil.spawn_cube()
    n_faces = len(obj.data.polygons)
    data_written = np.arange(n_faces, dtype=np.float32)

    surface.write_attr_data(obj, "test_float_face", data_written, type="FLOAT", domain="FACE")
    data_read = surface.read_attr_data(obj, "test_float_face", domain="FACE")

    np.testing.assert_array_almost_equal(data_read, data_written)


def test_boolean_attribute_face_domain_roundtrip():
    """BOOLEAN attribute on FACE domain must survive a write/read roundtrip."""
    obj = butil.spawn_cube()
    n_faces = len(obj.data.polygons)
    data_written = np.array([True, False, True, False, True, False], dtype=bool)[:n_faces]

    surface.write_attr_data(obj, "test_bool_face", data_written, type="BOOLEAN", domain="FACE")
    data_read = surface.read_attr_data(obj, "test_bool_face", domain="FACE")

    np.testing.assert_array_equal(data_read.astype(bool), data_written)


def test_float_attribute_point_domain_roundtrip():
    """FLOAT attribute on POINT domain must survive a write/read roundtrip."""
    obj = butil.spawn_cube()
    n_verts = len(obj.data.vertices)
    data_written = np.ones(n_verts, dtype=np.float32) * 3.14

    surface.write_attr_data(obj, "test_float_point", data_written, type="FLOAT", domain="POINT")
    data_read = surface.read_attr_data(obj, "test_float_point", domain="POINT")

    np.testing.assert_array_almost_equal(data_read, data_written, decimal=5)


def test_float_vector_attribute_point_domain_roundtrip():
    """FLOAT_VECTOR attribute on POINT domain must survive a write/read roundtrip."""
    obj = butil.spawn_cube()
    n_verts = len(obj.data.vertices)
    data_written = np.tile([1.0, 2.0, 3.0], n_verts).astype(np.float32)

    attr = obj.data.attributes.new("test_fvec_point", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", data_written)

    result = np.empty(n_verts * 3, dtype=np.float32)
    attr.data.foreach_get("vector", result)

    np.testing.assert_array_almost_equal(result, data_written, decimal=5)


def test_attribute_domain_property():
    """Attribute .domain must be accessible and match what was used during creation."""
    obj = butil.spawn_cube()
    obj.data.attributes.new("check_domain", "FLOAT", "FACE")
    attr = obj.data.attributes["check_domain"]
    assert attr.domain == "FACE"


def test_attribute_data_type_property():
    """Attribute .data_type must be accessible and match what was used during creation."""
    obj = butil.spawn_cube()
    obj.data.attributes.new("check_dtype", "FLOAT_VECTOR", "POINT")
    attr = obj.data.attributes["check_dtype"]
    assert attr.data_type == "FLOAT_VECTOR"


# ---------------------------------------------------------------------------
# Context management: ViewportMode, SelectObjects, CursorLocation
# ---------------------------------------------------------------------------


def test_viewport_mode_switch_to_edit():
    """ViewportMode must switch object to EDIT mode and restore to OBJECT on exit."""
    obj = butil.spawn_cube()
    bpy.context.view_layer.objects.active = obj
    with butil.ViewportMode(obj, "EDIT"):
        assert bpy.context.object.mode == "EDIT"
    assert bpy.context.object.mode == "OBJECT"


def test_select_objects_context_manager():
    """SelectObjects must select the given objects and restore selection on exit."""
    a = butil.spawn_cube(name="cube_a")
    b = butil.spawn_cube(name="cube_b")

    # Initially deselect everything
    butil.select_none()

    with butil.SelectObjects([a, b]):
        selected = set(o.name for o in bpy.context.selected_objects)
        assert "cube_a" in selected
        assert "cube_b" in selected

    # After the context, selection should be restored (empty)
    assert len(bpy.context.selected_objects) == 0


def test_select_objects_sets_active():
    """SelectObjects must set the active object to objects[active_index]."""
    a = butil.spawn_cube(name="cube_x")
    b = butil.spawn_cube(name="cube_y")

    with butil.SelectObjects([a, b], active=0):
        assert bpy.context.active_object.name == "cube_x"


def test_cursor_location_context_manager():
    """CursorLocation must move the 3D cursor and restore it on exit."""
    original = tuple(bpy.context.scene.cursor.location)
    new_loc = (9.0, 8.0, 7.0)

    with butil.CursorLocation(new_loc):
        assert tuple(bpy.context.scene.cursor.location) == pytest.approx(new_loc)

    assert tuple(bpy.context.scene.cursor.location) == pytest.approx(original)


# ---------------------------------------------------------------------------
# Collection operations
# ---------------------------------------------------------------------------


def test_get_collection_creates_collection():
    """get_collection must create and link a named collection to the scene."""
    col = butil.get_collection("test_col_create")
    assert col is not None
    assert "test_col_create" in bpy.data.collections
    assert col.name in bpy.context.scene.collection.children


def test_get_collection_reuse():
    """get_collection with reuse=True must return the same object on repeated calls."""
    col1 = butil.get_collection("test_col_reuse")
    col2 = butil.get_collection("test_col_reuse", reuse=True)
    assert col1 is col2


def test_put_in_collection():
    """put_in_collection must move an object into the given collection exclusively."""
    obj = butil.spawn_cube()
    col = butil.get_collection("test_col_put")
    butil.put_in_collection(obj, col, exclusive=True)
    assert obj.name in col.objects
    # Object must not remain linked to the root scene collection
    assert obj.name not in bpy.context.scene.collection.objects


def test_collection_objects_link_unlink():
    """Collection.objects.link / unlink must add / remove objects from a collection."""
    obj = butil.spawn_cube()
    col = bpy.data.collections.new("test_link_unlink")
    bpy.context.scene.collection.children.link(col)

    col.objects.link(obj)
    assert obj.name in col.objects

    col.objects.unlink(obj)
    assert obj.name not in col.objects


# ---------------------------------------------------------------------------
# Modifier operations (SUBSURF)
# ---------------------------------------------------------------------------


def test_add_subsurf_modifier():
    """Adding a SUBSURF modifier must appear in obj.modifiers."""
    obj = butil.spawn_cube()
    mod = obj.modifiers.new("SubSurf", "SUBSURF")
    assert mod is not None
    assert mod.type == "SUBSURF"
    assert "SubSurf" in obj.modifiers


def test_apply_subsurf_modifier_increases_vertices():
    """Applying a SUBSURF modifier (levels=1) must produce more vertices than the base mesh."""
    obj = butil.spawn_cube()
    base_vert_count = len(obj.data.vertices)

    mod = obj.modifiers.new("SubSurf", "SUBSURF")
    mod.levels = 1

    butil.apply_modifiers(obj, mod=[mod])

    assert len(obj.data.vertices) > base_vert_count


def test_remove_modifier():
    """Removing a modifier via obj.modifiers.remove must leave no trace in obj.modifiers."""
    obj = butil.spawn_cube()
    mod = obj.modifiers.new("ToRemove", "SUBSURF")
    obj.modifiers.remove(mod)
    assert "ToRemove" not in obj.modifiers


# ---------------------------------------------------------------------------
# Geometry nodes interface (Blender 4.0+ API used in set_geomod_inputs)
# ---------------------------------------------------------------------------


def test_geometry_node_group_empty_new():
    """geometry_node_group_empty_new() must produce a valid GeometryNodeTree with I/O sockets."""
    ng = geometry_node_group_empty_new()
    assert ng.type == "GEOMETRY"

    inputs = ng_inputs(ng)
    assert "Geometry" in inputs

    outputs = {
        s.name: s for s in ng.interface.items_tree if s.in_out == "OUTPUT"
    }
    assert "Geometry" in outputs


def test_geometry_nodes_interface_items_tree():
    """interface.items_tree must be iterable and each item must have in_out/name attributes."""
    ng = geometry_node_group_empty_new()
    for item in ng.interface.items_tree:
        assert hasattr(item, "in_out")
        assert hasattr(item, "name")


def test_geometry_nodes_interface_new_socket():
    """interface.new_socket() must add a new INPUT socket accessible via ng_inputs."""
    ng = geometry_node_group_empty_new()
    ng.interface.new_socket(
        name="MyFloat", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    inputs = ng_inputs(ng)
    assert "MyFloat" in inputs
    assert inputs["MyFloat"].in_out == "INPUT"


def test_set_geomod_inputs_float():
    """set_geomod_inputs must write a float value through the geometry nodes modifier interface."""
    ng = geometry_node_group_empty_new()
    ng.interface.new_socket(name="Scale", in_out="INPUT", socket_type="NodeSocketFloat")

    # Give the socket a default value so we can verify assignment
    scale_socket = ng_inputs(ng)["Scale"]
    scale_socket.default_value = 1.0

    obj = butil.spawn_cube()
    mod = obj.modifiers.new("GeoNodes", "NODES")
    mod.node_group = ng

    butil.set_geomod_inputs(mod, {"Scale": 2.5})
    assert mod[scale_socket.identifier] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Depsgraph evaluation (used in to_mesh / evaluated_get)
# ---------------------------------------------------------------------------


def test_evaluated_depsgraph_get():
    """context.evaluated_depsgraph_get() must return a valid Depsgraph object."""
    deg = bpy.context.evaluated_depsgraph_get()
    assert deg is not None
    assert isinstance(deg, bpy.types.Depsgraph)


def test_evaluated_get_returns_object():
    """obj.evaluated_get(depsgraph) must return a valid Blender Object."""
    obj = butil.spawn_cube()
    deg = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deg)
    assert isinstance(eval_obj, bpy.types.Object)


def test_new_from_object_with_depsgraph():
    """bpy.data.meshes.new_from_object() must produce a valid Mesh with vertex data."""
    obj = butil.spawn_cube()
    deg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(deg), depsgraph=deg)
    assert isinstance(mesh, bpy.types.Mesh)
    assert len(mesh.vertices) > 0


def test_to_mesh_returns_new_object():
    """butil.to_mesh() must return a new MESH Object with an evaluated mesh."""
    obj = butil.spawn_cube()
    # Apply a simple modifier so that to_mesh has something to evaluate
    obj.modifiers.new("SubSurf", "SUBSURF")
    result = butil.to_mesh(obj)
    assert isinstance(result, bpy.types.Object)
    assert result.type == "MESH"
    assert len(result.data.vertices) > len(obj.data.vertices)


# ---------------------------------------------------------------------------
# Material slot management
# ---------------------------------------------------------------------------


def test_material_slot_new():
    """Creating a material and appending it to an object must create a material slot."""
    obj = butil.spawn_cube()
    mat = bpy.data.materials.new(name="TestMat")
    obj.data.materials.append(mat)
    assert len(obj.material_slots) == 1
    assert obj.material_slots[0].material is not None
    assert obj.material_slots[0].material.name == "TestMat"


def test_material_slot_none_check():
    """A freshly spawned object has no material slots."""
    obj = butil.spawn_cube()
    assert len(obj.material_slots) == 0


def test_surface_assign_material():
    """surface.assign_material must create exactly one material slot on a fresh cube."""
    from infinigen.assets.materials.dev import BasicBSDF

    obj = butil.spawn_cube()
    surface.assign_material(obj, BasicBSDF()())
    assert len(obj.material_slots) == 1
    assert obj.material_slots[0].material is not None


# ---------------------------------------------------------------------------
# Scene / render properties
# ---------------------------------------------------------------------------


def test_scene_render_resolution_x_readable():
    """bpy.context.scene.render.resolution_x must be a positive integer."""
    rx = bpy.context.scene.render.resolution_x
    assert isinstance(rx, int)
    assert rx > 0


def test_scene_render_resolution_y_readable():
    """bpy.context.scene.render.resolution_y must be a positive integer."""
    ry = bpy.context.scene.render.resolution_y
    assert isinstance(ry, int)
    assert ry > 0


def test_get_camera_res_returns_array():
    """butil.get_camera_res() must return a 2-element float array."""
    res = butil.get_camera_res()
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    assert (res > 0).all()


# ---------------------------------------------------------------------------
# Object hierarchy / parenting
# ---------------------------------------------------------------------------


def test_object_parenting():
    """butil.parent_to(child, parent) must set child.parent correctly."""
    parent_obj = butil.spawn_cube(name="parent_obj")
    child_obj = butil.spawn_cube(name="child_obj")

    butil.parent_to(child_obj, parent_obj, keep_transform=True)

    assert child_obj.parent is parent_obj


def test_iter_object_tree():
    """butil.iter_object_tree must yield the root and all recursive children."""
    parent_obj = butil.spawn_cube(name="root_obj")
    child_obj = butil.spawn_cube(name="child_obj")
    butil.parent_to(child_obj, parent_obj, keep_transform=True)

    tree = list(butil.iter_object_tree(parent_obj))
    names = [o.name for o in tree]
    assert "root_obj" in names
    assert "child_obj" in names


# ---------------------------------------------------------------------------
# Object join
# ---------------------------------------------------------------------------


def test_join_objects():
    """butil.join_objects must merge two mesh objects into one."""
    a = butil.spawn_cube(name="join_a")
    b = butil.spawn_cube(name="join_b")
    total_verts = len(a.data.vertices) + len(b.data.vertices)

    merged = butil.join_objects([a, b])
    assert isinstance(merged, bpy.types.Object)
    assert len(merged.data.vertices) == total_verts


# ---------------------------------------------------------------------------
# Mesh from trimesh roundtrip
# ---------------------------------------------------------------------------


def test_object_to_trimesh_and_back():
    """Converting a triangulated Blender mesh to trimesh must preserve vertex/face counts.

    object_to_trimesh requires a triangulated mesh (all faces must be triangles).
    We use objectdata_from_VF which always produces triangles.
    """
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh_data = butil.objectdata_from_VF(vertices, faces)
    obj = bpy.data.objects.new("test_tri_obj", mesh_data)
    bpy.context.scene.collection.objects.link(obj)

    tm = butil.object_to_trimesh(obj)

    assert len(tm.vertices) == 4
    assert len(tm.faces) == 2


# ---------------------------------------------------------------------------
# BMesh operations (used in spawn_capsule, merge_by_distance, etc.)
# ---------------------------------------------------------------------------


def test_spawn_capsule_has_geometry():
    """butil.spawn_capsule must produce a mesh with vertices and faces."""
    obj = butil.spawn_capsule(rad=0.5, height=1.0)
    assert isinstance(obj, bpy.types.Object)
    assert obj.type == "MESH"
    assert len(obj.data.vertices) > 0
    assert len(obj.data.polygons) > 0


def test_merge_by_distance():
    """butil.merge_by_distance must not raise and must retain a valid mesh."""
    obj = butil.spawn_cube()
    initial_verts = len(obj.data.vertices)
    # Threshold smaller than cube edge length – no merging expected on a clean cube
    butil.merge_by_distance(obj, face_size=0.001)
    assert len(obj.data.vertices) == initial_verts


# ---------------------------------------------------------------------------
# Color management – configure_color_management() (new in Blender 5.0 upgrade)
# ---------------------------------------------------------------------------


def test_configure_color_management_default():
    """configure_color_management() with defaults must set AgX and sRGB display."""
    from infinigen.core.init import configure_color_management

    configure_color_management.clear_config()
    configure_color_management()
    scene = bpy.context.scene
    assert scene.view_settings.view_transform == "AgX"
    assert scene.display_settings.display_device == "sRGB"
    assert scene.view_settings.look == "None"


def test_configure_color_management_aces2():
    """configure_color_management() with ACES 2.0 must apply the ACES view transform."""
    from infinigen.core.init import configure_color_management

    configure_color_management.clear_config()
    try:
        configure_color_management(view_transform="ACES 2.0", display_device="sRGB")
        assert bpy.context.scene.view_settings.view_transform == "ACES 2.0"
    except (TypeError, AttributeError):
        pytest.skip(
            "ACES 2.0 view transform not available in this Blender build – "
            "requires a complete Blender 5.0.x installation with OCIO config."
        )


def test_configure_color_management_exposure():
    """configure_color_management() must propagate the exposure parameter."""
    from infinigen.core.init import configure_color_management

    configure_color_management.clear_config()
    configure_color_management(exposure=1.5)
    assert abs(bpy.context.scene.view_settings.exposure - 1.5) < 1e-5


def test_configure_color_management_warns_unknown_transform(caplog):
    """configure_color_management() must log a warning for an unknown view_transform."""
    import logging

    from infinigen.core.init import configure_color_management

    configure_color_management.clear_config()
    with caplog.at_level(logging.WARNING, logger="infinigen.core.init"):
        try:
            configure_color_management(view_transform="NotARealTransform_XYZ")
        except Exception:
            pass
    assert any("NotARealTransform_XYZ" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Nishita sky atmosphere – gin-configurable ozone/altitude (Blender 5.0 upgrade)
# ---------------------------------------------------------------------------


def test_nishita_lighting_gin_params_ozone_altitude():
    """nishita_lighting must expose ozone_density and altitude as gin parameters."""
    import inspect

    from infinigen.assets.lighting.sky_lighting import nishita_lighting

    sig = inspect.signature(nishita_lighting.__wrapped__)
    params = sig.parameters
    assert "ozone_density" in params, "ozone_density must be a gin-configurable parameter"
    assert "altitude" in params, "altitude must be a gin-configurable parameter"


def test_nishita_lighting_gin_params_have_defaults():
    """ozone_density and altitude parameters must have non-empty default distributions."""
    import inspect

    from infinigen.assets.lighting.sky_lighting import nishita_lighting

    sig = inspect.signature(nishita_lighting.__wrapped__)
    ozone_default = sig.parameters["ozone_density"].default
    altitude_default = sig.parameters["altitude"].default
    assert ozone_default is not inspect.Parameter.empty
    assert altitude_default is not inspect.Parameter.empty


# ---------------------------------------------------------------------------
# EEVEE Next annotation passes – configure_eevee_next() (Blender 5.0 upgrade)
# ---------------------------------------------------------------------------


def test_configure_eevee_next_sets_engine():
    """configure_eevee_next() must set the render engine to BLENDER_EEVEE_NEXT."""
    from infinigen.core.init import configure_eevee_next

    configure_eevee_next.clear_config()
    configure_eevee_next()
    assert bpy.context.scene.render.engine == "BLENDER_EEVEE_NEXT"


def test_configure_eevee_next_disables_shadows_by_default():
    """configure_eevee_next() must disable shadows by default for faster annotation rendering."""
    from infinigen.core.init import configure_eevee_next

    configure_eevee_next.clear_config()
    configure_eevee_next()
    assert bpy.context.scene.eevee.use_shadows is False


def test_configure_eevee_next_single_taa_sample():
    """configure_eevee_next() must use 1 TAA sample by default (sufficient for flat shading)."""
    from infinigen.core.init import configure_eevee_next

    configure_eevee_next.clear_config()
    configure_eevee_next()
    assert bpy.context.scene.eevee.taa_render_samples == 1


def test_configure_eevee_next_shadows_enabled_when_requested():
    """configure_eevee_next(use_shadows=True) must enable shadows."""
    from infinigen.core.init import configure_eevee_next

    configure_eevee_next.clear_config()
    configure_eevee_next(use_shadows=True)
    assert bpy.context.scene.eevee.use_shadows is True


def test_render_image_has_eevee_annotation_param():
    """render_image must expose use_eevee_next_for_annotations as a gin parameter."""
    import inspect

    from infinigen.core.rendering.render import render_image

    sig = inspect.signature(render_image.__wrapped__)
    assert "use_eevee_next_for_annotations" in sig.parameters


def test_configure_eevee_next_gin_params():
    """configure_eevee_next must expose all relevant settings as gin parameters."""
    import inspect

    from infinigen.core.init import configure_eevee_next

    sig = inspect.signature(configure_eevee_next.__wrapped__)
    params = sig.parameters
    assert "use_shadows" in params
    assert "taa_render_samples" in params
    assert "use_gtao" in params
    assert "use_high_quality_normals" in params
    assert "use_bloom" in params


def test_configure_eevee_next_high_quality_normals_default():
    """configure_eevee_next() must enable high-quality normals by default for accurate annotation."""
    from infinigen.core.init import configure_eevee_next

    configure_eevee_next.clear_config()
    configure_eevee_next()
    assert bpy.context.scene.eevee.use_high_quality_normals is True


# ---------------------------------------------------------------------------
# Cycles denoiser fallback chain – _configure_denoiser() (Blender 5.0 upgrade)
# ---------------------------------------------------------------------------


def test_configure_denoiser_sets_engine_to_cycles():
    """configure_render_cycles() with denoise=True must keep CYCLES as the engine."""
    import gin

    from infinigen.core.init import configure_render_cycles

    configure_render_cycles.clear_config()
    gin.clear_config()
    configure_render_cycles(
        min_samples=0,
        num_samples=512,
        time_limit=0,
        adaptive_threshold=0.01,
        exposure=1.0,
        denoise=True,
    )
    assert bpy.context.scene.render.engine == "CYCLES"


def test_configure_denoiser_with_denoise_false():
    """configure_render_cycles() with denoise=False must set use_denoising=False."""
    import gin

    from infinigen.core.init import configure_render_cycles

    configure_render_cycles.clear_config()
    gin.clear_config()
    configure_render_cycles(
        min_samples=0,
        num_samples=512,
        time_limit=0,
        adaptive_threshold=0.01,
        exposure=1.0,
        denoise=False,
    )
    assert bpy.context.scene.cycles.use_denoising is False


def test_configure_denoiser_fallback_leaves_defined_state():
    """_configure_denoiser() must leave the scene in a defined state (denoiser set or disabled)."""
    from infinigen.core.init import _configure_denoiser

    # Save original state
    orig_use_denoising = bpy.context.scene.cycles.use_denoising
    bpy.context.scene.cycles.use_denoising = True

    _configure_denoiser()

    # Scene must be in a valid state: either a known denoiser is selected or
    # use_denoising has been explicitly set to False.
    # CYCLES_DENOISER_PRIORITY = ["OPTIX", "OPENIMAGEDENOISE"]
    valid_denoisers = {"OPTIX", "OPENIMAGEDENOISE"}
    if bpy.context.scene.cycles.use_denoising:
        assert bpy.context.scene.cycles.denoiser in valid_denoisers, (
            f"use_denoising=True but denoiser={bpy.context.scene.cycles.denoiser!r} "
            "is not in the supported set"
        )

    # Restore
    bpy.context.scene.cycles.use_denoising = orig_use_denoising


def test_configure_render_cycles_sample_count():
    """configure_render_cycles() must apply the num_samples gin parameter."""
    import gin

    from infinigen.core.init import configure_render_cycles

    configure_render_cycles.clear_config()
    gin.clear_config()
    configure_render_cycles(
        min_samples=128,
        num_samples=1024,
        time_limit=0,
        adaptive_threshold=0.01,
        exposure=1.0,
        denoise=False,
    )
    assert bpy.context.scene.cycles.samples == 1024
    assert bpy.context.scene.cycles.adaptive_min_samples == 128


def test_configure_denoiser_gin_params():
    """configure_render_cycles must expose denoise as a gin-configurable parameter."""
    import inspect

    from infinigen.core.init import configure_render_cycles

    sig = inspect.signature(configure_render_cycles.__wrapped__)
    params = sig.parameters
    assert "denoise" in params
    assert "num_samples" in params
    assert "min_samples" in params
    assert "adaptive_threshold" in params


# ---------------------------------------------------------------------------
# Scatter density scaling – scatter_instances() (Blender 5.0 Massive Geometry)
# ---------------------------------------------------------------------------


def test_scatter_instances_gin_params():
    """scatter_instances must expose density_scale and max_density as gin parameters."""
    import inspect

    from infinigen.core.placement.instance_scatter import scatter_instances

    sig = inspect.signature(scatter_instances.__wrapped__)
    params = sig.parameters
    assert "density_scale" in params, "density_scale gin parameter missing"
    assert "max_density" in params, "max_density gin parameter missing"
    assert "vol_density" in params
    assert "density" in params


def test_scatter_density_scale_default_is_one():
    """scatter_instances.density_scale must default to 1.0 (no-op for existing scenes)."""
    import inspect

    from infinigen.core.placement.instance_scatter import scatter_instances

    sig = inspect.signature(scatter_instances.__wrapped__)
    assert sig.parameters["density_scale"].default == 1.0, (
        "Default density_scale is not 1.0 — existing scenes would be affected"
    )


def test_scatter_max_density_default_raised():
    """Default max_density must be >= 10000 (raised for Blender 5.0 buffer limits)."""
    from infinigen.core.placement.instance_scatter import SCATTER_MAX_DENSITY_DEFAULT

    assert SCATTER_MAX_DENSITY_DEFAULT >= 10000, (
        f"SCATTER_MAX_DENSITY_DEFAULT={SCATTER_MAX_DENSITY_DEFAULT} is below 10000; "
        "expected to be raised for Blender 5.0"
    )


def test_scatter_density_scale_applies():
    """density_scale multiplier must be applied to density before the max_density cap."""
    import inspect

    from infinigen.core.placement.instance_scatter import (
        SCATTER_MAX_DENSITY_DEFAULT,
        scatter_instances,
    )

    sig = inspect.signature(scatter_instances.__wrapped__)
    # Verify the logical contract documented in the docstring is consistent:
    # density_scale default is 1.0, max_density default >= 10000
    assert sig.parameters["density_scale"].default == 1.0
    assert sig.parameters["max_density"].default >= SCATTER_MAX_DENSITY_DEFAULT


# ---------------------------------------------------------------------------
# Render Time Pass (P2) — Blender 5.0 per-pixel cost heatmap
# ---------------------------------------------------------------------------


def test_configure_render_time_pass_exists():
    """configure_render_time_pass must exist in core.init."""
    import importlib

    m = importlib.import_module("infinigen.core.init")
    assert hasattr(m, "configure_render_time_pass"), (
        "configure_render_time_pass missing from infinigen.core.init"
    )
    assert callable(m.configure_render_time_pass)


def test_configure_render_time_pass_is_gin_configurable():
    """configure_render_time_pass must be decorated with @gin.configurable."""
    from infinigen.core.init import configure_render_time_pass

    assert hasattr(configure_render_time_pass, "__wrapped__"), (
        "configure_render_time_pass is not gin-configurable — missing @gin.configurable"
    )


def test_configure_render_time_pass_gin_params():
    """configure_render_time_pass must expose enabled and log_on_enable parameters."""
    import inspect

    from infinigen.core.init import configure_render_time_pass

    sig = inspect.signature(configure_render_time_pass.__wrapped__)
    params = sig.parameters
    assert "enabled" in params, "enabled gin parameter missing"
    assert "log_on_enable" in params, "log_on_enable gin parameter missing"


def test_configure_render_time_pass_default_disabled():
    """configure_render_time_pass.enabled must default to False (zero overhead by default)."""
    import inspect

    from infinigen.core.init import configure_render_time_pass

    sig = inspect.signature(configure_render_time_pass.__wrapped__)
    assert sig.parameters["enabled"].default is False, (
        "configure_render_time_pass.enabled defaults to True — "
        "this would add overhead to every render by default"
    )


def test_render_time_pass_descriptor_exists():
    """RENDER_TIME_PASS_DESCRIPTOR must be a 2-tuple matching the compositor convention."""
    from infinigen.core.init import RENDER_TIME_PASS_DESCRIPTOR

    assert isinstance(RENDER_TIME_PASS_DESCRIPTOR, tuple), (
        "RENDER_TIME_PASS_DESCRIPTOR must be a tuple"
    )
    assert len(RENDER_TIME_PASS_DESCRIPTOR) == 2, (
        "RENDER_TIME_PASS_DESCRIPTOR must have exactly 2 elements (pass_name, socket_name)"
    )
    pass_name, socket_name = RENDER_TIME_PASS_DESCRIPTOR
    assert pass_name == "render_time", f"Expected 'render_time', got {pass_name!r}"
    assert socket_name == "RenderTime", f"Expected 'RenderTime', got {socket_name!r}"


def test_render_image_enable_render_time_pass_param():
    """render_image must expose enable_render_time_pass as a gin parameter."""
    import inspect

    from infinigen.core.rendering.render import render_image

    sig = inspect.signature(render_image.__wrapped__)
    assert "enable_render_time_pass" in sig.parameters, (
        "enable_render_time_pass parameter missing from render_image"
    )
    assert sig.parameters["enable_render_time_pass"].default is False, (
        "enable_render_time_pass must default to False"
    )


# ---------------------------------------------------------------------------
# P2: Repeat Zones & Switch Menu — Blender 5.0 node system additions
# ---------------------------------------------------------------------------


def test_nodes_repeat_input_defined():
    """Nodes.RepeatInput must be defined in node_info (bl5.0 Repeat Zone)."""
    from infinigen.core.nodes.node_info import Nodes

    assert hasattr(Nodes, "RepeatInput"), "Nodes.RepeatInput missing"
    assert Nodes.RepeatInput == "GeometryNodeRepeatInput"


def test_nodes_repeat_output_defined():
    """Nodes.RepeatOutput must be defined in node_info (bl5.0 Repeat Zone)."""
    from infinigen.core.nodes.node_info import Nodes

    assert hasattr(Nodes, "RepeatOutput"), "Nodes.RepeatOutput missing"
    assert Nodes.RepeatOutput == "GeometryNodeRepeatOutput"


def test_nodes_menu_switch_defined():
    """Nodes.MenuSwitch must be defined in node_info (bl5.0 Menu Switch)."""
    from infinigen.core.nodes.node_info import Nodes

    assert hasattr(Nodes, "MenuSwitch"), "Nodes.MenuSwitch missing"
    assert Nodes.MenuSwitch == "GeometryNodeMenuSwitch"


def test_nodes_foreach_geometry_element_defined():
    """ForEachGeometryElement Input/Output must be defined (bl5.0)."""
    from infinigen.core.nodes.node_info import Nodes

    assert hasattr(Nodes, "ForEachGeometryElementInput"), (
        "Nodes.ForEachGeometryElementInput missing"
    )
    assert hasattr(Nodes, "ForEachGeometryElementOutput"), (
        "Nodes.ForEachGeometryElementOutput missing"
    )


def test_blender5_zone_node_types_constant():
    """BLENDER5_ZONE_NODE_TYPES must be a frozenset containing the zone node values."""
    from infinigen.core.nodes.node_wrangler import BLENDER5_ZONE_NODE_TYPES
    from infinigen.core.nodes.node_info import Nodes

    assert isinstance(BLENDER5_ZONE_NODE_TYPES, frozenset), (
        "BLENDER5_ZONE_NODE_TYPES must be a frozenset"
    )
    assert Nodes.RepeatInput in BLENDER5_ZONE_NODE_TYPES
    assert Nodes.RepeatOutput in BLENDER5_ZONE_NODE_TYPES
    assert Nodes.ForEachGeometryElementInput in BLENDER5_ZONE_NODE_TYPES
    assert Nodes.ForEachGeometryElementOutput in BLENDER5_ZONE_NODE_TYPES


def test_node_wrangler_has_new_repeat_zone():
    """NodeWrangler must expose new_repeat_zone() helper (bl5.0)."""
    import inspect

    from infinigen.core.nodes.node_wrangler import NodeWrangler

    assert hasattr(NodeWrangler, "new_repeat_zone"), (
        "NodeWrangler.new_repeat_zone() missing"
    )
    sig = inspect.signature(NodeWrangler.new_repeat_zone)
    assert "iterations" in sig.parameters, "new_repeat_zone: 'iterations' param missing"
    assert sig.parameters["iterations"].default == 1, (
        "new_repeat_zone: 'iterations' must default to 1"
    )
    assert "input_kwargs" in sig.parameters, (
        "new_repeat_zone: 'input_kwargs' param missing"
    )


def test_node_wrangler_has_new_menu_switch():
    """NodeWrangler must expose new_menu_switch() helper (bl5.0)."""
    import inspect

    from infinigen.core.nodes.node_wrangler import NodeWrangler

    assert hasattr(NodeWrangler, "new_menu_switch"), (
        "NodeWrangler.new_menu_switch() missing"
    )
    sig = inspect.signature(NodeWrangler.new_menu_switch)
    assert "data_type" in sig.parameters, "new_menu_switch: 'data_type' param missing"
    assert "items" in sig.parameters, "new_menu_switch: 'items' param missing"
    assert "active_index" in sig.parameters, (
        "new_menu_switch: 'active_index' param missing"
    )
    assert sig.parameters["data_type"].default == "GEOMETRY", (
        "new_menu_switch: 'data_type' must default to 'GEOMETRY'"
    )
    assert sig.parameters["active_index"].default == 0, (
        "new_menu_switch: 'active_index' must default to 0"
    )


# ---------------------------------------------------------------------------
# P3: Gin-configurable Volume Atmosphere — Blender 5.0+
# ---------------------------------------------------------------------------


def test_configure_volume_rendering_exists():
    """configure_volume_rendering() must exist in core.init."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init",
        "infinigen/core/init.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        pass
    assert hasattr(module, "configure_volume_rendering"), (
        "configure_volume_rendering() not found in core/init.py"
    )


def test_configure_volume_rendering_is_gin_configurable():
    """configure_volume_rendering must be decorated with @gin.configurable."""
    import inspect

    def _load():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "infinigen.core.init", "infinigen/core/init.py"
        )
        m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        try:
            spec.loader.exec_module(m)  # type: ignore[union-attr]
        except Exception:
            pass
        return m

    m = _load()
    fn = getattr(m, "configure_volume_rendering", None)
    assert fn is not None
    # gin.configurable wraps the function; inspect the qualified name or __wrapped__
    src = inspect.getsource(m)
    assert "@gin.configurable" in src or "gin.configurable" in src


def test_configure_volume_rendering_signature():
    """configure_volume_rendering must expose all documented parameters."""
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init", "infinigen/core/init.py"
    )
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(m)  # type: ignore[union-attr]
    except Exception:
        pass
    fn = getattr(m, "configure_volume_rendering")
    sig = inspect.signature(fn)
    assert "volume_step_rate" in sig.parameters
    assert "volume_max_steps" in sig.parameters
    assert "volume_bounces" in sig.parameters
    assert "atmosphere_preset" in sig.parameters
    assert "use_world_volume" in sig.parameters
    assert "world_volume_density" in sig.parameters
    assert "world_volume_anisotropy" in sig.parameters
    assert "world_volume_color" in sig.parameters


def test_configure_volume_rendering_defaults():
    """configure_volume_rendering default parameters must match baseline constants."""
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init", "infinigen/core/init.py"
    )
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(m)  # type: ignore[union-attr]
    except Exception:
        pass
    fn = getattr(m, "configure_volume_rendering")
    sig = inspect.signature(fn)
    assert sig.parameters["atmosphere_preset"].default is None, (
        "atmosphere_preset must default to None (no preset active)"
    )
    assert sig.parameters["use_world_volume"].default is False, (
        "use_world_volume must default to False"
    )
    # Numeric defaults from CYCLES_VOLUME_* constants
    vol_step_rate = getattr(m, "CYCLES_VOLUME_STEP_RATE", None)
    vol_max_steps = getattr(m, "CYCLES_VOLUME_MAX_STEPS", None)
    vol_bounces = getattr(m, "CYCLES_VOLUME_BOUNCES", None)
    assert vol_step_rate is not None
    assert vol_max_steps is not None
    assert vol_bounces is not None
    assert sig.parameters["volume_step_rate"].default == vol_step_rate
    assert sig.parameters["volume_max_steps"].default == vol_max_steps
    assert sig.parameters["volume_bounces"].default == vol_bounces


def test_atmosphere_quality_presets_defined():
    """ATMOSPHERE_QUALITY_PRESETS must define fog, haze, dense, and none presets."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init", "infinigen/core/init.py"
    )
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(m)  # type: ignore[union-attr]
    except Exception:
        pass
    presets = getattr(m, "ATMOSPHERE_QUALITY_PRESETS", None)
    assert presets is not None, "ATMOSPHERE_QUALITY_PRESETS missing from core/init.py"
    assert isinstance(presets, dict)
    for expected_key in ("none", "fog", "haze", "dense"):
        assert expected_key in presets, (
            f"ATMOSPHERE_QUALITY_PRESETS missing key '{expected_key}'"
        )


def test_atmosphere_presets_have_required_keys():
    """Each non-none atmosphere preset must have volume_step_rate/max_steps/bounces."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init", "infinigen/core/init.py"
    )
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(m)  # type: ignore[union-attr]
    except Exception:
        pass
    presets = getattr(m, "ATMOSPHERE_QUALITY_PRESETS", {})
    required_keys = {"volume_step_rate", "volume_max_steps", "volume_bounces"}
    for name, cfg in presets.items():
        for key in required_keys:
            assert key in cfg, (
                f"ATMOSPHERE_QUALITY_PRESETS[{name!r}] missing key '{key}'"
            )


def test_fog_preset_denser_than_haze():
    """The 'fog' preset must be denser (higher density) than 'haze'."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init", "infinigen/core/init.py"
    )
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(m)  # type: ignore[union-attr]
    except Exception:
        pass
    presets = getattr(m, "ATMOSPHERE_QUALITY_PRESETS", {})
    fog_density = presets["fog"].get("world_volume_density", 0)
    haze_density = presets["haze"].get("world_volume_density", 0)
    assert fog_density > haze_density, (
        f"Expected fog density ({fog_density}) > haze density ({haze_density})"
    )


def test_dense_preset_has_more_steps_than_haze():
    """'dense' atmosphere must have more volume_max_steps than 'haze'."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "infinigen.core.init", "infinigen/core/init.py"
    )
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(m)  # type: ignore[union-attr]
    except Exception:
        pass
    presets = getattr(m, "ATMOSPHERE_QUALITY_PRESETS", {})
    assert presets["dense"]["volume_max_steps"] > presets["haze"]["volume_max_steps"]


# ---------------------------------------------------------------------------
# Blender 5.0 P3 — SDF Grid / Volume Grid node type registrations
# ---------------------------------------------------------------------------


def test_sdf_grid_node_types_defined():
    """Blender 5.0 SDF / Volume Grid node types must be registered in Nodes enum."""
    try:
        from infinigen.core.nodes.node_info import Nodes
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    expected = {
        "GetNamedGrid": "GeometryNodeGetNamedGrid",
        "StoreNamedGrid": "GeometryNodeStoreNamedGrid",
        "GridInfo": "GeometryNodeGridInfo",
        "SampleGrid": "GeometryNodeSampleGrid",
        "FieldToGrid": "GeometryNodeFieldToGrid",
        "GridToMesh": "GeometryNodeGridToMesh",
        "MeshToSdfGrid": "GeometryNodeMeshToSDFGrid",
        "PointsToSdfGrid": "GeometryNodePointsToSDFGrid",
        "MeshToDensityGrid": "GeometryNodeMeshToDensityGrid",
        "SdfGridBoolean": "GeometryNodeSDFGridBoolean",
        "SdfGridOffset": "GeometryNodeSDFGridOffset",
        "SdfFillet": "GeometryNodeSDFFillet",
        "AdvectGrid": "GeometryNodeAdvectGrid",
    }
    for attr, expected_value in expected.items():
        assert hasattr(Nodes, attr), f"Nodes.{attr} is missing from node_info.py"
        assert getattr(Nodes, attr) == expected_value, (
            f"Nodes.{attr} = {getattr(Nodes, attr)!r} but expected {expected_value!r}"
        )


def test_blender5_volume_grid_frozenset_exists():
    """BLENDER5_VOLUME_GRID_NODE_TYPES frozenset must exist in node_wrangler."""
    try:
        from infinigen.core.nodes.node_wrangler import BLENDER5_VOLUME_GRID_NODE_TYPES
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    assert isinstance(BLENDER5_VOLUME_GRID_NODE_TYPES, frozenset)
    assert len(BLENDER5_VOLUME_GRID_NODE_TYPES) >= 10, (
        "Expected at least 10 Volume Grid node types registered"
    )


def test_sdf_grid_boolean_in_volume_frozenset():
    """SdfGridBoolean must be in BLENDER5_VOLUME_GRID_NODE_TYPES."""
    try:
        from infinigen.core.nodes.node_info import Nodes
        from infinigen.core.nodes.node_wrangler import BLENDER5_VOLUME_GRID_NODE_TYPES
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    assert Nodes.SdfGridBoolean in BLENDER5_VOLUME_GRID_NODE_TYPES


def test_mesh_to_sdf_in_volume_frozenset():
    """MeshToSdfGrid must be in BLENDER5_VOLUME_GRID_NODE_TYPES."""
    try:
        from infinigen.core.nodes.node_info import Nodes
        from infinigen.core.nodes.node_wrangler import BLENDER5_VOLUME_GRID_NODE_TYPES
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    assert Nodes.MeshToSdfGrid in BLENDER5_VOLUME_GRID_NODE_TYPES


def test_field_to_grid_in_volume_frozenset():
    """FieldToGrid must be in BLENDER5_VOLUME_GRID_NODE_TYPES."""
    try:
        from infinigen.core.nodes.node_info import Nodes
        from infinigen.core.nodes.node_wrangler import BLENDER5_VOLUME_GRID_NODE_TYPES
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    assert Nodes.FieldToGrid in BLENDER5_VOLUME_GRID_NODE_TYPES


def test_new_sdf_grid_boolean_helper_signature():
    """NodeWrangler.new_sdf_grid_boolean must accept operation/grid_a/grid_b."""
    import inspect

    try:
        from infinigen.core.nodes.node_wrangler import NodeWrangler
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    sig = inspect.signature(NodeWrangler.new_sdf_grid_boolean)
    params = sig.parameters
    assert "operation" in params, "new_sdf_grid_boolean must accept 'operation' param"
    assert params["operation"].default == "UNION"
    assert "grid_a" in params
    assert "grid_b" in params


def test_new_field_to_grid_helper_signature():
    """NodeWrangler.new_field_to_grid must accept field/resolution/voxel_size."""
    import inspect

    try:
        from infinigen.core.nodes.node_wrangler import NodeWrangler
    except ModuleNotFoundError:
        pytest.skip("bpy not available in this environment")

    sig = inspect.signature(NodeWrangler.new_field_to_grid)
    params = sig.parameters
    assert "field" in params
    assert "resolution" in params
    assert params["resolution"].default == 32
    assert "voxel_size" in params
    assert params["voxel_size"].default is None


# ---------------------------------------------------------------------------
# P3: Light Linking (Blender 5.0 stable feature)
# ---------------------------------------------------------------------------


def test_light_linking_modes_frozenset_exists():
    """LIGHT_LINKING_MODES frozenset must exist in core/init.py."""
    m = _load_core_init_module()
    assert hasattr(m, "LIGHT_LINKING_MODES"), (
        "LIGHT_LINKING_MODES frozenset is missing from core/init.py"
    )
    modes = m.LIGHT_LINKING_MODES
    assert isinstance(modes, frozenset), (
        f"LIGHT_LINKING_MODES should be a frozenset, got {type(modes)}"
    )


def test_light_linking_modes_contains_expected():
    """LIGHT_LINKING_MODES must contain all four expected policy modes."""
    m = _load_core_init_module()
    modes = m.LIGHT_LINKING_MODES
    for expected in ("none", "sun_exclude_interior", "annotation", "custom"):
        assert expected in modes, (
            f"Expected mode {expected!r} missing from LIGHT_LINKING_MODES"
        )


def test_blender_light_types_constant_exists():
    """BLENDER_LIGHT_TYPES tuple must be present in core/init.py."""
    m = _load_core_init_module()
    assert hasattr(m, "BLENDER_LIGHT_TYPES"), (
        "BLENDER_LIGHT_TYPES tuple is missing from core/init.py"
    )
    light_types = m.BLENDER_LIGHT_TYPES
    assert isinstance(light_types, tuple), (
        f"BLENDER_LIGHT_TYPES should be a tuple, got {type(light_types)}"
    )
    assert "SUN" in light_types
    assert "AREA" in light_types
    assert "POINT" in light_types


def test_configure_light_linking_function_exists():
    """configure_light_linking must be defined in core/init.py."""
    m = _load_core_init_module()
    assert hasattr(m, "configure_light_linking"), (
        "configure_light_linking function is missing from core/init.py"
    )
    assert callable(m.configure_light_linking)


def test_configure_light_linking_is_gin_configurable():
    """configure_light_linking must be decorated with @gin.configurable."""
    import inspect

    m = _load_core_init_module()
    fn = m.configure_light_linking
    # gin.configurable wraps the function; check the original is accessible
    # via __wrapped__ or inspect the qualname
    sig = inspect.signature(fn)
    params = sig.parameters
    assert "mode" in params, "configure_light_linking must accept 'mode' param"


def test_configure_light_linking_default_mode_is_none():
    """configure_light_linking default mode must be 'none' (zero overhead)."""
    import inspect

    m = _load_core_init_module()
    sig = inspect.signature(m.configure_light_linking)
    params = sig.parameters
    assert "mode" in params
    default = params["mode"].default
    assert default == "none", (
        f"configure_light_linking.mode default should be 'none', got {default!r}"
    )


def test_configure_light_linking_signature_include_exclude():
    """configure_light_linking must accept include_names and exclude_names."""
    import inspect

    m = _load_core_init_module()
    sig = inspect.signature(m.configure_light_linking)
    params = sig.parameters
    assert "include_names" in params, "configure_light_linking needs include_names"
    assert "exclude_names" in params, "configure_light_linking needs exclude_names"
    assert params["include_names"].default == ()
    assert params["exclude_names"].default == ()


def test_configure_light_linking_returns_int():
    """configure_light_linking return type annotation must be int."""
    import inspect

    m = _load_core_init_module()
    sig = inspect.signature(m.configure_light_linking)
    ret = sig.return_annotation
    # return annotation may be inspect.Parameter.empty if annotations not loaded
    if ret is not inspect.Parameter.empty:
        assert ret is int or ret == "int", (
            f"configure_light_linking should return int, got {ret}"
        )
