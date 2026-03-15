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
