"""Blender 5.1 compatibility patches — replaces removed operators."""
import bpy

def _primitive_vert_add_fix():
    """Replace bpy.ops.mesh.primitive_vert_add (removed in Blender 5.1)"""
    import bmesh
    from mathutils import Vector
    
    def add_vert():
        bm = bmesh.new()
        bm.verts.new((0, 0, 0))
        mesh = bpy.data.meshes.new("VertMesh")
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new("Vert", mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        return obj
    
    # Store original if it exists
    if not hasattr(bpy.ops.mesh, '_primitive_vert_add_original'):
        if hasattr(bpy.ops.mesh, 'primitive_vert_add'):
            try:
                bpy.ops.mesh._primitive_vert_add_original = bpy.ops.mesh.primitive_vert_add
            except:
                pass
    
    # Monkey-patch: replace the operator with our function
    if not hasattr(bpy.ops.mesh, 'primitive_vert_add') or True:
        # Create a callable object that mimics the operator
        class VertAddOp:
            def __call__(self):
                add_vert()
                return {'FINISHED'}
        bpy.ops.mesh.primitive_vert_add = VertAddOp()

_primitive_vert_add_fix()
