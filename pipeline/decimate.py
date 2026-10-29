"""Decimate a large GLB with Blender and export GLB (keeps texture/UVs). Usage: decimate.py in.glb out.glb target_faces"""
import sys, bpy
src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
for o in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
    n = len(o.data.polygons)
    m = o.modifiers.new("dec", "DECIMATE"); m.ratio = min(1.0, target / n)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.modifier_apply(modifier="dec")
    print(o.name, n, "->", len(o.data.polygons))
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
