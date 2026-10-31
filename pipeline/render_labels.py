"""Render a vertex-coloured PLY (piece colours) from fixed views. Usage: render_labels.py in.ply out_prefix"""
import sys, bpy
from mathutils import Vector
src, pre = sys.argv[1], sys.argv[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=src)
o = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
o.scale = (0.001, 0.001, 0.001)
mat = bpy.data.materials.new('vc'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes['Principled BSDF']; attr = nt.nodes.new('ShaderNodeVertexColor')
attr.layer_name = o.data.color_attributes[0].name if o.data.color_attributes else ''
nt.links.new(attr.outputs['Color'], bsdf.inputs['Base Color']); bsdf.inputs['Roughness'].default_value = 0.8
o.data.materials.append(mat)
sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = 16
sc.render.resolution_x = 900; sc.render.resolution_y = 650
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.85, 0.85, 0.85, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
bpy.ops.object.light_add(type='SUN', location=(1, -1, 2)); bpy.context.object.data.energy = 3.0
bpy.context.object.rotation_euler = (0.7, 0.3, 0.6)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'
pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))); mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr = (mn + mx) / 2; size = max(mx - mn); cam.data.ortho_scale = size * 1.12
for name, d in [('side', Vector((1, 0, 0))), ('three_q', Vector((0.8, -0.6, 0.4))), ('under', Vector((0.35, 0.0, -1))), ('front', Vector((0, -1, 0.15))), ('top', Vector((0.2, 0.0, 1))), ('back', Vector((-0.5, 0.8, 0.3)))]:
    cam.location = ctr + d.normalized() * size * 3; cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = f'{pre}_{name}.png'; bpy.ops.render.render(write_still=True)
