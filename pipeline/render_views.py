"""Render review views of a mesh with Blender (Cycles, CPU, headless).

Usage: python3 render_views.py mesh.(glb|ply|obj) out_dir [--plain] [--views side,front,three_quarter,back,top]
Mesh must already be in the project frame (X side, Y nose->tail with nose at -Y, Z up, mm).
--plain renders a soft white fabric-like material instead of any texture.
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

args = sys.argv[1:]
src, out = Path(args[0]), Path(args[1])
plain = "--plain" in args
views = "side,front,three_quarter,back,top"
if "--views" in args:
    views = args[args.index("--views") + 1]
views = views.split(",")
out.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = 1000, 750
scene.render.film_transparent = False
scene.unit_settings.scale_length = 0.001

ext = src.suffix.lower()
if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=str(src))
elif ext == ".ply":
    bpy.ops.wm.ply_import(filepath=str(src))
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=str(src), forward_axis="Y", up_axis="Z")
objs = [o for o in scene.objects if o.type == "MESH"]
# glTF importer converts Y-up to Z-up but keeps glTF forward; geometry passed here is already aligned in PLY/OBJ.
for o in objs:
    o.rotation_mode = "XYZ"
    if ext == ".ply":
        o.rotation_euler = (0, 0, 0)
    for p in o.data.polygons:
        p.use_smooth = True

if plain or not any(o.data.materials for o in objs):
    mat = bpy.data.materials.new("plush")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.93, 0.91, 0.87, 1)
    bsdf.inputs["Roughness"].default_value = 0.85
    for name in ("Sheen Weight", "Sheen"):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = 0.6
    for o in objs:
        o.data.materials.clear(); o.data.materials.append(mat)

# bounds
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr = (mn + mx) / 2
size = max(mx - mn)

# ground + world
bpy.ops.mesh.primitive_plane_add(size=size * 6, location=(ctr.x, ctr.y, mn.z))
ground = bpy.context.object
gm = bpy.data.materials.new("ground"); gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 1, 1, 1)
ground.data.materials.append(gm)
ground.is_shadow_catcher = False
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9

def add_light(loc, energy):
    bpy.ops.object.light_add(type="AREA", location=loc)
    l = bpy.context.object; l.data.energy = energy; l.data.size = size
    d = ctr - l.location; l.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
add_light(ctr + Vector((size, -size, size * 1.5)), 3e6 * (size / 1000) ** 2)
add_light(ctr + Vector((-size, size * 0.5, size)), 1.5e6 * (size / 1000) ** 2)

cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"; cam_data.ortho_scale = size * 1.15
cam_data.clip_end = size * 20
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam

dirs = {
    "side": Vector((1, 0, 0)),          # camera on the dog's left (+X)... looks at its left flank
    "front": Vector((0, -1, 0)),        # camera in front of the nose (-Y)
    "back": Vector((0, 1, 0)),
    "three_quarter": Vector((0.75, -0.65, 0.35)),
    "top": Vector((0.0001, 0, 1)),
}
for name in views:
    d = dirs[name].normalized()
    cam.location = ctr + d * size * 4
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Z" if name != "top" else "Y").to_euler()
    scene.render.filepath = str(out / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", name)
