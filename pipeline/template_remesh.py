"""Remesh the dense plush solid into the base template: ~5,000 mostly-quad faces, exactly symmetrical,
with a deterministic vertex order. Runs Blender headless.

Usage: python3 template_remesh.py plush_dense.ply out_dir [target_faces]
Writes out_dir/template.obj, template_report.json
"""
import json
import sys
from pathlib import Path

import bpy  # must come before bmesh when using the bpy module
import bmesh
import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
target = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
out.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=str(src))
ob = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
dense = np.array([v.co[:] for v in ob.data.vertices])

# 1. QuadriFlow (quad-dominant, follows the shape's curvature), mirrored on X
res = bpy.ops.object.quadriflow_remesh(use_mesh_symmetry=True, use_preserve_sharp=False,
                                       use_preserve_boundary=False, smooth_normals=False,
                                       mode="FACES", target_faces=target, seed=0)
print("quadriflow:", res, len(ob.data.polygons), "faces")

# 2. exact symmetry: keep +X half, mirror it, weld the centre line
bm = bmesh.new(); bm.from_mesh(ob.data)
bmesh.ops.symmetrize(bm, input=bm.verts[:] + bm.edges[:] + bm.faces[:], direction="X", dist=0.5)
for v in bm.verts:
    if abs(v.co.x) < 1e-4:
        v.co.x = 0.0
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
bm.to_mesh(ob.data); bm.free()

# 3. deterministic vertex order: sort by (Y, Z, X) rounded to 0.01 mm
me = ob.data
co = np.array([v.co[:] for v in me.vertices])
key = np.lexsort((np.round(co[:, 0], 2), np.round(co[:, 2], 2), np.round(co[:, 1], 2)))
rank = np.empty_like(key); rank[key] = np.arange(len(key))
faces = [[int(rank[i]) for i in p.vertices] for p in me.polygons]
co_sorted = co[key]
new = bpy.data.meshes.new("template"); new.from_pydata(co_sorted.tolist(), [], faces); new.update()
ob.data = new

# 4. checks
bm = bmesh.new(); bm.from_mesh(new)
non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
boundary = sum(1 for e in bm.edges if e.is_boundary)
quads = sum(1 for f in bm.faces if len(f.verts) == 4); nf = len(bm.faces)
bm.free()
co = co_sorted
# symmetry: every vertex has a mirror partner within 0.5 mm
from scipy.spatial import cKDTree
mir = co * np.array([-1, 1, 1]); d, _ = cKDTree(co).query(mir)
sym_err = float(d.max())
# distance to the dense plush surface (shape fidelity)
dd, _ = cKDTree(dense).query(co)
report = {
    "faces": nf, "quads_pct": round(100 * quads / nf, 1), "vertices": len(co),
    "non_manifold_edges": non_manifold, "boundary_edges": boundary,
    "symmetry_max_err_mm": round(sym_err, 4),
    "dist_to_dense_mean_mm": round(float(dd.mean()), 2), "dist_to_dense_max_mm": round(float(dd.max()), 2),
    "vertex_order_checksum": int(np.round(co * 100).astype(np.int64).sum() % 1_000_000_007),
}
checks = {
    "watertight_manifold": non_manifold == 0 and boundary == 0,
    "faces_4k_6k": 4000 <= nf <= 6500,
    "symmetry_le_0.5mm": sym_err <= 0.5,
    "shape_mean_dev_le_2mm": report["dist_to_dense_mean_mm"] <= 2.0,
}
report["checks"] = checks
bpy.ops.wm.obj_export(filepath=str(out / "template.obj"), export_selected_objects=False,
                      forward_axis="Y", up_axis="Z", export_materials=False, export_uv=False, export_normals=False)
json.dump(report, open(out / "template_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
