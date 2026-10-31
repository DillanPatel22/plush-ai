"""Turn an aligned real-dog mesh into a plush-shaped, perfectly symmetrical, watertight solid.

Steps (all in a voxel field so the result is always watertight):
 1. inside/outside from the generalised winding number (robust to holes in the raw mesh)
 2. mirror-average across X = 0 (symmetry by construction)
 3. morphological closing: fills crevices stuffing can't hold (toe splits, lip line, ear-to-head gaps,
    armpits) - anything narrower than 2 x close_mm
 4. gaussian rounding: softens edges the way stuffing does
 5. flat soles: everything below sole_mm is cut so paws (and later the sitting bottom) stand flat; the lower
    legs are then stretched back so the dog keeps its full height (only the toe curve is lost)
 6. marching cubes -> smooth dense surface

Ears (optional, --ear_outline): the raw model's ears are fused to the head, so they are removed in the voxel
field and the skull beneath is filled smoothly (see ear_fill.py).

Usage: python3 plushify.py aligned.ply out.ply [--pitch 3] [--close 12] [--close_head 6] [--round 3]
       [--round_body 10] [--sole 14] [--ear_outline ears/ear_outline_rel.npy] [--keep_height]
"""
import argparse
import json
import time

import igl
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ear_fill import ear_fill


ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--pitch", type=float, default=4.0, help="voxel size, mm")
ap.add_argument("--close", type=float, default=12.0, help="closing radius, mm")
ap.add_argument("--round", type=float, default=5.0, help="gaussian rounding sigma on the head, mm")
ap.add_argument("--round_body", type=float, default=10.0, help="gaussian rounding sigma on body, legs and tail, mm")
ap.add_argument("--head_end", type=float, default=230.0, help="head/neck blend point, mm from nose tip")
ap.add_argument("--sole", type=float, default=6.0, help="flat-sole cut height, mm")
ap.add_argument("--close_head", type=float, default=None, help="closing radius on the head, mm (default = --close)")
ap.add_argument("--ear_outline", default=None, help="ear outline npy (Y from nose tip, Z from ground), side view")
ap.add_argument("--ear_buffer", type=float, default=8.0, help="grow the ear outline by this much, mm")
ap.add_argument("--ear_root", type=float, default=12.0, help="ear-root cushion kept on the skull's top corners, mm")
ap.add_argument("--ear_mask", default=None, help="optional side-view ear mask npz from the 3D model's texture (mask, cell, y0, z0)")
ap.add_argument("--keep_height", action="store_true", help="stretch the lower legs back after the sole cut")
a = ap.parse_args()
t0 = time.time()

m = trimesh.load(a.src)
V = np.asarray(m.vertices, dtype=np.float64); F = np.asarray(m.faces, dtype=np.int64)
p = a.pitch
pad = a.close + 3 * max(a.round, 10.0) + 3 * p
# grid symmetric about X = 0 so mirroring is an exact index flip
xmax = np.abs(V[:, 0]).max() + pad
nx = int(np.ceil(xmax / p)); xs = (np.arange(-nx, nx + 1)) * p
ys = np.arange(V[:, 1].min() - pad, V[:, 1].max() + pad, p)
zs = np.arange(0.0 - pad, V[:, 2].max() + pad, p)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
Q = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
print(f"grid {X.shape} = {len(Q)/1e6:.1f} M points")
wn = igl.fast_winding_number(V, F, Q).reshape(X.shape)
print(f"winding number {time.time()-t0:.0f}s")

field = 0.5 * (wn + wn[::-1, :, :])              # mirror-average across X = 0
nose_y = V[:, 1].min()
ear_info = {}
if a.ear_outline:
    ol = np.load(a.ear_outline) + [nose_y, 0.0]
    em = None
    if a.ear_mask:
        d_ = np.load(a.ear_mask); em = (d_["mask"], float(d_["cell"]), float(d_["y0"]) + nose_y, float(d_["z0"]))
    field, ear_info = ear_fill(field, xs, ys, zs, p, ol, a.ear_buffer, extra_mask=em, root_mm=a.ear_root)
    field = 0.5 * (field + field[::-1, :, :])
    print("ears removed:", ear_info)
occ = field > 0.5
def close(o, mm):
    r = max(1, int(round(mm / p)))
    ball = np.linalg.norm(np.indices((2 * r + 1,) * 3) - r, axis=0) <= r
    return ndimage.binary_closing(np.pad(o, r), structure=ball)[r:-r, r:-r, r:-r]
head_y = ys < nose_y + a.head_end
if a.close_head is not None and a.close_head != a.close:
    occ = np.where(head_y[None, :, None], close(occ, a.close_head), close(occ, a.close))   # face keeps its stop, brow, lips
else:
    occ = close(occ, a.close)
occ = ndimage.binary_fill_holes(occ)
# keep the largest connected solid
lab, n = ndimage.label(occ)
if n > 1:
    sizes = ndimage.sum(occ, lab, range(1, n + 1)); occ = lab == 1 + int(np.argmax(sizes))
# stronger rounding on the body (muscle bumps can't be sewn) and lighter on the face (keeps likeness)
f = occ.astype(np.float32)
soft_head = ndimage.gaussian_filter(f, sigma=a.round / p)
soft_body = ndimage.gaussian_filter(f, sigma=a.round_body / p)
w_head = 1.0 / (1.0 + np.exp(-((nose_y + a.head_end) - ys) / 25.0))      # 1 on the head, 0 on the body
soft = soft_body + (soft_head - soft_body) * w_head[None, :, None]
soft = 0.5 * (soft + soft[::-1, :, :])            # keep exact symmetry after filtering
soft[:, :, zs < a.sole] = 0.0                      # flat soles
verts, faces, _, _ = measure.marching_cubes(soft, level=0.5, spacing=(p, p, p))
verts += np.array([xs[0], ys[0], zs[0]])
shift = verts[:, 2].min()
verts[:, 2] -= shift                               # stand on Z = 0
if a.keep_height:
    # the sole cut removes the toe curve; stretch the lower legs (below 25 % of height) so height is kept
    s_k = 0.25 * verts[:, 2].max()
    verts[:, 2] += shift * np.clip(verts[:, 2] / s_k, 0, 1)
out = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
out.fix_normals()
out.export(a.dst)
info = {**ear_info, "keep_height": a.keep_height, "close_head_mm": a.close_head, "pitch_mm": p, "close_mm": a.close, "round_head_mm": a.round, "round_body_mm": a.round_body, "head_end_mm": a.head_end, "sole_mm": a.sole,
        "vertices": int(len(out.vertices)), "faces": int(len(out.faces)),
        "watertight": bool(out.is_watertight), "volume_l": round(out.volume / 1e6, 2),
        "seconds": round(time.time() - t0)}
json.dump(info, open(a.dst.rsplit(".", 1)[0] + "_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
print("RESULT:", "PASS" if out.is_watertight else "FAIL")
