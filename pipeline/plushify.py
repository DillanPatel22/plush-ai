"""Turn an aligned real-dog mesh into a plush-shaped, perfectly symmetrical, watertight solid.

Steps (all in a voxel field so the result is always watertight):
 1. inside/outside from the generalised winding number (robust to holes in the raw mesh)
 2. mirror-average across X = 0 (symmetry by construction)
 3. morphological closing: fills crevices stuffing can't hold (toe splits, lip line, ear-to-head gaps,
    armpits) - anything narrower than 2 x close_mm
 4. gaussian rounding: softens edges the way stuffing does
 5. flat soles: everything below sole_mm is cut so paws (and later the sitting bottom) stand flat
 6. marching cubes -> smooth dense surface

Usage: python3 plushify.py aligned.ply out.ply [--pitch 4] [--close 12] [--round 6] [--sole 6]
"""
import argparse
import json
import time

import igl
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--pitch", type=float, default=4.0, help="voxel size, mm")
ap.add_argument("--close", type=float, default=12.0, help="closing radius, mm")
ap.add_argument("--round", type=float, default=5.0, help="gaussian rounding sigma on the head, mm")
ap.add_argument("--round_body", type=float, default=10.0, help="gaussian rounding sigma on body, legs and tail, mm")
ap.add_argument("--head_end", type=float, default=230.0, help="head/neck blend point, mm from nose tip")
ap.add_argument("--sole", type=float, default=6.0, help="flat-sole cut height, mm")
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
occ = field > 0.5
r = max(1, int(round(a.close / p)))
ball = np.linalg.norm(np.indices((2 * r + 1,) * 3) - r, axis=0) <= r
occ = ndimage.binary_closing(np.pad(occ, r), structure=ball)[r:-r, r:-r, r:-r]
occ = ndimage.binary_fill_holes(occ)
# keep the largest connected solid
lab, n = ndimage.label(occ)
if n > 1:
    sizes = ndimage.sum(occ, lab, range(1, n + 1)); occ = lab == 1 + int(np.argmax(sizes))
# stronger rounding on the body (muscle bumps can't be sewn) and lighter on the face (keeps likeness)
f = occ.astype(np.float32)
soft_head = ndimage.gaussian_filter(f, sigma=a.round / p)
soft_body = ndimage.gaussian_filter(f, sigma=a.round_body / p)
nose_y = V[:, 1].min()
w_head = 1.0 / (1.0 + np.exp(-((nose_y + a.head_end) - ys) / 25.0))      # 1 on the head, 0 on the body
soft = soft_body + (soft_head - soft_body) * w_head[None, :, None]
soft = 0.5 * (soft + soft[::-1, :, :])            # keep exact symmetry after filtering
soft[:, :, zs < a.sole] = 0.0                      # flat soles
verts, faces, _, _ = measure.marching_cubes(soft, level=0.5, spacing=(p, p, p))
verts += np.array([xs[0], ys[0], zs[0]])
verts[:, 2] -= verts[:, 2].min()                   # stand on Z = 0
out = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
out.fix_normals()
out.export(a.dst)
info = {"pitch_mm": p, "close_mm": a.close, "round_head_mm": a.round, "round_body_mm": a.round_body, "head_end_mm": a.head_end, "sole_mm": a.sole,
        "vertices": int(len(out.vertices)), "faces": int(len(out.faces)),
        "watertight": bool(out.is_watertight), "volume_l": round(out.volume / 1e6, 2),
        "seconds": round(time.time() - t0)}
json.dump(info, open(a.dst.rsplit(".", 1)[0] + "_info.json", "w"), indent=1)
print(json.dumps(info, indent=1))
print("RESULT:", "PASS" if out.is_watertight else "FAIL")
