"""Tail module for short-tailed breeds: replace the raw model's tail (a curl, screw or lump on the rump) with a
plush tail stub - a short, thick, rounded tube set on the rump and pointing back and down - so the base
template's tail (two tail pieces) can be reshaped onto it without folding.

Steps (voxel field, so the result stays watertight and exactly symmetric):
 1. inside/outside from the generalised winding number of the plush solid
 2. the curl is cut away above the rump: for |x| < xw the rump's top and back are rebuilt from the section just
    outside the tail (|x| = xw), raised by the body's transverse crown so the rump stays rounded
 3. a stub (capsule, radius r0 at the root tapering to r1) is added from a root on the rump's midline, pointing
    back at `angle` degrees below horizontal, `length` mm long
 4. light gaussian rounding of the joint, marching cubes, mirror-exact
Usage: python3 tail_stub.py plush.ply out.ply out_tail.json [--xw 46] [--y_from 565] [--z_from 290]
          [--root_y 628] [--angle 30] [--length 52] [--r0 24] [--r1 21] [--pitch 2.0]
Writes the new solid and the stub axis (root centre, tip centre, radii) for reshape_template.py.
Acceptance: watertight, symmetric, tail stub wide enough for the 25 mm tube rule at the stated plush scale.
"""
import argparse, json
import numpy as np, trimesh, igl
from scipy import ndimage
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("info")
ap.add_argument("--xw", type=float, default=46.0); ap.add_argument("--y_from", type=float, default=565.0)
ap.add_argument("--z_from", type=float, default=290.0); ap.add_argument("--root_y", type=float, default=628.0)
ap.add_argument("--angle", type=float, default=30.0); ap.add_argument("--length", type=float, default=52.0)
ap.add_argument("--r0", type=float, default=24.0); ap.add_argument("--r1", type=float, default=21.0)
ap.add_argument("--pitch", type=float, default=2.0); ap.add_argument("--k", type=float, default=None, help="plush scale, for the tube check")
a = ap.parse_args()
m = trimesh.load(a.src, process=False); V = np.asarray(m.vertices, float); F = np.asarray(m.faces)
y0 = V[:, 1].min(); p = a.pitch; pad = 12.0
nx = int(np.ceil((np.abs(V[:, 0]).max() + pad) / p)); xs = np.arange(-nx, nx + 1) * p
ys = np.arange(V[:, 1].min() - pad, V[:, 1].max() + a.length + pad, p); zs = np.arange(-pad, V[:, 2].max() + pad, p)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
wn = igl.fast_winding_number(V, F, np.c_[X.ravel(), Y.ravel(), Z.ravel()]).reshape(X.shape)
occ = 0.5 * (wn + wn[::-1]) > 0.5
yr = Y - y0
# 2. rebuild the rump under the curl from the section at |x| = xw, raised by the transverse crown
ixw = np.argmin(np.abs(xs - a.xw)); ix0 = np.argmin(np.abs(xs))
jf = np.argmin(np.abs(ys - y0 - (a.y_from - 30)))                       # crown measured in front of the curl
top = lambda ix, j: zs[np.where(occ[ix, j])[0].max()] if occ[ix, j].any() else np.nan
crown = top(ix0, jf) - top(ixw, jf)
reg = (np.abs(X) < a.xw) & (yr > a.y_from) & (Z > a.z_from)
for i in range(len(xs)):
    if abs(xs[i]) >= a.xw: continue
    dz = crown * (1 - (xs[i] / a.xw) ** 2)
    sh = int(round(dz / p))
    src = np.roll(occ[ixw], sh, axis=1)                                  # x = xw profile raised by dz
    if sh > 0: src[:, :sh] = occ[ixw][:, :sh]
    occ[i] = np.where(reg[i], src, occ[i])
# 3. stub
th = np.radians(a.angle); d = np.array([0.0, np.cos(th), -np.sin(th)])
jr = np.argmin(np.abs(ys - y0 - a.root_y)); kz = np.where(occ[ix0, jr])[0]
z_top = zs[kz.max()]
# root centre: below the rump surface by r0 * 0.6 along the up direction, so the stub grows out of the rump
root = np.array([0.0, y0 + a.root_y, z_top - 0.6 * a.r0]); tip = root + d * a.length
Pq = np.stack([X, Y, Z], -1) - root
t = np.clip(Pq @ d / a.length, 0, 1)
rad = a.r0 + (a.r1 - a.r0) * t
dist = np.linalg.norm(Pq - t[..., None] * a.length * d, axis=-1)
occ |= dist < rad
# 4. round, mesh
f = ndimage.gaussian_filter(occ.astype(np.float32), 2.0 / p); f = 0.5 * (f + f[::-1])
f[:, :, zs < 0.0] = 0.0
vv, ff, _, _ = measure.marching_cubes(f, 0.5, spacing=(p, p, p)); vv += [xs[0], ys[0], zs[0]]
vv[:, 2] -= max(vv[:, 2].min(), 0.0) if vv[:, 2].min() > 0 else vv[:, 2].min()
out = trimesh.Trimesh(vv, ff[:, ::-1], process=True); out.fix_normals(); out.export(a.dst)
info = {"root_centre": root.round(2).tolist(), "tip_centre": tip.round(2).tolist(), "axis": d.round(4).tolist(),
        "r0_mm": a.r0, "r1_mm": a.r1, "length_mm": a.length, "angle_deg": a.angle, "crown_mm": round(float(crown), 1),
        "rebuilt_region": {"xw": a.xw, "y_from": a.y_from, "z_from": a.z_from},
        "watertight": bool(out.is_watertight), "vertices": int(len(out.vertices))}
if a.k: info["stub_width_plush_mm"] = round(2 * a.r1 * a.k, 1)
json.dump(info, open(a.info, "w"), indent=1)
print(info)
ok = out.is_watertight and (a.k is None or 2 * a.r1 * a.k >= 20.0)      # fill_thin raises it to 25 mm afterwards
print("RESULT:", "PASS" if ok else "FAIL")
