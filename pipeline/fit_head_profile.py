"""Fit the head's side profile to the side reference photo (a reshape: vertex count and order unchanged).

The 3D model generator softens the stop and tapers the muzzle. For every Y from the nose back to the neck, the
head's top and bottom outline heights are read from the photo (scaled by withers height, nose tip and ground
aligned) and from the mesh; head vertices are remapped vertically so the mesh outline matches the photo's
(linear between bottom and top at each Y), fading out over the neck. X is untouched, so symmetry is exact.

Usage: python3 fit_head_profile.py work_dir mesh.obj out.obj [--fade 170 230] [--zmin 420]
Ends with an acceptance test: head outline within 2 mm of the photo outline (p90) for Y < 170 mm.
"""
import json, sys
import numpy as np, trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import align_measure as am

W, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
F0, F1 = (float(sys.argv[sys.argv.index("--fade") + 1]), float(sys.argv[sys.argv.index("--fade") + 2])) if "--fade" in sys.argv else (170.0, 230.0)
tr = json.load(open(f"{W}/ref/face_trace.json"))
T = trimesh.load(src, process=False); V = np.asarray(T.vertices, float).copy(); F = np.asarray(T.faces)
tri = np.r_[F[:, :3], F[:, [0, 2, 3]]] if F.shape[1] == 4 else F
H, WD = 800, 320
ZMIN = float(sys.argv[sys.argv.index("--zmin") + 1]) if "--zmin" in sys.argv else 420.0   # head above this (Labrador 420)

def side_mask(Vv):
    im = Image.new("L", (WD, H), 0); d = ImageDraw.Draw(im); y0 = Vv[:, 1].min()
    P = np.c_[Vv[:, 1] - y0, H - Vv[:, 2]]
    for t in tri: d.polygon([tuple(P[i]) for i in t], fill=255)
    return np.array(im) > 0

def profile(mask):
    top = np.full(WD, np.nan); bot = np.full(WD, np.nan)
    for y in range(WD):
        col = np.where(mask[: int(H - ZMIN), y])[0]
        if len(col): top[y] = H - col.min(); bot[y] = H - col.max()
    return top, bot

ph = am.photo_mask(f"{W}/{tr['side_image']}"); rs, _ = am.measure_mask(ph, 1.0)
s = tr["withers_mm"] / rs["withers_height_mm"]; nose_px = np.where(ph.any(0))[0].min(); ground = rs["landmarks_px"]["ground"]
rr, cc = np.nonzero(ph); pm = np.zeros((H, WD), bool)
yy = np.round((cc - nose_px) * s).astype(int); zz = np.round(H - (ground - rr) * s).astype(int)
ok = (yy >= 0) & (yy < WD) & (zz >= 0) & (zz < H); pm[zz[ok], yy[ok]] = True
pm = ndimage.binary_closing(pm, iterations=2)
tp, bp = profile(pm)
tm, bm = profile(side_mask(V))

def smooth(a, n=5):
    a = a.copy(); ok_ = np.isfinite(a); a[~ok_] = np.interp(np.where(~ok_)[0], np.where(ok_)[0], a[ok_]); return ndimage.uniform_filter1d(a, n)
tp, bp, tm, bm = (smooth(a, 7) for a in (tp, bp, tm, bm))     # 7 mm: keeps the stop, drops fur wisps
yv = np.arange(WD, dtype=float)
w = np.clip((F1 - yv) / (F1 - F0), 0, 1); w = w * w * (3 - 2 * w)             # 1 on the head, 0 behind the neck

y0 = V[:, 1].min(); yr = V[:, 1] - y0
sel = (yr < F1) & (V[:, 2] > ZMIN)
yi = np.clip(yr[sel], 0, WD - 1)
t_m, b_m, t_p, b_p, ww = (np.interp(yi, yv, a) for a in (tm, bm, tp, bp, w))
# remap only between this column's bottom and top; vertices below the head's bottom line (neck/chest) keep z
z = V[sel, 2]; u = (z - b_m) / np.maximum(t_m - b_m, 1e-6)
z_new = np.where(u >= 0, b_p + u * (t_p - b_p), z + (b_p - b_m) * np.clip(1 + u * 4, 0, 1))
V[sel, 2] = z + ww * (z_new - z)
V[np.abs(V[:, 0]) < 1e-9, 0] = 0.0
out = trimesh.load(src, process=False); out.vertices = V; out.export(dst)

tm2, bm2 = map(smooth, profile(side_mask(V)))
dev = np.r_[np.abs(tm2 - tp)[:int(F0)], np.abs(bm2 - bp)[:int(F0)]]
mv = np.abs(V[:, 2] - np.asarray(T.vertices)[:, 2])
rep = {"outline_dev_p90_mm": round(float(np.percentile(dev, 90)), 2), "outline_dev_max_mm": round(float(dev.max()), 2),
       "before_dev_p90_mm": round(float(np.percentile(np.r_[np.abs(tm - tp)[:int(F0)], np.abs(bm - bp)[:int(F0)]], 90)), 2),
       "max_vertex_move_mm": round(float(mv.max()), 1), "vertices": int(len(V))}
json.dump(rep, open(dst.rsplit(".", 1)[0] + "_headfit.json", "w"), indent=1)
print(rep)
print("RESULT:", "PASS" if rep["outline_dev_p90_mm"] <= 2.0 and len(V) == len(T.vertices) else "FAIL")
