"""Ear module from the side-view ear mask: a flat two-layer ear piece and its attachment line on the head.

Drop ears lie close to the head, so the side view shows their true shape except where they curve
around the skull; that is corrected by the mean tilt of the ear surface (area ratio).

Usage: python3 ear_module.py aligned_merged.ply ear_mask2d.npz template.obj out_dir
Writes ear_outline_mm.npy (closed outline, real dog scale, u = front->back, v = up),
ear_attach_idx.npy (outline indices on the attachment edge), ear_attach_on_template.npy, ear_module.json
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.signal import savgol_filter
from skimage import measure

src, mpath, tpath, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
d = np.load(mpath); mask = d['mask'].astype(float); cell = float(d['cell']); outer = d['outer']; y0 = float(d['y0']); z0 = float(d['z0'])
cs = measure.find_contours(np.pad(mask, 1), 0.5)
c = max(cs, key=len) - 1.0                       # (iy, iz) in cells
yz = np.c_[y0 + (c[:, 0] + 0.5) * cell, z0 + (c[:, 1] + 0.5) * cell]
# resample evenly (2 mm) and smooth the jagged cell outline
seg = np.linalg.norm(np.diff(yz, axis=0, append=yz[:1]), axis=1); s = np.r_[0, np.cumsum(seg)][:-1]; L = seg.sum()
t = np.arange(0, L, 2.0)
yz = np.c_[np.interp(t, s, yz[:, 0], period=L), np.interp(t, s, yz[:, 1], period=L)]
w = 15 if len(yz) > 30 else (len(yz) // 2) * 2 - 1
yz = np.c_[savgol_filter(yz[:, 0], w, 3, mode='wrap'), savgol_filter(yz[:, 1], w, 3, mode='wrap')]
# tilt correction: ear surface normals vs the side-view direction
m = trimesh.load(src, process=False)
V = np.asarray(m.vertices); N = np.asarray(m.vertex_normals)
ny, nz = mask.shape
iy = ((V[:, 1] - y0) / cell).astype(int); iz = ((V[:, 2] - z0) / cell).astype(int)
ok = (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz) & (V[:, 0] > 30)
sel = np.zeros(len(V), bool); sel[ok] = (mask[iy[ok], iz[ok]] > 0.5) & (V[ok, 0] >= outer[iy[ok], iz[ok]] - 8)
cos = np.clip(np.abs(N[sel, 0]), 0.3, 1.0).mean()
k = 1.0 / np.sqrt(cos)                          # spread the correction over both directions
ctr = yz.mean(0)
outline = (yz - ctr) * k + ctr
# attachment edge: the top of the ear where it meets the skull (top 15 % of its height)
zt = outline[:, 1].max(); h = np.ptp(outline[:, 1])
attach = np.where(outline[:, 1] >= zt - 0.15 * h)[0]
# put the attachment points onto the ear-less template (right side), ordered front -> back
T = trimesh.load(tpath, process=False)
xs = np.interp(yz[attach, 0], np.arange(ny) * cell + y0, np.zeros(ny)) + 200.0       # start well outside, project inwards
pts = np.c_[np.full(len(attach), 200.0), yz[attach, 0], yz[attach, 1]]
on, dist, _ = trimesh.proximity.closest_point(T, pts)
on = on[np.argsort(on[:, 1])]
np.save(out / "ear_outline_mm.npy", outline)
np.save(out / "ear_attach_idx.npy", attach)
np.save(out / "ear_attach_on_template.npy", on)
area = 0.5 * abs(np.dot(outline[:, 0], np.roll(outline[:, 1], 1)) - np.dot(outline[:, 1], np.roll(outline[:, 0], 1)))
c_ = outline - outline.mean(0); _, sv, vt = np.linalg.svd(c_, full_matrices=False); ext = np.ptp(c_ @ vt.T, axis=0)
info = {"tilt_correction": round(float(k), 3), "ear_area_cm2_real": round(area / 100, 1),
        "ear_length_mm_real": round(float(ext[0]), 1), "ear_width_mm_real": round(float(ext[1]), 1),
        "attach_line_mm_real": round(float(np.linalg.norm(np.diff(on, axis=0), axis=1).sum()), 1)}
json.dump(info, open(out / "ear_module.json", "w"), indent=1)
print(info)
plausible = 100 <= info["ear_length_mm_real"] <= 200 and 60 <= info["ear_width_mm_real"] <= 140
print("RESULT:", "PASS" if plausible else "CHECK")
