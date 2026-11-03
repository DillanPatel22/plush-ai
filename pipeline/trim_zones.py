"""Fur trim zones for a breed: which length the craftsman trims each area to, on the 3D shape and on every piece.

One white stock fur per plush (design rule), long enough for the longest zone; the craftsman trims each zone.
Zones and starting lengths come from a breed finish spec (work/<breed>/finish_spec.json), lengths in mm at the
plush size given there and scaled linearly for other sizes (fur in proportion to the dog).

Usage: python3 trim_zones.py finish_spec.json seam_mesh.obj face_labels.npy flat_dir pieces_dir nap.json k out_prefix
Writes out_prefix_faces.npy (zone index per seam-mesh face), out_prefix.ply (3D, zone colours),
out_prefix_pieces.png (pieces at one scale, zone colours, pile arrows).
"""
import json, math, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

spec, smesh, flab, fdir, pdir, napj, k, pre = sys.argv[1:9]; k = float(k)
sp_ = json.load(open(spec)); Z = sp_["trim_zones"]
S = trimesh.load(smesh, process=False); lab = np.load(flab)
C = S.triangles_center; x, y, z = C.T; y = y - S.vertices[:, 1].min()
head = np.isin(lab, [1, 2, 3, 4, 21, 22])
y_line = np.interp(z, [560, 600, 650, 700], [102, 95, 86, 78])
muzzle = head & ((np.isin(lab, [21, 22])) | ((lab == 1) & (y < 95)) | ((lab == 4) & (y < 120)))
front_leg = (y > 230) & (y < 430); rear_leg = (y > 690) & (y < 910)
legs = ((front_leg & (z < 235)) | (rear_leg & (z < 260))) & ~head
tail = np.isin(lab, [12, 13])
neck_chest = np.isin(lab, [9, 26]) | (~head & (y < 330) & (z > 430))
zone = np.full(len(C), Z.index(next(q for q in Z if q["zone"] == "body")))
for name, m_ in [("neck and chest", neck_chest), ("legs", legs), ("tail", tail), ("head and ears", head & ~muzzle), ("muzzle", muzzle)]:
    zone[m_] = Z.index(next(q for q in Z if q["zone"] == name))
np.save(pre + "_faces.npy", zone)
pal = np.array([q["colour"] for q in Z], np.uint8)
vis = trimesh.Trimesh(S.vertices[S.faces].reshape(-1, 3), np.arange(len(S.faces) * 3).reshape(-1, 3), process=False)
vis.visual.vertex_colors = np.c_[np.repeat(pal[zone], 3, 0), np.full(len(zone) * 3, 255, np.uint8)]
vis.export(pre + ".ply")

# pieces at one scale, coloured by zone (piece faces -> nearest seam-mesh face)
lay = json.load(open(Path(pdir) / "pieces.json"))["pieces"]; nap = json.load(open(napj))["pieces"]
tree = cKDTree(C); PX = 3.0; items = []
for pid, info in lay.items():
    if info.get("mirror_of") or not info.get("faces"): continue
    d = np.load(Path(fdir) / f"{pid}.npz"); V3, F, uv = d["V3"], d["F"], d["uv"].copy(); r = nap[pid]
    if r["flip_u"]: uv[:, 0] *= -1
    a = r["rotation_rad"]; R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); uv = uv @ R.T
    fz = zone[tree.query(V3[F].mean(1) / k)[1]]
    items.append((pid, uv, F, fz, 2 if any(i.get("mirror_of") == pid for i in lay.values()) else 1))
items.sort(key=lambda t: -(t[1][:, 1].max() - t[1][:, 1].min()))
W = 2400; pad = 14; xx = yy = row = 0; place = []
for it in items:
    mn, mx = it[1].min(0), it[1].max(0); w, h = (mx - mn) * PX + 2 * pad + [0, 26]
    if xx + w > W: xx = 0; yy += row; row = 0
    place.append((it, mn, mx, xx, yy)); xx += w; row = max(row, h)
img = Image.new("RGB", (W, int(yy + row + 10)), "white"); d = ImageDraw.Draw(img)
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
for (pid, uv, F, fz, n), mn, mx, x0, y0 in place:
    P = (uv - mn) * PX; P[:, 1] = (mx[1] - mn[1]) * PX - P[:, 1]; P += [x0 + pad, y0 + pad + 24]
    for t, zz in zip(F, fz): d.polygon([tuple(P[i]) for i in t], fill=tuple(int(v) for v in pal[zz]))
    d.text((x0 + pad, y0 + 2), pid + (" x2" if n == 2 else ""), fill=(30, 30, 30), font=f)
img.save(pre + "_pieces.png")
share = {q["zone"]: round(float((zone == i).mean()), 3) for i, q in enumerate(Z)}
print("zone share of surface:", share)
print("RESULT:", "PASS" if all(v > 0 for v in share.values()) else "CHECK (a zone is empty)")
