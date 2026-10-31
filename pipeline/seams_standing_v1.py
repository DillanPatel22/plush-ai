"""Seam layout v1 for the standing dog template: label every face with its pattern piece and split.

Pieces (15 incl. ears, which are built separately):
  P01 head gusset (nose -> back of skull, centre top)       P02/P03 side head R/L
  P04 chin gusset (chin -> throat)                           P05/P06 body side R/L (incl. tail half)
  P07 underbody gusset (chest front, belly, inner legs)      P08-P11 soles (FR, FL, RR, RL)
Seams: head gusset edges; neck ring (head/body); centre back from nape to tail tip; centre rear from tail
tip down to the gusset; front and back edges of every leg (side/gusset); sole ovals.

Usage: python3 seams_standing_v1.py template_medium.obj out_dir
All geometry in the template frame at real dog scale (mm).
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
import scipy.sparse as sp
from scipy.spatial import cKDTree

src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
T = trimesh.load(src, process=False)
m = trimesh.Trimesh(T.vertices, T.faces, process=False)        # triangulate quads
for _ in range(2):
    v, f = trimesh.remesh.subdivide(m.vertices, m.faces); m = trimesh.Trimesh(v, f, process=False)
C = m.triangles_center; x, y, z = C.T; ax = np.abs(x)
# smoothed normals (from a heavily smoothed copy) so normal-based seams follow the big shape, not small bumps
ms_ = m.copy(); trimesh.smoothing.filter_taubin(ms_, lamb=0.5, nu=0.53, iterations=40)
N = ms_.face_normals
y0 = m.vertices[:, 1].min()
y = y - y0                                                     # nose tip at 0

# ---- neck ring: plane through nape A and throat B (Y, Z) ----
A = np.array([285.0, 660.0]); B = np.array([180.0, 470.0])
nrm = np.array([B[1] - A[1], -(B[0] - A[0])])                  # perpendicular to AB in (y, z)
s = (y - A[0]) * nrm[0] + (z - A[1]) * nrm[1]
if ((0 - A[0]) * nrm[0] + (590 - A[1]) * nrm[1]) > 0: s = -s   # make the nose side negative
head = (s < 0) & (z > 430)

g = np.interp(y, [0, 30, 80, 120, 170, 220, 270, 310], [10, 14, 18, 24, 32, 30, 24, 20])
c = np.interp(y, [20, 60, 120, 160, 185], [12, 22, 28, 30, 0])            # chin gusset stops at the jaw angle
label = np.full(len(C), -1)
label[head & (ax < g) & (N[:, 2] > 0.2)] = 1                                   # head gusset
# chin gusset: the underside strip of the jaw, within 18 mm of the head's lowest point at that position
yb = np.arange(0, 400, 5.0); zlow = np.full(len(yb), np.inf)
hv = m.vertices[(m.vertices[:, 2] > 430)]
for i, yy in enumerate(yb):
    sel_ = np.abs(hv[:, 1] - y0 - yy) < 5
    if sel_.any(): zlow[i] = hv[sel_, 2].min()
zlow_f = np.interp(y, yb[np.isfinite(zlow)], zlow[np.isfinite(zlow)])
label[head & (ax < c) & (z < zlow_f + 18) & (N[:, 2] < 0)] = 4                  # chin gusset
side_head = head & (label < 0)
label[side_head & (x >= 0)] = 2; label[side_head & (x < 0)] = 3
body = ~head
sole = body & (N[:, 2] < -0.85) & (z < 3)
front_leg = (y > 230) & (y < 430); rear_leg = (y > 690) & (y < 910)
leg_top = np.where(front_leg, 255.0, np.where(rear_leg, 320.0, 0.0))
inner_leg = body & (front_leg | rear_leg) & (z < leg_top) & (N[:, 0] * np.sign(x) < -0.05) & (ax > 6)
b = np.interp(y, [200, 320, 450, 600, 750, 900], [38, 48, 58, 60, 52, 44])     # belly gusset half-width
belly = body & (ax < b) & (N[:, 2] < -0.25) & (z < 360) & (z > 180) & (y > 200) & (y < 910)
cw = np.interp(z, [230, 330, 430, 480], [44, 40, 34, 30])                        # chest strip half-width
chest = body & (y < 330) & (N[:, 1] < -0.2) & (ax < cw) & (z < 480) & (z > 230)
under = (inner_leg | belly | chest) & ~sole
label[body & ~under & ~sole & (x >= 0)] = 5; label[body & ~under & ~sole & (x < 0)] = 6
label[under] = 7
# tail: separate left/right halves from the root (saddle at the root distorts the body side otherwise)
tail = body & (y > 895) & (z > 380) & ~sole
label[tail & (x >= 0)] = 12; label[tail & (x < 0)] = 13
# soles: which paw
paw = np.where(front_leg, 0, 2) + np.where(x >= 0, 0, 1)                    # FR, FL, RR, RL
label[sole] = 8 + paw[sole]

# ---- clean up: majority filter on face adjacency, then enforce mirror symmetry ----
adj = m.face_adjacency
Aj = sp.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(len(C), len(C))).tocsr()
for _ in range(12):
    new = label.copy()
    for i in range(len(C)):
        nb = label[Aj.indices[Aj.indptr[i]:Aj.indptr[i + 1]]]
        vals, cnt = np.unique(nb, return_counts=True)
        if cnt.max() >= 2 and vals[np.argmax(cnt)] != label[i]: new[i] = vals[np.argmax(cnt)]
    label = new
swap = {2: 3, 3: 2, 5: 6, 6: 5, 8: 9, 9: 8, 10: 11, 11: 10, 12: 13, 13: 12}
mir = cKDTree(C).query(C * [-1, 1, 1])[1]
neg = x < 0
label[neg] = [swap.get(l, l) for l in label[mir[neg]]]
# islands: keep each label's largest connected component, give the rest to the majority neighbour
from scipy.sparse.csgraph import connected_components
for it in range(3):
    for L in np.unique(label):
        idx = np.where(label == L)[0]
        sub = Aj[idx][:, idx]; n, lab = connected_components(sub, directed=False)
        if n <= 1: continue
        big = np.argmax(np.bincount(lab))
        for k in np.where(lab != big)[0]:
            i = idx[k]; nb = label[Aj.indices[Aj.indptr[i]:Aj.indptr[i + 1]]]; nb = nb[nb != L]
            if len(nb): label[i] = np.bincount(nb).argmax()

# ---- straighten the seams: smooth every seam curve within the surface (junction points stay put) ----
orig = m.copy()
VF = m.vertex_faces
vlabels = [set(label[f] for f in row if f >= 0) for row in VF]
nlab = np.array([len(s_) for s_ in vlabels])
E = m.edges_unique
def pair(i): return tuple(sorted(vlabels[i])) if nlab[i] == 2 else None
nbrs = {}
for a_, b_ in E:
    if nlab[a_] >= 2 and nlab[b_] >= 2:
        pa, pb = pair(a_), pair(b_)
        # boundary edge: both faces on the edge carry different labels
        nbrs.setdefault(a_, []).append(b_); nbrs.setdefault(b_, []).append(a_)
# keep only true seam edges (the edge separates two different labels)
ef = m.face_adjacency_edges; fa = m.face_adjacency
seam_edge = set()
for (u_, w_), (f1, f2) in zip(ef, fa):
    if label[f1] != label[f2]: seam_edge.add((min(u_, w_), max(u_, w_)))
nb_seam = {}
for u_, w_ in seam_edge:
    nb_seam.setdefault(u_, []).append(w_); nb_seam.setdefault(w_, []).append(u_)
movable = [v_ for v_, nb_ in nb_seam.items() if len(nb_) == 2 and nlab[v_] == 2]
Vs = m.vertices.copy()
for it in range(30):
    newV = Vs.copy()
    for v_ in movable:
        a_, b_ = nb_seam[v_]
        newV[v_] = 0.5 * Vs[v_] + 0.25 * (Vs[a_] + Vs[b_])
    cp, _, _ = trimesh.proximity.closest_point(orig, newV[movable])
    newV[movable] = cp
    Vs = newV
# keep the straightened seams exactly symmetric (left pieces are mirrors of right pieces)
mv_idx = cKDTree(orig.vertices).query(orig.vertices * [-1, 1, 1])[1]
Vs = 0.5 * (Vs + Vs[mv_idx] * [-1, 1, 1])
Vs[np.abs(orig.vertices[:, 0]) < 1e-6, 0] = 0.0
m = trimesh.Trimesh(Vs, m.faces, process=False)
print("seam vertices smoothed:", len(movable))

names = {1: "P01 head gusset", 2: "P02 side head R", 3: "P03 side head L", 4: "P04 chin gusset",
         5: "P05 body side R", 6: "P06 body side L", 7: "P07 underbody gusset",
         8: "P08 sole front R", 9: "P09 sole front L", 10: "P10 sole rear R", 11: "P11 sole rear L",
         12: "P12 tail R", 13: "P13 tail L"}
report = {}
for L, nm in names.items():
    fi = np.where(label == L)[0]
    piece = m.submesh([fi], append=True)
    piece.export(out / f"P{L:02d}.obj")
    euler = piece.euler_number; loops = len(trimesh.graph.split(piece.edges[trimesh.grouping.group_rows(piece.edges_sorted, require_count=1)], only_watertight=False)) if False else None
    report[f"P{L:02d}"] = {"name": nm, "faces": int(len(fi)), "area_cm2_real": round(piece.area / 100, 1), "euler": int(euler)}
np.save(out / "face_labels.npy", label)
m.export(out / "subdivided.obj")
# coloured preview mesh
pal = np.array([[0, 0, 0], [230, 80, 60], [70, 140, 220], [70, 140, 220], [240, 170, 40], [120, 190, 90], [120, 190, 90],
                [170, 100, 200], [90, 90, 90], [90, 90, 90], [90, 90, 90], [90, 90, 90], [60, 190, 190], [60, 190, 190]], np.uint8)
vis = m.copy(); vis.visual.face_colors = np.c_[pal[np.clip(label, 0, 13)], np.full(len(label), 255, np.uint8)]
vis.export(out / "labelled.ply")
json.dump(report, open(out / "pieces.json", "w"), indent=1)
for k, v in report.items(): print(k, v)
bad = [k for k, v in report.items() if v["euler"] != 1]
print("RESULT:", "PASS" if not bad else f"CHECK topology {bad} (euler != 1 means not a single disk)")
