"""Seam layout v2 (PROPOSAL for craftsman approval, not production): label every face with its pattern piece.

Changes from v1, all to make the fabric's pile run the way a Labrador's coat does and to shape the face:
  - legs split from the body below the elbow (front) and the stifle (hind): outer and inner leg pieces, so the
    pile can run down the legs (in v1 the legs were part of the body side and underbody pieces)
  - chest panel ("bib") over the chest and the fronts of the shoulders down to the elbows, separate from the
    belly (pile runs down on the bib, back on the belly); belly in two halves
    with a centre seam underneath (one piece saddles between the hind legs: 22 % stretch there)
  - muzzle sides split from the side heads along a line from the stop, in front of the eye, to the mouth corner
    (sharper stop, fuller cheeks, a natural place to set the eyes)
Unchanged: head gusset, chin gusset, body sides with centre-back seam, two-piece tail, soles, two-layer ears.

Usage: python3 seams_standing_v2.py template_medium.obj out_dir
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
import scipy.sparse as sp
from scipy.spatial import cKDTree

src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
T = trimesh.load(src, process=False)
m = trimesh.Trimesh(T.vertices, T.faces, process=False)        # triangulate quads
while len(m.faces) < 60000:                                     # ~4 mm triangles at real-dog scale
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
# ---- v2: legs below the elbow / stifle are their own pieces (outer + inner), chest panel, belly ----
FRONT_CUT, HIND_CUT = 235.0, 260.0                 # elbow and stifle heights on this template (real mm)
cut_z = np.where(front_leg, FRONT_CUT, np.where(rear_leg, HIND_CUT, -1.0))
lower_leg = body & (front_leg | rear_leg) & (z < cut_z) & ~sole
leg_in = lower_leg & (N[:, 0] * np.sign(x) < -0.05)
leg_out = lower_leg & ~leg_in
under = ((inner_leg & ~lower_leg) | belly | chest) & ~sole
label[body & ~sole & (x >= 0)] = 5; label[body & ~sole & (x < 0)] = 6
label[under & (x >= 0)] = 10; label[under & (x < 0)] = 11          # belly halves: centre seam underneath
# chest panel as a bib: every forward-facing surface of the chest and the fronts of the shoulders down to the elbows
bib = body & (y < 360) & (N[:, 1] < -0.3) & (z < 520) & (z > FRONT_CUT - 5) & ~sole & ~lower_leg
label[bib | (chest & ~sole & ~lower_leg)] = 9
R_ = x >= 0
label[leg_out & front_leg & R_] = 14; label[leg_out & front_leg & ~R_] = 15
label[leg_in & front_leg & R_] = 16; label[leg_in & front_leg & ~R_] = 17
label[leg_out & rear_leg & R_] = 18; label[leg_out & rear_leg & ~R_] = 19
label[leg_in & rear_leg & R_] = 24; label[leg_in & rear_leg & ~R_] = 25
# tail: separate left/right halves from the root
tail = body & (y > 895) & (z > 380) & ~sole
label[tail & (x >= 0)] = 12; label[tail & (x < 0)] = 13
# soles: which paw
paw = np.where(front_leg, 0, 2) + np.where(x >= 0, 0, 1)                    # FR, FL, RR, RL
label[sole] = 30 + paw[sole]
# muzzle sides: front of the side head, seam from the stop down in front of the eye to the mouth corner
y_line = np.interp(z, [560, 600, 650, 700], [102, 95, 86, 78])
muzzle = (label == 2) | (label == 3)
muzzle &= y < y_line
label[muzzle & (x >= 0)] = 21; label[muzzle & (x < 0)] = 22

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
# v2 pieces meet at more three-way junctions; a wider (two-ring) vote removes the zigzags there
A2 = (Aj @ Aj).tocsr(); A2.setdiag(0); A2.eliminate_zeros()
for _ in range(8):
    new = label.copy()
    for i in range(len(C)):
        nb = label[A2.indices[A2.indptr[i]:A2.indptr[i + 1]]]
        vals, cnt = np.unique(nb, return_counts=True)
        if vals[np.argmax(cnt)] != label[i] and cnt.max() > 0.6 * len(nb): new[i] = vals[np.argmax(cnt)]
    label = new
swap = {2: 3, 3: 2, 21: 22, 22: 21, 5: 6, 6: 5, 10: 11, 11: 10, 14: 15, 15: 14, 16: 17, 17: 16, 18: 19, 19: 18, 24: 25, 25: 24,
        30: 31, 31: 30, 32: 33, 33: 32, 12: 13, 13: 12}
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
# sole rims sit on a sharp edge: projecting smoothed points there flips triangles, so they stay put
movable = [v_ for v_, nb_ in nb_seam.items() if len(nb_) == 2 and nlab[v_] == 2 and m.vertices[v_, 2] > 4.0]
print("three-way junctions:", sum(1 for v_ in nb_seam if nlab[v_] >= 3))
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

PIECES = {1: ("P01", "head gusset", None), 2: ("P02", "side head R", None), 3: ("P03", "side head L", "P02"),
          4: ("P04", "chin gusset", None), 21: ("P05", "muzzle side R", None), 22: ("P06", "muzzle side L", "P05"),
          5: ("P07", "body side R", None), 6: ("P08", "body side L", "P07"), 9: ("P09", "chest panel", None),
          10: ("P10", "belly R", None), 11: ("P11", "belly L", "P10"), 14: ("P12", "front leg outer R", None), 15: ("P13", "front leg outer L", "P12"),
          16: ("P14", "front leg inner R", None), 17: ("P15", "front leg inner L", "P14"),
          18: ("P16", "hind leg outer R", None), 19: ("P17", "hind leg outer L", "P16"),
          24: ("P18", "hind leg inner R", None), 25: ("P19", "hind leg inner L", "P18"),
          30: ("P20", "sole front R", None), 31: ("P21", "sole front L", "P20"), 32: ("P22", "sole hind R", None),
          33: ("P23", "sole hind L", "P22"), 12: ("P24", "tail R", None), 13: ("P25", "tail L", "P24")}
names = {L: f"{pid} {nm}" for L, (pid, nm, _) in PIECES.items()}
report = {}
for L, (pid, nm, mirror_of) in PIECES.items():
    fi = np.where(label == L)[0]
    if not len(fi): report[pid] = {"name": nm, "faces": 0, "euler": None}; continue
    piece = m.submesh([fi], append=True)
    piece.export(out / f"{pid}.obj")
    report[pid] = {"name": nm, "label": int(L), "faces": int(len(fi)), "area_cm2_real": round(piece.area / 100, 1),
                   "euler": int(piece.euler_number), "mirror_of": mirror_of}
np.save(out / "face_labels.npy", label)
m.export(out / "subdivided.obj")
# coloured preview mesh
COL = {1: (230, 80, 60), 2: (70, 140, 220), 3: (70, 140, 220), 4: (240, 170, 40), 21: (250, 120, 170), 22: (250, 120, 170),
       5: (120, 190, 90), 6: (120, 190, 90), 9: (250, 220, 90), 10: (170, 100, 200), 11: (170, 100, 200), 14: (40, 120, 90), 15: (40, 120, 90),
       16: (150, 220, 200), 17: (150, 220, 200), 18: (200, 120, 60), 19: (200, 120, 60), 24: (240, 190, 150), 25: (240, 190, 150),
       30: (90, 90, 90), 31: (90, 90, 90), 32: (90, 90, 90), 33: (90, 90, 90), 12: (60, 190, 190), 13: (60, 190, 190)}
fc = np.array([COL.get(int(l), (0, 0, 0)) for l in label], np.uint8)
vis = m.copy(); vis.visual.face_colors = np.c_[fc, np.full(len(label), 255, np.uint8)]
vis.export(out / "labelled.ply")
json.dump({"layout": "standing v2 (proposal)", "pieces": report}, open(out / "pieces.json", "w"), indent=1)
for k, v in report.items(): print(k, v)
bad = [k for k, v in report.items() if v["euler"] != 1]
print("RESULT:", "PASS" if not bad else f"CHECK topology {bad} (euler != 1 means not a single disk)")
