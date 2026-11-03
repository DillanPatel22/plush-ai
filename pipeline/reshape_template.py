"""Reshape the dog base template onto a breed's plush solid. Vertex count and order never change, so every
seam, notch and piece number of a frozen layout carries over to the breed.

Method (trimesh MIT, scipy BSD, numpy BSD):
 1. Landmarks found the same way on both meshes (project frame, symmetric): nose tip, eyes (from each breed's
    face_points file), stop, muzzle top, chin, cheeks, skull top, chest front, withers, topline, loin, belly,
    point of buttock, tail root / middle / tip, and for every leg the sole centre, toe front and two
    cross-section centres up the leg. Left/right landmarks are exact mirrors.
 2. A smoothed thin-plate-spline warp driven by the landmark pairs moves the template roughly onto the breed
    (a Labrador's long muzzle, legs and tail onto a bulldog's short ones).
 3. Non-rigid ICP (Amberg et al. 2007, trimesh.registration.nricp_amberg) from stiff to soft, landmarks held,
    lands the template on the breed's surface; then a few rounds of tangential relaxation (each vertex moved
    towards the centre of its neighbours and projected back onto the breed's surface) even out the faces.
 4. Exact symmetry (the template's own mirror map), soles back on Z = 0.

Usage: python3 reshape_template.py base.obj base_face_points.json target.ply target_face_points.json out.obj
          [--target_tail tail_stub.json]
Acceptance (printed): vertex count/order unchanged; template -> breed surface distance mean <= 1.5 mm and
p95 <= 4 mm; breed -> template p95 <= 5 mm; folded-over edges (dihedral > 150 deg) no more than the base
template's own + 5; slivers (< 5 deg) <= 1.5 %; symmetry 0.0 mm.
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
import align_measure as am

base_p, bfp_p, tgt_p, tfp_p, out_p = sys.argv[1:6]
B = trimesh.load(base_p, process=False); Tm = trimesh.load(tgt_p, process=False)
V0 = np.asarray(B.vertices, float); F = np.asarray(B.faces)
mirror = cKDTree(V0).query(V0 * [-1, 1, 1])[1]
assert np.abs(V0[mirror] * [-1, 1, 1] - V0).max() < 1e-4, "base template is not exactly symmetric"


def symm(V):
    V = 0.5 * (V + V[mirror] * [-1, 1, 1]); V[np.abs(V0[:, 0]) < 1e-9, 0] = 0.0; return V


def landmarks(m, eye, tail=None):
    """Named landmark points (right side only for pairs, +X) on a symmetric mesh in the project frame."""
    P, _ = trimesh.sample.sample_surface(m, 250000, seed=3)
    y0 = m.vertices[:, 1].min(); P = P - [0, y0, 0]
    x, y, z = P.T; ax = np.abs(x)
    r, _ = am.measure(m, 1.0); lm = r["landmarks_px"]; Hw = r["withers_height_mm"]; E = r["elbow_height_mm"]
    H_img = int(m.vertices[:, 2].max()) + 20
    to_y = lambda c: c - 10.0; to_z = lambda rr: H_img - 10.0 - rr
    mid = ax < 0.012 * Hw
    L = {}
    hz = mid & (z > 0.4 * Hw); L["nose"] = P[hz][np.argmin(y[hz])]
    e = np.array(eye) - [0, y0, 0]; L["eye"] = e
    ny, ey = L["nose"][1], e[1]; dm = ey - ny
    def top_at(yy, sel=mid, w=3.0):
        s = sel & (np.abs(y - yy) < w); return P[s][np.argmax(z[s])]
    def bot_at(yy, sel=mid, w=3.0):
        s = sel & (np.abs(y - yy) < w) & (z > 0.2 * Hw); return P[s][np.argmin(z[s])]
    L["stop"] = top_at(ey); L["muzzle_top"] = top_at(ny + 0.5 * dm)
    s = mid & (y < ey) & (y > ny) & (z > 0.3 * Hw); L["chin"] = P[s][np.argmin(z[s])]
    s = mid & (y > ny) & (y < ey + 1.2 * dm); L["skull_top"] = P[s][np.argmax(z[s])]
    zc = e[2] - 0.35 * (e[2] - L["chin"][2])
    s = (x > 0) & (np.abs(y - (ey + 0.1 * dm)) < 4) & (np.abs(z - zc) < 4); L["cheek"] = P[s][np.argmax(x[s])]
    s = (x > 0) & (np.abs(y - (ey + 0.6 * dm)) < 4) & (np.abs(z - e[2]) < 4); L["skull_side"] = P[s][np.argmax(x[s])]
    wy = to_y(lm["withers"][0]); L["withers"] = top_at(wy)
    s = mid & (z > E) & (z < E + 0.45 * (Hw - E)); L["chest_front"] = P[s][np.argmin(y[s])]
    s = mid & (z > 0.40 * Hw) & (z < 0.58 * Hw); L["buttock"] = P[s][np.argmax(y[s])]
    fl, rl = r["front_leg_y_mm"], r["rear_leg_y_mm"]
    loin_y = 0.5 * (rl[0] + rl[1]); L["loin"] = top_at(loin_y)
    L["topline"] = top_at(0.5 * (wy + loin_y))
    L["belly"] = bot_at(0.5 * (fl[1] + rl[0]))
    L["brisket"] = bot_at(fl[1] + 0.1 * (rl[0] - fl[1]))
    # tail: either a stub axis given by tail_stub.py (short-tailed breeds) or found on the mesh (long tails)
    if tail is not None:
        rc, tc = (np.array(tail[k], float) - [0, y0, 0] for k in ("root_centre", "tip_centre"))
        d_ = np.array(tail["axis"], float); Lt = tail["length_mm"]; r0, r1 = tail["r0_mm"], tail["r1_mm"]
        q = P - rc; t = np.clip(q @ d_ / Lt, 0, 1); rad = np.linalg.norm(q - t[:, None] * Lt * d_, axis=1)
        on = mid & (np.abs(rad - (r0 + (r1 - r0) * t)) < 4) & (q @ d_ < Lt)        # exposed stub surface, midline
        upv = np.array([0.0, -d_[2], d_[1]]); upv = upv if upv[2] > 0 else -upv
        above = (q @ upv) > 0
        i_top = np.where(on & above)[0][np.argmin(t[on & above])]; i_und = np.where(on & ~above)[0][np.argmin(t[on & ~above])]
        L["tail_root_top"] = P[i_top]; L["tail_root_under"] = P[i_und]
        tj = 0.5 * (t[i_top] + t[i_und]); c0 = rc + d_ * Lt * tj
        L["tail_c0"] = c0; L["tail_c1"] = 0.5 * (c0 + tc); L["tail_c2"] = tc; L["tail_end"] = tc + d_ * r1
    else:
        bt = L["buttock"]
        s = mid & (y > bt[1] + 10) & (z > bt[2])
        yb = np.arange(bt[1] + 10, y[s].max(), 4.0)
        zmin = np.array([z[s & (np.abs(y - v) < 2)].min() if (s & (np.abs(y - v) < 2)).any() else np.nan for v in yb])
        root_y = yb[np.argmax(zmin > bt[2] + 60)] + 4
        tail_pts = (y > root_y) & (z > bt[2]) & (ax < 0.15 * Hw)
        cen = lambda yy: P[tail_pts & (np.abs(y - yy) < 4)].mean(0)
        tip_y = y[tail_pts].max()
        L["tail_c0"] = cen(root_y + 4); L["tail_c1"] = cen(0.5 * (root_y + tip_y)); L["tail_c2"] = cen(tip_y - 12)
        L["tail_end"] = P[tail_pts][np.argmax(y[tail_pts])]
        s2 = mid & tail_pts & (np.abs(y - root_y - 4) < 3)
        L["tail_root_top"] = P[s2][np.argmax(z[s2])]; L["tail_root_under"] = P[s2][np.argmin(z[s2])]
    # legs: sole centre, toe front, and cross-section centres at 1/3 and 2/3 of the elbow height
    sole = (z < 1.0) & (x > 0)
    for leg, (ya, yb) in (("front", fl), ("hind", rl)):
        cy = 0.5 * (ya + yb); half = 0.5 * (rl[0] - fl[1])
        s = sole & (np.abs(y - cy) < half); c = P[s].mean(0); L[f"{leg}_sole"] = c
        L[f"{leg}_toe"] = P[s][np.argmin(y[s])]
        for f in (0.33, 0.66):
            zz = f * E
            s = (x > 0) & (np.abs(z - zz) < 2.5) & (np.abs(y - cy) < half)
            # keep the slab cluster under the sole (the leg), not the brisket or belly
            pts = P[s][:, :2]; lab_, n_ = ndimage.label(np.histogram2d(pts[:, 0], pts[:, 1], bins=[np.arange(0, 200, 3), np.arange(-50, 1200, 3)])[0] > 0)
            ix = np.clip(((pts[:, 0]) // 3).astype(int), 0, lab_.shape[0] - 1); iy = np.clip(((pts[:, 1] + 50) // 3).astype(int), 0, lab_.shape[1] - 1)
            labs = lab_[ix, iy]; d = np.array([np.linalg.norm(pts[labs == k].mean(0) - c[:2]) if (labs == k).any() else 1e9 for k in range(1, n_ + 1)])
            k = 1 + int(np.argmin(d)); L[f"{leg}_leg_{int(f * 100)}"] = np.r_[pts[labs == k].mean(0), zz]
    return {k: np.asarray(v, float) + [0, y0, 0] for k, v in L.items()}, r


PAIRED = ("eye", "cheek", "skull_side", "front_", "hind_")
INTERIOR = ("tail_c", "front_leg_", "hind_leg_")                         # inside the body: warp only, not ICP


def pairs(L):
    names = sorted(L); pts = []; nm = []
    for k in names:
        p = L[k]
        if k.startswith(PAIRED):
            pts.append([abs(p[0]), p[1], p[2]]); nm.append(k + "_R")
            pts.append([-abs(p[0]), p[1], p[2]]); nm.append(k + "_L")
        else:
            pts.append([0.0, p[1], p[2]]); nm.append(k)
    return nm, np.array(pts)


bfp = json.load(open(bfp_p)); tfp = json.load(open(tfp_p))
tail_t = json.load(open(sys.argv[sys.argv.index("--target_tail") + 1])) if "--target_tail" in sys.argv else None
LB, rB = landmarks(B, bfp["eye_R"]); LT, rT = landmarks(Tm, tfp["eye_R"], tail_t)
nB, PB = pairs(LB); nT, PT = pairs(LT); assert nB == nT
print("landmarks:", len(nB))
for n_, a_, b_ in zip(nB, PB, PT): print(f"  {n_:14s} base {np.round(a_, 0)} -> breed {np.round(b_, 0)}")

# 2. smoothed thin-plate-spline warp, landmarks snapped to template vertices
src_idx = cKDTree(V0).query(PB)[1]
S0 = V0[src_idx]
tps = RBFInterpolator(S0, PT - S0, kernel="thin_plate_spline", smoothing=1e-3 * len(S0))
V1 = symm(V0 + tps(V0))
print("TPS warp: mean move", round(float(np.linalg.norm(V1 - V0, axis=1).mean()), 1), "mm")

# 3. non-rigid ICP in normalised units (1 unit = 500 mm), then tangential relaxation on the breed surface
sc = 500.0
src = trimesh.Trimesh(V1 / sc, F, process=False); tgt = trimesh.Trimesh(Tm.vertices / sc, Tm.faces, process=False)
steps = [[2.0, 10, 0.5, 8], [0.6, 5, 0.5, 8], [0.2, 2, 0.5, 8], [0.06, 1, 0.3, 8], [0.02, 0.5, 0.2, 8]]
surf = np.array([not n_.startswith(INTERIOR) for n_ in nB])             # axis centres steer the warp only
V2 = trimesh.registration.nricp_amberg(src, tgt, source_landmarks=src_idx[surf], target_positions=PT[surf] / sc,
                                                   steps=steps, distance_threshold=0.08, gamma=1) * sc
V2 = symm(V2)
Tm_ = trimesh.Trimesh(Tm.vertices, Tm.faces, process=False)
def folded_pct(Vx):
    Rx = trimesh.Trimesh(Vx, F, process=False); _, dd, tr_ = trimesh.proximity.closest_point(Tm_, Rx.triangles_center)
    return round(100 * float((np.einsum("ij,ij->i", Rx.face_normals, Tm_.face_normals[tr_]) < 0).mean()), 3), round(float(dd.mean()), 2)
print("folded % / mean dist after TPS:", folded_pct(V1), "after NRICP:", folded_pct(V2))
nbr = trimesh.Trimesh(V0, F, process=False).vertex_neighbors
V = V2.copy()
for it in range(12):
    C = np.array([V[n].mean(0) for n in nbr])
    V = 0.6 * V + 0.4 * C
    V, _, _ = trimesh.proximity.closest_point(Tm_, V)
    V[src_idx[surf]] = PT[surf]                       # surface landmarks stay put
    V = symm(V)
    print("  relax", it, folded_pct(V))
# untangle: where faces still fold over (concave creases the projection snapped across), relax the fold and
# its two-ring neighbourhood without projecting, until nothing folds; plush stuffing rounds those creases anyway
Fs = trimesh.Trimesh(V0, F, process=False)
vf = Fs.vertex_faces
for it in range(40):
    Rx = trimesh.Trimesh(V, F, process=False); _, _, tr_ = trimesh.proximity.closest_point(Tm_, Rx.triangles_center)
    bad = np.where(np.einsum("ij,ij->i", Rx.face_normals, Tm_.face_normals[tr_]) < 0)[0]
    if len(bad) == 0: break
    zone = np.unique(F[bad])
    for _ in range(2): zone = np.unique(np.concatenate([zone] + [nbr[v_] for v_ in zone]))
    C = np.array([V[nbr[v_]].mean(0) for v_ in zone])
    V[zone] = 0.5 * V[zone] + 0.5 * C
    V[src_idx[surf]] = PT[surf]; V = symm(V)
print("untangle:", it, "rounds,", len(bad), "faces still folded")
# 4. soles back on the ground
sole0 = V0[:, 2] < 0.5
V[sole0, 2] = 0.0; V[V[:, 2] < 0, 2] = 0.0
V = symm(V)
out = trimesh.load(base_p, process=False); out.vertices = V; out.export(out_p)

# ---- acceptance ----
R = trimesh.Trimesh(V, F, process=False)
d1 = np.abs(trimesh.proximity.signed_distance(Tm_, V[np.random.default_rng(0).choice(len(V), 4000, replace=False)]))
Pt, _ = trimesh.sample.sample_surface(Tm_, 4000, seed=5); _, d2, _ = trimesh.proximity.closest_point(R, Pt)
_, _, tri = trimesh.proximity.closest_point(Tm_, R.triangles_center)
facing_away = float((np.einsum("ij,ij->i", R.face_normals, Tm_.face_normals[tri]) < 0).mean())   # info: crease ambiguity
fold_edges = int((np.degrees(R.face_adjacency_angles) > 150).sum())                               # faces folded over
fold_base = int((np.degrees(trimesh.Trimesh(V0, F, process=False).face_adjacency_angles) > 150).sum())
sym = float(np.abs(V[mirror] * [-1, 1, 1] - V).max())
ang = R.face_angles.min(1) * 180 / np.pi
rep = {"vertices": int(len(V)), "same_count_and_order": bool(len(V) == len(V0) and out.faces.shape == B.faces.shape),
       "to_breed_mean_mm": round(float(d1.mean()), 2), "to_breed_p95_mm": round(float(np.percentile(d1, 95)), 2),
       "breed_to_template_p95_mm": round(float(np.percentile(d2, 95)), 2), "breed_to_template_max_mm": round(float(d2.max()), 2),
       "fold_edges_gt150deg": fold_edges, "fold_edges_base": fold_base, "facing_away_from_breed_pct": round(100 * facing_away, 3),
       "symmetry_max_mm": round(sym, 4),
       "min_angle_p1_deg": round(float(np.percentile(ang, 1)), 1), "slivers_lt5deg_pct": round(float((ang < 5).mean() * 100), 2),
       "landmarks": {n_: [round(float(c), 1) for c in p] for n_, p in zip(nT, PT)},
       "breed_measures": {k: rT[k] for k in ("withers_height_mm", "elbow_height_mm", "body_length_mm", "length_to_height", "elbow_over_height", "nose_to_chest_front_mm")}}
json.dump(rep, open(out_p.rsplit(".", 1)[0] + "_reshape.json", "w"), indent=1, default=float)
print({k: v for k, v in rep.items() if k not in ("landmarks", "breed_measures")})
ok = (rep["same_count_and_order"] and rep["to_breed_mean_mm"] <= 1.5 and rep["to_breed_p95_mm"] <= 4.0
      and rep["breed_to_template_p95_mm"] <= 5.0 and rep["fold_edges_gt150deg"] <= rep["fold_edges_base"] + 5
      and rep["slivers_lt5deg_pct"] <= 1.5 and rep["symmetry_max_mm"] < 1e-3)
print("RESULT:", "PASS" if ok else "FAIL")
