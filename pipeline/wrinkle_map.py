"""Wrinkle map for needle-sculpting (design rule: the 3D shape stays smooth; wrinkles go on a wrinkle map plus a
small fabric allowance).

Breed wrinkles are drawn in a front view laid out from the face landmarks (eye centres, nose, chin) and projected
onto the plush surface with rays from the front, so every dog of the breed gets the same set, scaled to its face.
For each wrinkle the depth of the matching crease on the photo-built 3D model (crease_depth.npy: how far the raw
model lies under the smooth plush surface, locally) is reported, as a guide for how deep to sculpt.

Bulldog set (Kennel Club: head skin finely wrinkled; defined stop with a furrow to mid-skull; over-nose wrinkle not
hiding eyes or nose; slight dewlap each side of the neck):
  nose rope (over the nose), forehead furrow (stop to mid-skull), two forehead wrinkles each side, a fold under
  each eye down the cheek beside the jowl, and a short dewlap fold each side of the throat.

Usage: python3 wrinkle_map.py mesh.obj face_points.json nose_z_mm k out.json [--crease crease_depth.npy]
Writes polylines (real mm), lengths at plush size, starting sculpt depths and fabric allowances.
"""
import json, sys
import numpy as np, trimesh
from scipy.spatial import cKDTree

mesh, fpj, nose_z, k, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
M = trimesh.load(mesh, process=False); V = M.vertices; fp = json.load(open(fpj))
ex, ey, ez = fp["eye_R"]; nose_w = fp["nose_width_mm_plush"] / k
crease = np.load(sys.argv[sys.argv.index("--crease") + 1]) if "--crease" in sys.argv else None
mid = np.abs(V[:, 0]) < 6; head = V[:, 1] - V[:, 1].min() < 2.2 * (ey - V[:, 1].min())
chin_z = V[mid & (V[:, 1] - V[:, 1].min() < 1.2 * (ey - V[:, 1].min())) & (V[:, 2] > 0.5 * V[:, 2].max()), 2].min()   # under the jaw
ray = trimesh.ray.ray_triangle.RayMeshIntersector(M)


def front(xz):
    """project front-view points (x, z) onto the plush surface (first hit from the front)."""
    xz = np.asarray(xz, float)
    o = np.c_[xz[:, 0], np.full(len(xz), V[:, 1].min() - 300), xz[:, 1]]
    loc, idx, _ = ray.intersects_location(o, np.tile([0, 1.0, 0], (len(xz), 1)), multiple_hits=False)
    P = np.full((len(xz), 3), np.nan)
    if len(idx): P[idx] = np.asarray(loc).reshape(-1, 3)
    return P[np.isfinite(P[:, 0])]


def side(yz):
    """project side-view points (y, z) onto the +X side of the plush (first hit from the side)."""
    yz = np.asarray(yz, float)
    o = np.c_[np.full(len(yz), V[:, 0].max() + 300), yz[:, 0], yz[:, 1]]
    loc, idx, _ = ray.intersects_location(o, np.tile([-1.0, 0, 0], (len(yz), 1)), multiple_hits=False)
    P = np.full((len(yz), 3), np.nan)
    if len(idx): P[idx] = np.asarray(loc).reshape(-1, 3)
    return P[np.isfinite(P[:, 0])]


def arc(x0, x1, zf, n=24):
    x = np.linspace(x0, x1, n); return np.c_[x, zf(x)]


nz_top = nose_z + 0.36 * nose_w
W = {"nose rope": [arc(-0.85 * ex, 0.85 * ex, lambda x: nz_top + 5 + 9 * (1 - (x / (0.85 * ex)) ** 2))],
     "forehead furrow": [np.c_[np.zeros(16), np.linspace(ez + 4, ez + 1.0 * ex, 16)]]}
W["forehead wrinkles"] = [arc(s * 0.25 * ex, s * 0.9 * ex, lambda x, f=f: ez + f * ex + 0.12 * ex * np.abs(x) / (0.9 * ex))
                          for f in (0.45, 0.75) for s in (1, -1)]
W["under-eye folds"] = [np.c_[s * np.linspace(0.95 * ex, 1.25 * ex, 16), np.linspace(ez - 0.2 * ex, ez - 0.75 * ex, 16)] for s in (1, -1)]
y_jb = V[:, 1].min() + 1.25 * (ey - V[:, 1].min())               # just behind the jowls
DEW = np.c_[np.linspace(y_jb, y_jb + 40, 14), np.linspace(chin_z - 5, chin_z - 35, 14)]
START = {"nose rope": (4, "across the muzzle tops and head gusset"), "forehead furrow": (3, "across the head gusset"),
         "forehead wrinkles": (2, "across the side heads"), "under-eye folds": (2, "across the side heads"),
         "dewlap folds": (2, "across the chest halves")}
tree = cKDTree(V)
res = {"k": k, "note": "starting values for the first sample; depths and allowances in plush mm", "wrinkles": []}
for name, lines in list(W.items()) + [("dewlap folds", [DEW, DEW])]:
    for i, l in enumerate(lines):
        P = side(l) if name == "dewlap folds" else front(l)
        if name == "dewlap folds" and i == 1: P = P * [-1, 1, 1]          # left side: mirror
        if len(P) < 3: continue
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        cd = float(np.median(crease[tree.query(P)[1]])) if crease is not None else None
        d, where = START[name]
        res["wrinkles"].append({"name": name, "part": i, "points_mm": P.round(1).tolist(), "length_plush_mm": round(L * k, 1),
                                "model_crease_depth_real_mm": None if cd is None else round(cd, 1),
                                "sculpt_depth_plush_mm": d, "fabric_allowance_plush_mm": round(1.5 * d, 1), "allowance_where": where})
json.dump(res, open(out, "w"), indent=1)
for w in res["wrinkles"]: print(w["name"], w["part"], "length", w["length_plush_mm"], "mm, model crease", w["model_crease_depth_real_mm"], "mm")
ok = len({w["name"] for w in res["wrinkles"]}) == len(W) + 1
print("RESULT:", "PASS" if ok else "FAIL (a wrinkle did not land on the surface)")
