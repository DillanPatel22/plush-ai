"""Markings rule check: how far the nearest seams sit from the dog's crisp white marking edges (design rule: where a
dog has a crisp marking edge - blaze, bib, socks - the nearest seam may be moved onto it so the colour change hides in
the seam; each move needs the craftsman's approval).

White = coat colour (from colour_map.py, lightly smoothed) lighter than --white (mean RGB). Two seams are measured,
slice by slice: the head gusset edge against the blaze (across the forehead, slices along Y), and the chest-half edge
against the bib (slices along Z, how far behind or in front of the seam the coloured coat starts).

Usage: python3 markings_check.py seam_mesh.obj colours.npy face_labels.npy k out.json [--white 165]
Acceptance: both edges found; offsets reported in real and plush mm.
"""
import json, sys
import numpy as np, trimesh, scipy.sparse as sp

mesh, colp, labp, k, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
WH = float(sys.argv[sys.argv.index("--white") + 1]) if "--white" in sys.argv else 165.0
m = trimesh.load(mesh, process=False); col = np.load(colp).astype(float); lab = np.load(labp); V = m.vertices; y0 = V[:, 1].min()
E = m.edges_unique; A = sp.coo_matrix((np.ones(len(E) * 2), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V),) * 2).tocsr()
D = np.asarray(A.sum(1)).ravel(); L = col.mean(1)
for _ in range(8): L = 0.5 * L + 0.5 * (A @ L) / D
white = L > WH
fa, fe = m.face_adjacency, m.face_adjacency_edges


def seam_pts(a, b):
    s = ((lab[fa[:, 0]] == a) & (lab[fa[:, 1]] == b)) | ((lab[fa[:, 0]] == b) & (lab[fa[:, 1]] == a))
    return V[fe[s]].mean(1)


res = {"white_threshold": WH, "white_share": round(float(white.mean()), 3), "head_gusset_vs_blaze": [], "chest_vs_bib": []}
g = seam_pts(1, 2)
for yy in range(60, 181, 20):                       # across the forehead: x of the gusset seam vs x where the blaze ends
    s = np.abs(g[:, 1] - y0 - yy) < 4
    if not s.any(): continue
    gx = float(g[s, 0].mean())
    idx = np.where((np.abs(V[:, 1] - y0 - yy) < 3) & (V[:, 0] >= 0))[0]
    idx = idx[V[idx, 2] > np.percentile(V[idx, 2], 70)]; idx = idx[np.argsort(V[idx, 0])]
    nw = idx[~white[idx]]; bx = float(V[nw, 0].min()) if len(nw) else float("nan")
    res["head_gusset_vs_blaze"].append({"y_from_nose_mm": yy, "seam_x_mm": round(gx, 1), "blaze_edge_x_mm": round(bx, 1),
                                        "offset_real_mm": round(bx - gx, 1), "offset_plush_mm": round((bx - gx) * k, 1)})
c = seam_pts(9, 5)
for zz in range(150, 271, 30):                      # up the chest: y of the chest seam vs y where the coloured coat begins
    s = (np.abs(c[:, 2] - zz) < 5) & (c[:, 0] > 0)
    if not s.any(): continue
    cy = float(c[s, 1].mean() - y0); cx = float(c[s, 0].mean())
    idx = np.where((np.abs(V[:, 2] - zz) < 3) & (V[:, 0] > 0) & (np.abs(V[:, 0] - cx) < 35) & (V[:, 1] - y0 < 330))[0]
    idx = idx[np.argsort(V[idx, 1])]; nw = idx[~white[idx] & (V[idx, 1] - y0 > cy - 60)]
    by = float(V[nw, 1].min() - y0) if len(nw) else float("nan")
    res["chest_vs_bib"].append({"z_mm": zz, "seam_y_mm": round(cy, 1), "colour_starts_y_mm": round(by, 1),
                                "offset_real_mm": round(by - cy, 1), "offset_plush_mm": round((by - cy) * k, 1)})
json.dump(res, open(out, "w"), indent=1)
for key in ("head_gusset_vs_blaze", "chest_vs_bib"):
    print(key, [(r.get("y_from_nose_mm", r.get("z_mm")), r["offset_plush_mm"]) for r in res[key]])
ok = all(np.isfinite(r["offset_real_mm"]) for key in ("head_gusset_vs_blaze", "chest_vs_bib") for r in res[key]) and res["chest_vs_bib"]
print("RESULT:", "PASS" if ok else "CHECK (an edge was not found)")
