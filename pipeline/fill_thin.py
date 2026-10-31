"""Fill out legs and tail so every tube is at least min_w wide at the chosen plush size.

Method: voxelise the template, find its medial skeleton (with the local radius of every skeleton point),
and push each surface vertex whose nearest skeleton point is too thin out to the minimum radius.
The displacement is smoothed over the surface so tubes blend into the body, and the +X half is
mirrored for exact symmetry. Vertex count and order never change (the template stays the template).

Usage: python3 fill_thin.py template.obj measures.json plush_nose_to_tail_mm min_w_mm out.obj
"""
import json, sys
import numpy as np, trimesh, igl
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

src, meas, size_mm, min_w, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
T = trimesh.load(src, process=False); V0 = np.asarray(T.vertices, float); F = np.asarray(T.faces, np.int64)
if F.shape[1] != 3:
    T = trimesh.Trimesh(V0, F, process=False); F = np.asarray(T.faces, np.int64)
ms = json.load(open(meas))
k = size_mm / (ms["nose_to_chest_front_mm"] + ms["body_length_mm"])
r_min = (min_w / k) / 2 * 1.08                      # real-scale minimum radius, 8 % margin for sewing loss
p = 3.0
lo = V0.min(0) - 10; hi = V0.max(0) + 10
xs, ys, zs = (np.arange(lo[i], hi[i], p) for i in range(3))
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
occ = (igl.fast_winding_number(V0, F, np.c_[X.ravel(), Y.ravel(), Z.ravel()]) > 0.5).reshape(X.shape)
dist = ndimage.distance_transform_edt(occ) * p
skel = skeletonize(occ) > 0
si = np.argwhere(skel)
S = np.c_[xs[si[:, 0]], ys[si[:, 1]], zs[si[:, 2]]]; R = dist[skel]
# vertex -> nearest skeleton point
d, j = cKDTree(S).query(V0)
thin = R[j] < r_min
vec = V0 - S[j]; nrm = np.linalg.norm(vec, axis=1, keepdims=True); dirv = vec / np.maximum(nrm, 1e-9)
target_r = np.maximum(nrm[:, 0], r_min)
disp = np.where(thin[:, None], dirv * (target_r - nrm[:, 0])[:, None], 0.0)
# smooth the displacement over the surface (uniform Laplacian), so transitions are gentle
E = T.edges_unique
A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V0), len(V0))); A = (A + A.T).tocsr(); A.data[:] = 1
Dinv = sp.diags(1.0 / np.asarray(A.sum(1)).ravel())
for _ in range(6):
    disp = 0.5 * disp + 0.5 * (Dinv @ A @ disp)
V = V0 + disp
# exact symmetry and soles stay on the ground
mirror = cKDTree(V0).query(V0 * [-1, 1, 1])[1]
neg = V0[:, 0] < 0
V[neg] = V[mirror[neg]] * [-1, 1, 1]
V[np.abs(V0[:, 0]) < 1e-6, 0] = 0.0
V[V0[:, 2] < 0.5, 2] = 0.0
V[:, 2] = np.maximum(V[:, 2], 0.0)
trimesh.Trimesh(V, np.asarray(trimesh.load(src, process=False).faces) if False else F, process=False)
out = trimesh.load(src, process=False); out.vertices = V; out.export(dst)
rep = {"scale": round(k, 4), "min_radius_real_mm": round(r_min, 1), "vertices_moved": int((np.linalg.norm(disp, axis=1) > 0.5).sum()),
       "max_move_mm_real": round(float(np.linalg.norm(disp, axis=1).max()), 1), "skeleton_points": int(len(S))}
json.dump(rep, open(dst.rsplit(".", 1)[0] + "_fill_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
