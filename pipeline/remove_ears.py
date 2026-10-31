"""Remove the ear flaps from an aligned raw dog mesh by a smooth (bilaplacian) fill, giving the bare skull.

Usage: python3 remove_ears.py aligned_merged.ply ear_mask2d.npz out.ply
The ear mask is a side-view (Y,Z) mask of the ear (from texture saturation), shared by both sides.
"""
import json, sys
import numpy as np, trimesh
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

src, mpath, dst = sys.argv[1:4]
m = trimesh.load(src, process=False)
V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
d = np.load(mpath); mask = d['mask']; cell = float(d['cell']); outer = d['outer']; y0 = float(d['y0']); z0 = float(d['z0'])
ny, nz = mask.shape
E = m.edges_unique
A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))); A = (A + A.T).tocsr(); A.data[:] = 1
# keep only the main connected surface
nc, lab = connected_components(A, directed=False)
main = lab == np.argmax(np.bincount(lab))
iy = ((V[:, 1] - y0) / cell).astype(int); iz = ((V[:, 2] - z0) / cell).astype(int)
ok = (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz) & (np.abs(V[:, 0]) > 30)
inm = np.zeros(len(V), bool); inm[ok] = mask[iy[ok], iz[ok]]
lay = np.zeros(len(V), bool); lay[ok] = np.abs(V[ok, 0]) >= outer[iy[ok], iz[ok]] - 25
reg = inm & lay & main
for _ in range(3):
    reg = reg | ((A @ reg.astype(float) > 0) & main)
L = sp.diags(np.asarray(A.sum(1)).ravel()) - A; Q = (L @ L).tocsr()
free = np.where(reg)[0]; fixed = np.where(~reg)[0]
X = spla.spsolve(Q[free][:, free].tocsc(), -(Q[free][:, fixed] @ V[fixed]))
W = V.copy(); W[free] = X
mv = np.linalg.norm(W - V, axis=1)
keepF = main[F].all(1)
out = trimesh.Trimesh(W, F[keepF], process=False); out.remove_unreferenced_vertices(); out.export(dst)
rep = {"region_vertices": int(len(free)), "max_move_mm": round(float(mv.max()), 1), "mean_move_mm": round(float(mv[free].mean()), 1),
       "p95_move_mm": round(float(np.percentile(mv[free], 95)), 1)}
json.dump(rep, open(dst.rsplit('.', 1)[0] + '_report.json', 'w'), indent=1)
print(rep)
ok_ = rep["max_move_mm"] < 130  # ears on this model flare up to ~55 mm out from the skull
print("RESULT:", "PASS" if ok_ else "FAIL (moves too large: region probably wrong)")
