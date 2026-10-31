"""Preview only (not part of the frozen template): hang the ear module on the head so review renders show the
dog with ears. The flap is the traced ear outline below the attachment line, draped 3 mm off the head surface.

Usage: python3 attach_ears_preview.py template.obj ears_dir out.obj
"""
import sys
from pathlib import Path
import numpy as np, trimesh
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp

T = trimesh.load(sys.argv[1], process=False); E = Path(sys.argv[2]); out = sys.argv[3]
outline = np.load(E / "ear_outline_mm.npy")                       # (y, z) side view, template frame
att_on = np.load(E / "ear_attach_on_template.npy")                # attachment line on the +X side of the head
GAP = 3.0; STEP = 3.0
ray = trimesh.ray.ray_triangle.RayMeshIntersector(T)

# flap = outline below the attachment line (the part above it is the fold over the ear base)
ya, za = att_on[:, 1], att_on[:, 2]
below = Polygon(np.r_[[[ya[0] - 200, za[0]]], np.c_[ya, za], [[ya[-1] + 200, za[-1]], [ya[-1] + 200, 0], [ya[0] - 200, 0]]])
flap = Polygon(outline).buffer(0).intersection(below)
if flap.geom_type != "Polygon": flap = max(flap.geoms, key=lambda g: g.area)

# triangulate: boundary every STEP mm plus an interior grid
b = np.array(flap.exterior.coords)[:-1]
L = np.linalg.norm(np.diff(b, axis=0, append=b[:1]), axis=1); s = np.r_[0, np.cumsum(L)][:-1]
t = np.arange(0, L.sum(), STEP)
bnd = np.c_[np.interp(t, s, b[:, 0], period=L.sum()), np.interp(t, s, b[:, 1], period=L.sum())]
gy, gz = np.meshgrid(np.arange(*flap.bounds[0::2], STEP), np.arange(*flap.bounds[1::2], STEP))
inner = np.c_[gy.ravel(), gz.ravel()]
inner = inner[[flap.buffer(-STEP * 0.6).contains(Point(p)) for p in inner]]
P2 = np.r_[bnd, inner]
tri = Delaunay(P2).simplices
tri = tri[[flap.contains(Point(c)) for c in P2[tri].mean(1)]]

# drape: x from a ray off +X at the same (y, z), clamped to the attachment height; smoothed
def side_x(y, z):
    loc, idx, _ = ray.intersects_location(np.c_[np.full(len(y), 400.0), y, z], np.tile([-1.0, 0, 0], (len(y), 1)), multiple_hits=False)
    x = np.full(len(y), np.nan); x[idx] = loc[:, 0]; return x
zc = np.minimum(P2[:, 1], np.interp(P2[:, 0], ya, za))
x = side_x(P2[:, 0], zc)
if np.isnan(x).any():
    ok = ~np.isnan(x); x[~ok] = x[ok][cKDTree(P2[ok]).query(P2[~ok])[1]]
e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(P2),) * 2).tocsr()
D = np.asarray(A.sum(1)).ravel()
for _ in range(15): x = np.maximum(x, 0.5 * x + 0.5 * (A @ x) / D)      # smooth, never into the head
X = x + GAP
ears = [trimesh.Trimesh(np.c_[X * sd, P2], tri if sd > 0 else tri[:, ::-1], process=False) for sd in (1, -1)]
trimesh.util.concatenate([T] + ears).export(out)
print(f"preview ears: flap {flap.area / 100:.1f} cm2 real (of {Polygon(outline).area / 100:.1f}), {len(tri)} tris per ear")
