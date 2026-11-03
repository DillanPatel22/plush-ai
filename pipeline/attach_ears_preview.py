"""Review preview only (never used for the pattern): the canonical shape dressed as the finished plush will look,
so it can be compared with the customer photos. Adds the ear flaps hung from their attachment line (standing
off the head a little more towards the tip, as a sewn ear with a pleated base does), eyes and nose at the
positions read from the photos, and yellow-Labrador colouring.

Usage: python3 attach_ears_preview.py canonical.obj ears_dir out.obj face_points.json k
       [--colours vertex_colours.npy --ear-json ear.json] [--seams face_labels.npy]
--colours: per-vertex coat colours (colour_map.py) instead of one flat colour; --seams: face labels of a seam
layout on this same mesh, drawn as slightly darker lines so the seams can be seen on the preview.
Writes out.obj (shape only) and out.ply (vertex colours, for rendering).
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp

def opt(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
pos = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
T = trimesh.load(pos[0], process=False); E = Path(pos[1]); out = Path(pos[2])
fp = json.load(open(pos[3])) if len(pos) > 3 else None; k = float(pos[4]) if len(pos) > 4 else None
VCOL = np.load(opt("--colours")) if opt("--colours") else None
EARJ = json.load(open(opt("--ear-json"))) if opt("--ear-json") else {}
SEAMS = np.load(opt("--seams")) if opt("--seams") else None
outline = np.load(E / "ear_outline_mm.npy")
att_on = np.load(E / "ear_attach_on_template.npy")
GAP0, FLARE, FLARE_MAX, STEP = 3.0, 0.5, 50.0, 3.0
COAT, EAR, EYE, NOSE = (232, 196, 140), (214, 168, 108), (52, 32, 22), (62, 48, 46)
if EARJ.get("ear_rgb"): EAR = tuple(EARJ["ear_rgb"])
ray = trimesh.ray.ray_triangle.RayMeshIntersector(T)

ya, za = att_on[:, 1], att_on[:, 2]
below = Polygon(np.r_[[[ya[0] - 200, za[0]]], np.c_[ya, za], [[ya[-1] + 200, za[-1]], [ya[-1] + 200, 0], [ya[0] - 200, 0]]])
flap = Polygon(outline).buffer(0).intersection(below.buffer(2.0))
if flap.geom_type != "Polygon": flap = max(flap.geoms, key=lambda g: g.area)
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

def side_x(y, z):
    loc, idx, _ = ray.intersects_location(np.c_[np.full(len(y), 400.0), y, z], np.tile([-1.0, 0, 0], (len(y), 1)), multiple_hits=False)
    x = np.full(len(y), np.nan); x[idx] = loc[:, 0]; return x
z_att = np.interp(P2[:, 0], ya, za)
x = side_x(P2[:, 0], np.minimum(P2[:, 1], z_att))
if np.isnan(x).any():
    ok = ~np.isnan(x); x[~ok] = x[ok][cKDTree(P2[ok]).query(P2[~ok])[1]]
e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(P2),) * 2).tocsr()
D = np.asarray(A.sum(1)).ravel()
for _ in range(15): x = np.maximum(x, 0.5 * x + 0.5 * (A @ x) / D)
# a hanging Lab ear lies against the cheek at its front edge and turns forward-out towards its back edge, so
# from the front it frames the face: stand-off grows with distance behind the front edge and below the root
yb = np.array(flap.exterior.coords)
y_front = np.array([yb[np.abs(yb[:, 1] - zz) < 6, 0].min() if (np.abs(yb[:, 1] - zz) < 6).any() else yb[:, 0].min() for zz in P2[:, 1]])
drop = np.clip(z_att - P2[:, 1], 0, None)
X = x + GAP0 + np.minimum(FLARE * np.clip(P2[:, 0] - y_front, 0, None) * np.clip(drop / 30.0, 0, 1), FLARE_MAX)
parts = [(T, COAT)]
for sd in (1, -1):
    ear = trimesh.Trimesh(np.c_[X * sd, P2], tri if sd > 0 else tri[:, ::-1], process=False)
    back = trimesh.Trimesh(np.c_[(X + 2.5) * sd, P2], tri[:, ::-1] if sd > 0 else tri, process=False)
    parts += [(ear, EAR), (back, EAR)]

if fp is not None and k:
    r_e = fp["eye_diam_mm_plush"] / k / 2
    for sd in (1, -1):
        c = np.array(fp["eye_R"]) * [sd, 1, 1]
        cp, _, ti = trimesh.proximity.closest_point(T, c[None]); n = T.face_normals[ti[0]]
        sph = trimesh.creation.icosphere(subdivisions=3, radius=r_e); sph.apply_translation(cp[0] - n * 0.45 * r_e)
        parts.append((sph, EYE))
nose_w = (fp["nose_width_mm_plush"] / k) if (fp is not None and k) else 45.0
# nose leather: a crisp patch cut from a finely subdivided copy of the muzzle tip, lifted 1.5 mm off the surface
V = T.vertices; tip = V[np.argmin(V[:, 1])]
tm_ = trimesh.Trimesh(V, T.faces, process=False)
near = np.where((tm_.triangles_center[:, 1] - tip[1]) < 0.8 * nose_w)[0]
sub = tm_.submesh([near], append=True)
for _ in range(3):
    v_, f_ = trimesh.remesh.subdivide(sub.vertices, sub.faces); sub = trimesh.Trimesh(v_, f_, process=False)
c_ = sub.triangles_center; zc = tip[2] + 0.05 * nose_w
ell = (c_[:, 0] / (0.5 * nose_w)) ** 2 + ((c_[:, 2] - zc) / (0.36 * nose_w)) ** 2
nose_f = np.where((ell < 1.0) & (c_[:, 1] - tip[1] < 0.45 * nose_w) & (sub.face_normals[:, 1] < 0.2))[0]
nose_m = sub.submesh([nose_f], append=True); nose_m.vertices = nose_m.vertices + nose_m.vertex_normals * 1.5
parts.append((nose_m, NOSE))
nose = nose_f

meshes, cols = [], []
for i, (m, c) in enumerate(parts):
    col = np.tile(np.r_[c, 255], (len(m.vertices), 1)).astype(np.uint8)
    if i == 0 and VCOL is not None:
        col[:, :3] = VCOL
    meshes.append(m); cols.append(col)
if SEAMS is not None and VCOL is not None:                       # seams drawn as thin cords just above the fur
    tm = trimesh.Trimesh(T.vertices, T.faces, process=False); vn = tm.vertex_normals
    fa, fe = tm.face_adjacency, tm.face_adjacency_edges
    se = fe[SEAMS[fa[:, 0]] != SEAMS[fa[:, 1]]]
    R_ = 1.3; ring = 6; ang = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    allv, allf = [], []; off = 0
    for a_, b_ in se:
        p0 = T.vertices[a_] + vn[a_] * 0.8; p1 = T.vertices[b_] + vn[b_] * 0.8
        d_ = p1 - p0; L_ = np.linalg.norm(d_)
        if L_ < 1e-6: continue
        d_ /= L_; u_ = np.cross(d_, [0, 0, 1.0]);
        if np.linalg.norm(u_) < 1e-3: u_ = np.cross(d_, [0, 1.0, 0])
        u_ /= np.linalg.norm(u_); w_ = np.cross(d_, u_)
        circ = (np.cos(ang)[:, None] * u_ + np.sin(ang)[:, None] * w_) * R_
        allv += [p0 + circ, p1 + circ]
        for j in range(ring):
            j2 = (j + 1) % ring
            allf += [[off + j, off + j2, off + ring + j], [off + j2, off + ring + j2, off + ring + j]]
        off += 2 * ring
    if allv:
        cord = trimesh.Trimesh(np.vstack(allv), np.array(allf), process=False)
        ccol = np.tile(np.r_[(np.median(VCOL, 0) * 0.62).astype(np.uint8), 255], (len(cord.vertices), 1)).astype(np.uint8)
        meshes.append(cord); cols.append(ccol)
allm = trimesh.util.concatenate(meshes)
allm.export(out)
cm = trimesh.Trimesh(allm.vertices, allm.faces, vertex_colors=np.vstack(cols), process=False)
cm.export(out.with_suffix(".ply"))
print(f"preview: ear flap {flap.area / 100:.1f} cm2 real, eyes {'yes' if fp else 'no'}, nose patch {len(nose)} faces")
