"""Coat colour and markings from the photo-textured 3D model, transferred onto any mesh in the project frame.

The generator's textured model (aligned/aligned_textured.glb, made from the customer's photos) is sampled at the
closest point to every vertex of the target mesh; colours are lightly smoothed (fur texture noise) and saved. The
same colours drive the review preview, the airbrush guide on each pattern piece and the colour-zone swatches.
Fur always starts white (design rule): these colours are what the airbrush/dye artist reproduces.

Usage: python3 colour_map.py aligned_textured.glb target.obj out_prefix [--ear-outline ears/ear_outline_rel.npy]
Writes out_prefix_colours.npy (N x 3 uint8 sRGB per target vertex), out_prefix_colours.ply, out_prefix_ear.json.
"""
import json, sys
import numpy as np, trimesh
import scipy.sparse as sp
from shapely import contains_xy
from shapely.geometry import Polygon

glb, tgt, pre = sys.argv[1], sys.argv[2], sys.argv[3]
scene = trimesh.load(glb, force="scene")
src = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and getattr(g.visual, "kind", None) == "texture"][0]
img = src.visual.material.baseColorTexture.convert("RGB"); tex = np.asarray(img).astype(float); th, tw = tex.shape[:2]
uv = np.asarray(src.visual.uv, float)
T = trimesh.load(tgt, process=False); V = np.asarray(T.vertices, float)
y_shift = V[:, 1].min() - src.vertices[:, 1].min()              # both frames: nose tip at min Y, ground at Z = 0


def sample(P):
    cp, _, tri = trimesh.proximity.closest_point(src, P)
    bc = trimesh.triangles.points_to_barycentric(src.triangles[tri], cp)
    t = (uv[src.faces[tri]] * bc[:, :, None]).sum(1)
    x = np.clip(t[:, 0] * (tw - 1), 0, tw - 1); y = np.clip((1 - t[:, 1]) * (th - 1), 0, th - 1)   # glTF v points up
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); x1, y1 = np.minimum(x0 + 1, tw - 1), np.minimum(y0 + 1, th - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy) + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


col = sample(V - [0, y_shift, 0])
# light smoothing over the mesh graph (removes the generator's painted fur strands, keeps markings)
E = T.edges_unique
A = sp.coo_matrix((np.ones(len(E) * 2), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V),) * 2).tocsr()
D = np.asarray(A.sum(1)).ravel()
for _ in range(3): col = 0.5 * col + 0.5 * (A @ col) / D[:, None]
# Baked shadow: the generator darkens paws and the underside (ground contact). Where the surface is near the ground
# or faces down, lightness is raised to at least 90 % of the body's median, keeping the hue. Real dark socks or
# markings there are rare on these breeds; the review sheet shows the result so the artist can override it.
Tn = trimesh.Trimesh(V, T.faces, process=False); vn = Tn.vertex_normals
under = (V[:, 2] < 45.0) | (vn[:, 2] < -0.5)
lum_ = col.mean(1); floor = 0.9 * np.median(lum_[~under])
low = under & (lum_ < floor)
col[low] = col[low] * (floor / np.maximum(lum_[low], 1))[:, None]
col = np.clip(col, 0, 255)
print(f"shadow-corrected vertices: {int(low.sum())}")
# exact symmetry like the shape (left = right): average each vertex with its mirror
from scipy.spatial import cKDTree
mir = cKDTree(V).query(V * [-1, 1, 1])[1]
col = 0.5 * (col + col[mir])
col8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
np.save(pre + "_colours.npy", col8)
trimesh.Trimesh(V, T.faces, vertex_colors=np.c_[col8, np.full(len(V), 255, np.uint8)], process=False).export(pre + "_colours.ply")

ear = {}
if "--ear-outline" in sys.argv:
    ol = np.load(sys.argv[sys.argv.index("--ear-outline") + 1])
    poly = Polygon(ol + [src.vertices[:, 1].min(), 0]).buffer(-6)
    S = src.vertices; inside = contains_xy(poly, S[:, 1], S[:, 2]) & (S[:, 0] > 0.6 * np.abs(S[:, 0]).max())
    if inside.sum() > 20:
        vc = sample(S[inside]); ear = {"ear_rgb": [int(round(c)) for c in np.median(vc, 0)], "samples": int(inside.sum())}
json.dump(ear, open(pre + "_ear.json", "w"))
lum = col.mean(1)
print(f"colours for {len(V)} vertices; median {np.median(col, 0).round(0)}, ear {ear.get('ear_rgb')}")
print("RESULT:", "PASS" if 30 < np.median(lum) < 250 and len(col8) == len(V) else "FAIL")
