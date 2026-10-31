"""Plush simplification of the canonical template: remove muscle definition from body, legs and tail while
keeping the silhouette, and keep the face lightly smoothed (likeness lives there).

Volume-preserving Taubin smoothing (no shrinking), strong on body/legs/tail, light on the head, blended
along the neck. Soles stay flat, symmetry stays exact, vertex count/order unchanged.

Usage: python3 simplify_plush.py template.obj out.obj [--body_iters 60] [--head_iters 6] [--head_end 230]
"""
import argparse, json
import numpy as np, trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--body_iters", type=int, default=60)
ap.add_argument("--head_iters", type=int, default=6)
ap.add_argument("--head_end", type=float, default=230.0)
a = ap.parse_args()
T = trimesh.load(a.src, process=False)
V0 = np.asarray(T.vertices, float)
tri = trimesh.Trimesh(V0, T.faces, process=False)          # triangulated copy for the smoother


def taubin(iters):
    m = tri.copy()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=iters)
    return np.asarray(m.vertices)

Pb = taubin(a.body_iters); Ph = taubin(a.head_iters)
nose_y = V0[:, 1].min()
w = 1.0 / (1.0 + np.exp(-((nose_y + a.head_end) - V0[:, 1]) / 30.0))          # 1 = head
V = Ph * w[:, None] + Pb * (1 - w[:, None])
# soles: vertices that were on the ground stay on the ground
sole = V0[:, 2] < 0.5
V[sole, 2] = 0.0
V[:, 2] -= V[:, 2].min() if V[:, 2].min() < 0 else 0
# exact symmetry
mirror = cKDTree(V0).query(V0 * [-1, 1, 1])[1]
neg = V0[:, 0] < 0
V[neg] = V[mirror[neg]] * [-1, 1, 1]
V[np.abs(V0[:, 0]) < 1e-6, 0] = 0.0
out = trimesh.load(a.src, process=False); out.vertices = V; out.export(a.dst)
mv = np.linalg.norm(V - V0, axis=1)
rep = {"body_iters": a.body_iters, "head_iters": a.head_iters, "mean_move_mm": round(float(mv.mean()), 2),
       "max_move_mm": round(float(mv.max()), 2), "volume_change_pct": round((out.volume / T.volume - 1) * 100, 2) if T.is_volume else None}
json.dump(rep, open(a.dst.rsplit(".", 1)[0] + "_simplify.json", "w"), indent=1)
print(rep)
